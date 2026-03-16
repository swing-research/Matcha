import torch, torch.multiprocessing as mp
import numpy as np
import threading
import time
import copy
import os
import sys
import queue as py_queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ml_collections import ConfigDict

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from utils.setup_utils import get_subtomogram_paths_from_particles, setup_data_splits
from utils.io_utils import load_subtomogram_cpu, load_ctf_cpu, load_ctf_relion5_cpu, join_data, join_data_relion_old
from utils.setup_utils import find_mrc_in_selected_subdirs

def _read_worker(p:str, ctf_loader, base_str:str, replace_str:str):
    """Thread worker: load one subtomogram + its CTF with error capture.
    Parameters:
    -p. str, path to subtomogram
    -ctf_loader: function to load the CTF
    -base_str, str to be replaced in path to get CTF path
    -replace_str, str to replace base_str with to get CTF path
    Returns:
    - tuple (p, st_cpu, ctf_cpu, error)
    """
    try:
        st_cpu = load_subtomogram_cpu(p, dtype=np.float32)   # np or torch
        ctf_cpu = ctf_loader(p.replace(base_str, replace_str), transpose=False)  # np
        #st_cpu = np.roll(np.fft.irfftn(np.fft.rfftn(np.roll(st_cpu, shift=(-0,-0,-0), axis=(0,1,2)), norm="forward")* ctf_cpu, norm="forward").real, shift=(0,0,0), axis=(0,1,2))
        return (p, st_cpu, ctf_cpu, None)
    except FileNotFoundError as e:
        return (p, None, None, e)
    except Exception as e:
        # Catch-all so a single bad file doesn't kill the pool
        return (p, None, None, e)

def _to_torch_cpu(x: any):
    """Safely convert numpy/torch to a contiguous CPU float32 tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous()
        # Ensure float32 (change if you want to preserve dtype)
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        return t
    else:
        # assume numpy
        return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float32)
    
 # Helper to flush a batch to a specific queue index
def _flush_batch(
    rr_idx,
    ref,
    half,
    buf_vol,
    buf_ctf,
    buf_paths,
    out_queues,
    stop_event=None,
    put_timeout: float = 0.5,
):
    if not buf_vol:
        return rr_idx
    batch = torch.stack(buf_vol, dim=0).contiguous()
    batch_ctf = torch.stack(buf_ctf, dim=0).contiguous() if buf_ctf[0] is not None else None

    while True:
        if stop_event is not None and stop_event.is_set():
            return rr_idx
        try:
            out_queues[rr_idx].put((ref, half, buf_paths, batch, batch_ctf), timeout=put_timeout)
            break
        except py_queue.Full:
            if stop_event is not None and stop_event.is_set():
                return rr_idx
            continue

  #  print(f"cpu_reader: Dispatched batch of size {len(buf_vol)} to queue {rr_idx} for half {half}. Qsize; {out_queues[rr_idx].qsize()}")
    return (rr_idx + 1) % len(out_queues)

def cpu_reader(
    run_data:dict,
    out_queues:list,
    config: ConfigDict,
    batch_size: int,
    num_workers: int = None,
    unordered_completion: bool = False,
    stop_event=None,
    queue_put_timeout: float = 0.5,
):
    """Compatibility wrapper for class-based CPU reader implementation."""
    AlignmentRunJob.cpu_reader(
        run_data=run_data,
        out_queues=out_queues,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        unordered_completion=unordered_completion,
        stop_event=stop_event,
        queue_put_timeout=queue_put_timeout,
    )



def worker_entry(config: ConfigDict, queue:mp.Queue, stop_event=None):
    """
    Entry point for each worker process.
    Parameters:
    - config: ConfigDict
        Configuration dictionary containing all necessary parameters.
    - queue: mp.Queue
        Multiprocessing queue for inter-process communication.
    """
    from align_subtomograms import align  # must be top-level import

    try:
        align(config, queue)
    except Exception:
        if stop_event is not None:
            stop_event.set()
        raise


class AlignmentRunJob:
    """Orchestrate multiprocessing workers, CPU reading, and output join."""

    def __init__(self, config: ConfigDict):
        """Initialize run-level state used across worker and reader setup."""
        self.config = copy.deepcopy(config)
        self.ctx = mp.get_context("spawn")
        self.num_gpus = len(self.config.gpu_ids)
        self.stop_event = self.ctx.Event()
        self.reader_queue_size = self._resolve_reader_queue_size(self.config)
        self.cpu_reader_workers = self._resolve_reader_workers(self.config)
        self.reader_unordered_completion = bool(self.config.get("reader_unordered_completion", False))
        self.reader_max_inflight = int(self.config.get("reader_max_inflight", max(1, self.cpu_reader_workers * 4)))
        self.config.reader_max_inflight = self.reader_max_inflight
        if self.reader_max_inflight < self.cpu_reader_workers:
            print(
                "[WARN] reader_max_inflight is smaller than cpu_reader_workers "
                f"({self.reader_max_inflight} < {self.cpu_reader_workers}). "
                "This can underutilize reader threads."
            )
        self.queue_put_timeout = float(self.config.get("queue_put_timeout", 0.5))
        self.queues = [self.ctx.Queue(maxsize=self.reader_queue_size) for _ in range(self.num_gpus)]
        self.processes = []
        self.reader = None
        self.run_data = None
        self.start_time = None

    @staticmethod
    def _resolve_reader_queue_size(config: ConfigDict) -> int:
        """Resolve queue maxsize from config with a safe lower bound."""
        queue_size = int(config.get("reader_queue_size", 2))
        return max(1, queue_size)

    @staticmethod
    def _resolve_reader_workers(config: ConfigDict, num_workers: int = None) -> int:
        """Resolve CPU reader workers from explicit arg, config, or SLURM."""
        if num_workers is not None:
            return max(1, int(num_workers))

        cfg_workers = config.get("cpu_reader_workers", None)
        if cfg_workers is not None:
            return max(1, int(cfg_workers))

        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            try:
                return max(1, int(slurm_cpus))
            except ValueError:
                pass

        cpu_count = os.cpu_count() or 8
        return max(1, min(8, cpu_count))

    @staticmethod
    def cpu_reader(
        run_data: dict,
        out_queues: list,
        config: ConfigDict,
        batch_size: int,
        num_workers: int = None,
        unordered_completion: bool = False,
        stop_event=None,
        queue_put_timeout: float = 0.5,
    ):
        """Read/decode on CPU (I/O-bound) with threads and RR dispatch batches to GPU queues."""
        class _LogProgress:
            """Minimal non-interactive progress reporter (stdout-friendly)."""

            def __init__(self, total: int, desc: str = "cpu_reader"):
                self.total = max(0, int(total))
                self.done = 0
                self.desc = desc
                self._next_pct = 1
                print(f"{self.desc}: 0/{self.total} (0%)", file=sys.stdout, flush=True)

            def update(self, step: int = 1):
                self.done += int(step)
                if self.total <= 0:
                    return
                pct = int((100 * self.done) / self.total)
                if pct >= self._next_pct or self.done >= self.total:
                    shown = min(self.done, self.total)
                    print(
                        f"{self.desc}: {shown}/{self.total} ({pct}%)",
                        file=sys.stdout,
                        flush=True,
                    )
                    while self._next_pct <= pct:
                        self._next_pct += 1

            def close(self):
                if self.total > 0 and self.done < self.total:
                    print(
                        f"{self.desc}: {self.done}/{self.total} ({int((100 * self.done) / self.total)}%)",
                        file=sys.stdout,
                        flush=True,
                    )

        progress_bar = None
        try:
            if config.relion_version == "relion5":
                ctf_loader = load_ctf_relion5_cpu
                base_str, replace_str = "_data.", "_weights."
            else:
                ctf_loader = load_ctf_cpu
                base_str, replace_str = "_subtomo", "_ctf"

            total_paths = sum(len(run_data[0][half]["subtomogram_paths"]) for half in run_data[0].keys())
            if tqdm is not None and total_paths > 0 and bool(config.get("show_progress_bar", True)):
                if sys.stdout.isatty():
                    progress_bar = tqdm(
                        total=total_paths,
                        desc="cpu_reader",
                        unit="vol",
                        dynamic_ncols=True,
                        mininterval=float(config.get("progress_mininterval", 1.0)),
                        smoothing=0.0,
                        ascii=os.environ.get("SLURM_JOB_ID") is not None,
                        leave=True,
                        file=sys.stdout,
                        bar_format=(
                            "{desc}: {percentage:3.0f}%|{bar}| "
                            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                        ),
                    )
                else:
                    progress_bar = _LogProgress(total=total_paths, desc="cpu_reader")

            requested_workers = AlignmentRunJob._resolve_reader_workers(config=config, num_workers=num_workers)
            for half in run_data[0].keys():
                if stop_event is not None and stop_event.is_set():
                    break

                paths = run_data[0][half]["subtomogram_paths"]
                ref = run_data[0][half]["ref"]
                if len(paths) == 0:
                    continue

                buf_vol, buf_ctf, buf_paths = [], [], []
                rr = 0
                max_workers = min(requested_workers, len(paths))
                max_inflight = max(
                    max_workers,
                    int(config.get("reader_max_inflight", max_workers * 4)),
                )
                should_stop = False

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for chunk_start in range(0, len(paths), max_inflight):
                        if stop_event is not None and stop_event.is_set():
                            should_stop = True
                            break

                        chunk_paths = paths[chunk_start : chunk_start + max_inflight]
                        futures = [
                            ex.submit(_read_worker, p, ctf_loader, base_str, replace_str)
                            for p in chunk_paths
                        ]
                        future_iterable = as_completed(futures) if unordered_completion else futures

                        for future in future_iterable:
                            if stop_event is not None and stop_event.is_set():
                                should_stop = True
                                break

                            p, st_cpu, ctf_cpu, err = future.result()
                            if progress_bar is not None:
                                progress_bar.update(1)
                            if err is not None or st_cpu is None:
                                print(f"[WARN] Skipping {p}: {err}")
                                continue

                            t_vol = _to_torch_cpu(st_cpu)
                            t_ctf = _to_torch_cpu(ctf_cpu)
                            del st_cpu, ctf_cpu

                            buf_vol.append(t_vol)
                            buf_ctf.append(t_ctf)
                            buf_paths.append(p)

                            if len(buf_vol) == batch_size:
                                rr = _flush_batch(
                                    rr,
                                    ref,
                                    half,
                                    buf_vol,
                                    buf_ctf,
                                    buf_paths,
                                    out_queues,
                                    stop_event=stop_event,
                                    put_timeout=queue_put_timeout,
                                )
                                buf_vol, buf_ctf, buf_paths = [], [], []
                                if stop_event is not None and stop_event.is_set():
                                    should_stop = True
                                    break

                        if should_stop:
                            for pending_future in futures:
                                pending_future.cancel()
                            break

                if should_stop:
                    break
                if buf_vol:
                    rr = _flush_batch(
                        rr,
                        ref,
                        half,
                        buf_vol,
                        buf_ctf,
                        buf_paths,
                        out_queues,
                        stop_event=stop_event,
                        put_timeout=queue_put_timeout,
                    )
                    buf_vol, buf_ctf, buf_paths = [], [], []
        except Exception as e:
            print(f"[ERROR] cpu_reader encountered an error: {e}")
            if stop_event is not None:
                stop_event.set()
        finally:
            if progress_bar is not None:
                progress_bar.close()
            for q in out_queues:
                if stop_event is not None and stop_event.is_set():
                    try:
                        q.put_nowait(None)
                    except py_queue.Full:
                        pass
                else:
                    q.put(None)
            print("cpu_reader: Finished reading all data.")

    def _prepare_run_data(self):
        """Discover subtomogram paths and build per-half data splits."""
        if bool(self.config.get("particle_paths_from_star", False)):
            subtomogram_paths = get_subtomogram_paths_from_particles(self.config.run_data_path)
        else:
            subtomogram_paths = find_mrc_in_selected_subdirs(
                self.config.path_subtomograms,
                self.config.subdirs_subtomograms,
                self.config.subtomograms_file,
            )
        print(f"Found {len(subtomogram_paths)} subtomograms for alignment.")
        subtomogram_paths = subtomogram_paths[:]
        self.run_data = setup_data_splits(self.config, subtomogram_paths)

    def _configure_output_path(self):
        """Set per-run output suffix and print final run configuration."""
        self.start_time = time.time()

        if bool(self.config.get("relion_external_mode", False)):
            output_dir = Path(str(self.config.path_output))
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = output_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            output_stem = str(self.config.get("relion_output_stem", "matcha_particles")).strip()
            if not output_stem:
                output_stem = "matcha_particles"

            self.config.path_output_tmp = str(tmp_dir / output_stem)
            self.config.path_output = str(output_dir / output_stem)
        else:
            time_suffix = int(self.start_time)
            output_stem = Path(str(self.config.path_output)).name
            run_prefix = f"{output_stem}_run_{time_suffix}"

            tmp_dir = Path(str(self.config.get("tmp_dir", "tmp")))
            outputs_dir = Path(str(self.config.get("outputs_dir", "outputs")))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            outputs_dir.mkdir(parents=True, exist_ok=True)

            self.config.path_output_tmp = str(tmp_dir / run_prefix)
            self.config.path_output = str(outputs_dir / run_prefix)

        print(
            f"Reader settings: workers={self.cpu_reader_workers}, "
            f"queue_maxsize={self.reader_queue_size}, "
            f"unordered_completion={self.reader_unordered_completion}, "
            f"max_inflight={self.reader_max_inflight}"
        )

    def _spawn_workers(self):
        """Start one alignment worker process per configured GPU."""
        for i in range(self.num_gpus):
            worker_config = copy.deepcopy(self.config)
            worker_config.gpu_id = self.config.gpu_ids[i]
            worker_config.worker_id = i
            process = self.ctx.Process(target=worker_entry, args=(worker_config, self.queues[i], self.stop_event))
            process.start()
            self.processes.append(process)

    def _start_reader(self):
        """Start background CPU reader thread that feeds all worker queues."""
        self.reader = threading.Thread(
            target=AlignmentRunJob.cpu_reader,
            args=(
                self.run_data,
                self.queues,
                self.config,
                self.config.num_subtomograms_per_batch,
                self.cpu_reader_workers,
                self.reader_unordered_completion,
                self.stop_event,
                self.queue_put_timeout,
            ),
            daemon=False,
        )
        self.reader.start()

    def _wait_for_completion(self):
        """Join reader thread and worker processes in the original order."""
        while self.reader is not None and self.reader.is_alive():
            dead_workers = [process for process in self.processes if process.exitcode not in (None, 0)]
            if dead_workers and not self.stop_event.is_set():
                print("[WARN] Worker terminated before reader finished. Stopping reader early.")
                self.stop_event.set()
                break
            time.sleep(0.2)

        if self.reader is not None:
            self.reader.join()

        if self.stop_event.is_set():
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        for process in self.processes:
            process.join()

    def _join_results(self):
        """Join per-worker outputs and report success status."""
        if self.config.relion_version == "relion5":
            success = join_data(self.config.run_data_path, workers=self.num_gpus, config=self.config)
        else:
            success = join_data_relion_old(self.config.run_data_path, workers=self.num_gpus, config=self.config)

        if success:
            print(f"Done writing to {self.config.path_output+'.star'}")
        else:
            print("Failed to join data.")
        return success

    def run(self):
        """Execute full alignment lifecycle (setup, spawn, stream, join)."""
        self._prepare_run_data()
        self._configure_output_path()
        self._spawn_workers()
        self._start_reader()
        self._wait_for_completion()

        end_time = time.time()
        print(f"Alignment completed in {end_time - self.start_time:.2f} seconds.")
        return self._join_results()


def run_align(config:ConfigDict):
    """
    Run the accelerated alignment algorithm with multiprocessing.
    Parameters:
    - config: ConfigDict
        Configuration dictionary containing all necessary parameters.
    Returns:
    - success: bool
        True if the alignment and data joining were successful, False otherwise.
    """
    return AlignmentRunJob(config=config).run()
