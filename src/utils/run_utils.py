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
from utils.io_utils import load_subtomogram_cpu, load_ctf_relion5_cpu, join_data

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
        st_cpu = load_subtomogram_cpu(p, dtype=np.float32)
        ctf_cpu = ctf_loader(p.replace(base_str, replace_str), transpose=False) if ctf_loader is not None else None
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
    import signal

    def _sigterm_handler(signum, frame):
        print(f"[ERROR] GPU worker (device {config.gpu_id}) received SIGTERM.", flush=True)
        try:
            dev = torch.device(f"cuda:{int(config.gpu_id)}")
            free, total = torch.cuda.mem_get_info(dev)
            used_pct = (total - free) / total * 100
            print(
                f"[ERROR] GPU memory at termination: "
                f"{(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB used ({used_pct:.0f}%).",
                flush=True,
            )
            if used_pct > 90:
                print(
                    "[ERROR] GPU was nearly full — likely an out-of-memory condition. "
                    "Try lowering 'auto_batch_safety' in config.yaml.",
                    flush=True,
                )
        except Exception:
            pass
        if stop_event is not None:
            stop_event.set()
        sys.exit(2)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    from align_subtomograms import align  # must be top-level import

    try:
        align(config, queue)
    except Exception as e:
        if stop_event is not None:
            stop_event.set()
        if isinstance(e, torch.cuda.OutOfMemoryError) or (
            isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
        ):
            print(f"[ERROR] GPU worker ran out of memory: {e}", flush=True)
            print("[ERROR] Try lowering 'auto_batch_safety' in config.yaml.", flush=True)
            sys.exit(2)  # distinct exit code for OOM
        raise


class AlignmentRunJob:
    """Orchestrate multiprocessing workers, CPU reading, and output join."""

    def __init__(self, config: ConfigDict):
        """Initialize run-level state used across worker and reader setup."""
        raw = config.to_dict()
        self._auto_batch = str(raw.get("num_subtomograms_per_batch", "")).lower() == "auto"
        if self._auto_batch:
            raw["num_subtomograms_per_batch"] = 1  # placeholder; replaced before workers spawn
        self.config = ConfigDict(raw)
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

            def __init__(self, total: int, desc: str = "matcha"):
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
            if bool(config.get("do_ctf_correction", True)):
                ctf_loader = load_ctf_relion5_cpu
                base_str, replace_str = "_data.", "_weights."
            else:
                ctf_loader = None
                base_str, replace_str = "", ""

            total_paths = sum(len(run_data[0][half]["subtomogram_paths"]) for half in run_data[0].keys())
            if tqdm is not None and total_paths > 0 and bool(config.get("show_progress_bar", True)):
                if sys.stdout.isatty():
                    progress_bar = tqdm(
                        total=total_paths,
                        desc="matcha",
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
                    progress_bar = _LogProgress(total=total_paths, desc="matcha")

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
            print(f"[ERROR] matcha: reader encountered an error: {e}")
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
            if stop_event is not None and stop_event.is_set():
                print("matcha: Stopped early — a GPU worker failed.")
            else:
                print("matcha: Finished reading all data.")

    def _prepare_run_data(self):
        """Discover subtomogram paths and build per-half data splits."""
        subtomogram_paths = get_subtomogram_paths_from_particles(self.config.particles_starfile)
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

    def _find_batch_size(self) -> int:
        """Probe GPU peak memory with a real forward pass, then extrapolate.

        Probes at batch_size = S and 2S (where S = micro_batch_split) so that
        both the rotation correlator (full-batch) and the shift matcher
        (internally micro-batched at bs//2) scale between the two runs.
        Fake zero inputs are used; a fake Fourier reference is injected directly
        into ShiftMatcher so no template file is required.

            max_batch = (free_vram * safety - fixed) / per_sample

        The safety margin is read from config key 'auto_batch_safety' (default 0.80).
        """
        from align_subtomograms import AlignmentRuntimeBuilder
        from utils.io_utils import extract_subtomogram_patch_batch
        from utils.volume_rotation import rotate_volumes
        import quaternionic

        probe_gpu_id = self.config.gpu_ids[0]
        probe_device = torch.device(f"cuda:{int(probe_gpu_id)}")
        torch.cuda.set_device(probe_device)

        micro_batch_split = int(self.config.get("micro_batch_split", 2))
        # Probe at S and 2S so ShiftMatcher.micro_batch_size (= bs//2) also scales.
        probe_sizes = (micro_batch_split, 2 * micro_batch_split)

        mems = []
        for bs in probe_sizes:
            probe_dict = self.config.to_dict()
            probe_dict["gpu_id"] = probe_gpu_id
            probe_dict["num_subtomograms_per_batch"] = bs
            probe_config = ConfigDict(probe_dict)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(probe_device)
            # Use mem_get_info so non-PyTorch CUDA allocations (NUFFT, cuFFT plans, etc.)
            # are included in the measurement — max_memory_allocated misses these.
            free_base, _total = torch.cuda.mem_get_info(probe_device)

            runtime = fake_sub = fake_ctf = fake_shift = fake_rot = None
            try:
                runtime = AlignmentRuntimeBuilder.prepare_config(probe_config)
                N = int(runtime.N)
                dtype_c = runtime.execution.dtype
                dtype_r = runtime.execution.dtype_real

                # Inject fake template state so both matchers can run without real files.
                fake_template_np = np.zeros((N, N, N), dtype=np.float32)
                runtime.execution.Correlator.set_template(fake_template_np)
                runtime.execution.ShiftMatcher.reference_data_fourier = torch.zeros(
                    1, N, N, N, device=probe_device, dtype=dtype_c
                )

                fake_sub = torch.zeros(bs, N, N, N, device=probe_device, dtype=dtype_r)
                fake_ctf = (
                    torch.zeros(bs, N, N, N // 2 + 1, device=probe_device, dtype=dtype_r)
                    if bool(probe_config.get("do_ctf_correction", True))
                    else None
                )
                fake_shift = torch.zeros(bs, 1, 3, device=probe_device, dtype=dtype_r)
                fake_rot = quaternionic.array(np.tile([[1.0, 0.0, 0.0, 0.0]], (bs, 1)))

                with torch.no_grad():
                    # Step 1: patch extraction + rotation (like _extract_rotated_batch)
                    # normalize=False avoids div-by-zero on zero input
                    batched_data = extract_subtomogram_patch_batch(
                        volume=fake_sub,
                        shift=fake_shift,
                        config=runtime,
                        normalize=False,
                    )
                    microbatch_size = max(1, bs // 2)
                    batched_data = rotate_volumes(
                        batched_data,
                        fake_rot,
                        microbatch_size=microbatch_size,
                        permute_before_sample=False,
                    )

                    # Step 2: rotation search (full batch)
                    coeffs, bh_norms = runtime.execution.Correlator.get_sigma(batched_data)
                    del batched_data
                    runtime.execution.RotationMatcher.search_orientations(sigma=coeffs)
                    del coeffs, bh_norms

                    # Step 3: shift search (internally micro-batched at bs//2)
                    runtime.execution.ShiftMatcher.search_shifts(
                        subtomos=fake_sub,
                        ctfs=fake_ctf,
                        shift_zyx=fake_shift,
                        rotation_tracker=fake_rot,
                    )

                torch.cuda.synchronize(probe_device)
                free_after, _ = torch.cuda.mem_get_info(probe_device)
                # Memory consumed by this probe = baseline free minus current free.
                # This captures PyTorch tensors + NUFFT/cuFFT allocations uniformly.
                mems.append(free_base - free_after)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(
                    f"[auto_batch] OOM during probe at batch_size={bs}; "
                    f"falling back to batch_size={probe_sizes[0]}."
                )
                return probe_sizes[0]
            finally:
                del runtime, fake_sub, fake_ctf, fake_shift, fake_rot
                torch.cuda.empty_cache()

        mem_lo, mem_hi = mems
        # Each probe step adds micro_batch_split particles, so per-sample cost is:
        per_sample = (mem_hi - mem_lo) / micro_batch_split
        fixed = mem_lo - micro_batch_split * per_sample

        free, total = torch.cuda.mem_get_info(probe_device)

        if per_sample <= 0:
            print("[auto_batch] Per-sample memory estimate is non-positive; "
                  f"using batch_size={probe_sizes[0]}.")
            return probe_sizes[0]

        safety = float(self.config.get("auto_batch_safety", 0.80))
        max_batch = max(micro_batch_split, int((free * safety - max(0, fixed)) / per_sample))
        # Round down to a multiple of micro_batch_split so all micro-batch
        # chunks are equal size (avoids buffer mismatch in CrossCorrelationMatcher).
        max_batch = max(micro_batch_split, (max_batch // micro_batch_split) * micro_batch_split)

        gib = 1 << 30
        print(
            f"[auto_batch] GPU {probe_gpu_id}: {free/gib:.1f}/{total/gib:.1f} GiB free — "
            f"fixed {fixed/gib:.2f} GiB, per-sample {per_sample/1e6:.1f} MB → "
            f"batch_size = {max_batch}"
        )
        return max_batch

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
                gpu_ids = [self.config.gpu_ids[self.processes.index(p)] for p in dead_workers]
                print(f"[ERROR] GPU worker(s) on device(s) {gpu_ids} crashed. Stopping reader.")
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
        success = join_data(self.config.particles_starfile, workers=self.num_gpus, config=self.config)

        if success:
            print(f"Done writing to {self.config.path_output+'.star'}")
        else:
            print("Failed to join data.")
        return success

    def run(self):
        """Execute full alignment lifecycle (setup, spawn, stream, join)."""
        self._prepare_run_data()
        self._configure_output_path()
        if self._auto_batch:
            self.config.num_subtomograms_per_batch = self._find_batch_size()
        self._spawn_workers()
        self._start_reader()
        self._wait_for_completion()

        end_time = time.time()
        failed = [p for p in self.processes if p.exitcode not in (None, 0)]
        if failed:
            gpu_ids = [self.config.gpu_ids[i] for i, p in enumerate(self.processes) if p.exitcode not in (None, 0)]
            exit_codes = [p.exitcode for p in failed]
            print(f"[ERROR] Alignment failed — GPU worker(s) on device(s) {gpu_ids} exited with error (exit codes: {exit_codes}).")
            if any(p.exitcode == 2 for p in failed):
                print("[ERROR] GPU ran out of memory. Try lowering 'auto_batch_safety' in config.yaml.")
            elif any(p.exitcode in (-9, -15) for p in failed):
                print("[ERROR] Worker(s) received a termination signal. This is usually caused by a scheduler "
                      "time or memory limit (e.g. SLURM walltime/mem), or the job was cancelled externally. "
                      "If this correlates with a higher 'auto_batch_safety', the SLURM RAM limit may be too low "
                      "for the requested batch size — try lowering 'auto_batch_safety' in config.yaml or "
                      "requesting more memory in your SLURM submission script (--mem / --mem-per-cpu).")
        else:
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
