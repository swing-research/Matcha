import gc
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

from matcha.utils.setup_utils import get_subtomogram_paths_from_particles, setup_data_splits
from matcha.utils.io_utils import load_subtomogram_cpu, load_ctf_relion5_cpu, join_data

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

    from matcha.align_subtomograms import align

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


def _is_cuda_oom_error(exc: Exception) -> bool:
    """Return True for PyTorch and CUDA-driver OOM variants."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda_error_out_of_memory" in msg
        or "cudamemoryerror" in msg
        or "cumemalloc" in msg
    )


def _round_batch_size_for_config(config: ConfigDict, batch_size: int) -> int:
    """Round down to a valid multiple of micro_batch_split."""
    micro_batch_split = max(1, int(config.get("micro_batch_split", 2)))
    batch_size = max(micro_batch_split, int(batch_size))
    return max(micro_batch_split, (batch_size // micro_batch_split) * micro_batch_split)


def _probe_batch_size_impl(config_dict: dict, probe_path: str, batch_size: int) -> dict:
    """Run one dry-run GPU probe and report whether the candidate batch size is safe."""
    from matcha.align_subtomograms import AlignmentJob
    from matcha.utils.io_utils import extract_subtomogram_patch_batch
    from matcha.utils.rotation_ops import update_rotation_estimate
    from matcha.utils.setup_utils import get_prior_shifts, get_rotation_tracker
    from matcha.utils.volume_rotation import rotate_volumes

    config = ConfigDict(copy.deepcopy(config_dict))
    batch_size = _round_batch_size_for_config(config, batch_size)
    probe_gpu_id = config.gpu_ids[0]
    probe_device = torch.device(f"cuda:{int(probe_gpu_id)}")
    safety = float(config.get("auto_batch_safety", 0.80))

    torch.cuda.set_device(probe_device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(probe_device)
    torch.cuda.reset_peak_memory_stats()

    free_start, total = torch.cuda.mem_get_info(probe_device)
    reserve_target = free_start * max(0.0, 1.0 - safety)
    min_free = free_start
    min_stage = "start"
    stage = "runtime_init"

    def _mark(label: str):
        nonlocal min_free, min_stage
        torch.cuda.synchronize(probe_device)
        free_now, _ = torch.cuda.mem_get_info(probe_device)
        if free_now < min_free:
            min_free = free_now
            min_stage = label

    job = runtime = subtomos = ctfs = subtomograms = current_shift = None
    base_shifts = local_shifts = prior_shifts = coeffs = bh_norms = None
    rotation_score = shift_estimate = shift_scores = None
    batched_data = None
    rotation_tracker = None
    alphas = betas = gammas = None
    try:
        probe_dict = config.to_dict()
        probe_dict["gpu_id"] = probe_gpu_id
        probe_dict["num_subtomograms_per_batch"] = batch_size
        probe_config = ConfigDict(probe_dict)

        job = AlignmentJob(input_config=probe_config, in_queue=None)
        runtime = job.runtime_config
        _mark("runtime_init")

        N = int(runtime.N)
        dtype_r = runtime.execution.dtype_real
        probe_template = np.zeros((N, N, N), dtype=np.float32)

        stage = "template_setup"
        runtime.execution.Correlator.set_template(probe_template)
        runtime.execution.ShiftMatcher.set_reference(probe_template)
        _mark("template_setup")

        stage = "input_alloc"
        subtomos = torch.zeros(batch_size, N, N, N, device=probe_device, dtype=dtype_r)
        ctfs = (
            torch.zeros(batch_size, N, N, N // 2 + 1, device=probe_device, dtype=dtype_r)
            if bool(probe_config.get("do_ctf_correction", True))
            else None
        )
        _mark("input_alloc")

        tomogram_file_names = [probe_path] * batch_size
        base_shifts = runtime.execution.ShiftMatcher.get_base_shifts(
            num_subtomograms_per_batch=batch_size,
            num_base_shifts=1,
            dtype_real=dtype_r,
        )
        rotation_tracker = get_rotation_tracker(
            tomogram_file_names=tomogram_file_names,
            df=job.df,
            config=runtime,
        )
        prior_shifts = get_prior_shifts(
            subtomograms=None,
            tomogram_file_names=tomogram_file_names,
            df=job.df,
            config=runtime,
        )
        local_shifts = prior_shifts.clone()
        prior_shifts.zero_()
        _mark("tracker_setup")

        stage = "clone"
        subtomograms = subtomos.clone()
        _mark("clone")

        stage = "extract_rotate"
        current_shift = prior_shifts + base_shifts + local_shifts
        batched_data = extract_subtomogram_patch_batch(
            volume=subtomograms,
            shift=current_shift,
            config=runtime,
            normalize=False,
        )
        batched_data = rotate_volumes(
            batched_data,
            rotation_tracker,
            microbatch_size=max(1, batch_size // 2),
            permute_before_sample=False,
        )
        _mark("extract_rotate")

        stage = "rotation_search"
        coeffs, bh_norms = runtime.execution.Correlator.get_sigma(batched_data)
        del batched_data
        batched_data = None
        alphas, betas, gammas, rotation_score = runtime.execution.RotationMatcher.search_orientations(
            sigma=coeffs,
        )
        rotation_tracker, _ = update_rotation_estimate(
            alphas=alphas.reshape(-1),
            betas=betas.reshape(-1),
            gammas=gammas.reshape(-1),
            rotation_tracker=rotation_tracker,
        )
        _mark("rotation_search")

        stage = "shift_search"
        shift_estimate, shift_scores = runtime.execution.ShiftMatcher.search_shifts(
            subtomos=subtomos,
            ctfs=ctfs,
            shift_zyx=current_shift,
            rotation_tracker=rotation_tracker,
        )
        local_shifts += shift_estimate.unsqueeze(1)
        _mark("shift_search")

        safe = min_free >= reserve_target
        return {
            "batch_size": batch_size,
            "safe": safe,
            "oom": False,
            "probe_error": False,
            "stage": min_stage,
            "min_free": min_free,
            "reserve_target": reserve_target,
            "free_start": free_start,
            "total": total,
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }
    except Exception as exc:
        return {
            "batch_size": batch_size,
            "safe": False,
            "oom": _is_cuda_oom_error(exc),
            "probe_error": True,
            "stage": stage,
            "min_free": min_free,
            "reserve_target": reserve_target,
            "free_start": free_start,
            "total": total,
            "error": str(exc).splitlines()[0],
        }
    finally:
        del job, runtime, subtomos, ctfs, subtomograms
        del current_shift, base_shifts, local_shifts, prior_shifts
        del coeffs, bh_norms, rotation_score, shift_estimate, shift_scores, batched_data
        del rotation_tracker, alphas, betas, gammas
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(probe_device)
        except Exception:
            pass


def _probe_batch_size_worker(config_dict: dict, probe_path: str, batch_size: int, conn) -> None:
    """Run a probe in a child process so CUDA failures do not poison the parent context."""
    result = _probe_batch_size_impl(config_dict, probe_path, batch_size)
    try:
        conn.send(result)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        os._exit(0)


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

    def _round_batch_size(self, batch_size: int) -> int:
        """Round down to a valid multiple of micro_batch_split."""
        micro_batch_split = max(1, int(self.config.get("micro_batch_split", 2)))
        batch_size = max(micro_batch_split, int(batch_size))
        return max(micro_batch_split, (batch_size // micro_batch_split) * micro_batch_split)

    def _is_cuda_oom(self, exc: Exception) -> bool:
        """Return True for PyTorch and CUDA-driver OOM variants."""
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cuda_error_out_of_memory" in msg
            or "cudamemoryerror" in msg
            or "cumemalloc" in msg
        )

    def _legacy_batch_size_hint(self) -> int:
        """Keep the old linear estimator only as a starting hint for validation."""
        from matcha.align_subtomograms import AlignmentRuntimeBuilder
        from matcha.utils.io_utils import extract_subtomogram_patch_batch
        from matcha.utils.volume_rotation import rotate_volumes
        import quaternionic

        probe_gpu_id = self.config.gpu_ids[0]
        probe_device = torch.device(f"cuda:{int(probe_gpu_id)}")
        torch.cuda.set_device(probe_device)

        micro_batch_split = int(self.config.get("micro_batch_split", 2))
        probe_sizes = (micro_batch_split, 2 * micro_batch_split)

        mems = []
        for bs in probe_sizes:
            probe_dict = self.config.to_dict()
            probe_dict["gpu_id"] = probe_gpu_id
            probe_dict["num_subtomograms_per_batch"] = bs
            probe_config = ConfigDict(probe_dict)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(probe_device)
            free_base, _total = torch.cuda.mem_get_info(probe_device)

            runtime = fake_sub = fake_ctf = fake_shift = fake_rot = None
            try:
                runtime = AlignmentRuntimeBuilder.prepare_config(probe_config)
                N = int(runtime.N)
                dtype_r = runtime.execution.dtype_real

                fake_template_np = np.zeros((N, N, N), dtype=np.float32)
                runtime.execution.Correlator.set_template(fake_template_np)
                runtime.execution.ShiftMatcher.set_reference(fake_template_np)

                fake_sub = torch.zeros(bs, N, N, N, device=probe_device, dtype=dtype_r)
                fake_ctf = (
                    torch.zeros(bs, N, N, N // 2 + 1, device=probe_device, dtype=dtype_r)
                    if bool(probe_config.get("do_ctf_correction", True))
                    else None
                )
                fake_shift = torch.zeros(bs, 1, 3, device=probe_device, dtype=dtype_r)
                fake_rot = quaternionic.array(np.tile([[1.0, 0.0, 0.0, 0.0]], (bs, 1)))

                with torch.no_grad():
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
                    coeffs, bh_norms = runtime.execution.Correlator.get_sigma(batched_data)
                    del batched_data
                    runtime.execution.RotationMatcher.search_orientations(sigma=coeffs)
                    del coeffs, bh_norms
                    runtime.execution.ShiftMatcher.search_shifts(
                        subtomos=fake_sub,
                        ctfs=fake_ctf,
                        shift_zyx=fake_shift,
                        rotation_tracker=fake_rot,
                    )

                torch.cuda.synchronize(probe_device)
                free_after, _ = torch.cuda.mem_get_info(probe_device)
                mems.append(free_base - free_after)

            except Exception as exc:
                if self._is_cuda_oom(exc):
                    torch.cuda.empty_cache()
                    return probe_sizes[0]
                raise
            finally:
                del runtime, fake_sub, fake_ctf, fake_shift, fake_rot
                gc.collect()
                torch.cuda.empty_cache()

        mem_lo, mem_hi = mems
        per_sample = (mem_hi - mem_lo) / micro_batch_split
        fixed = mem_lo - micro_batch_split * per_sample

        free, _total = torch.cuda.mem_get_info(probe_device)
        if per_sample <= 0:
            return probe_sizes[0]

        safety = float(self.config.get("auto_batch_safety", 0.80))
        max_batch = int((free * safety - max(0, fixed)) / per_sample)
        return self._round_batch_size(max_batch)

    def _pick_probe_particle_path(self) -> str | None:
        """Choose one real particle path so probe metadata matches the run."""
        if not self.run_data:
            return None

        for run_group in self.run_data:
            if not isinstance(run_group, dict):
                continue
            for half_data in run_group.values():
                if not isinstance(half_data, dict):
                    continue
                paths = list(half_data.get("subtomogram_paths", []) or [])
                if paths:
                    return paths[0]
        return None

    def _log_probe_result(self, result: dict) -> None:
        """Print one concise line for a validated candidate batch size."""
        gib = 1 << 30
        bs = result["batch_size"]
        if result["safe"]:
            print(
                f"[auto_batch] probe batch_size={bs}: min_free {result['min_free']/gib:.1f} GiB "
                f"(target {result['reserve_target']/gib:.1f} GiB, stage={result['stage']}) -> ok"
            )
            return

        if result.get("probe_error", False):
            detail = result.get("error", "probe failed")
            print(
                f"[auto_batch] probe batch_size={bs}: failure at {result['stage']} -> too large "
                f"({detail})"
            )
            return

        if result.get("oom", False):
            detail = result.get("error", "CUDA OOM")
            print(
                f"[auto_batch] probe batch_size={bs}: OOM at {result['stage']} -> too large "
                f"({detail})"
            )
            return

        print(
            f"[auto_batch] probe batch_size={bs}: min_free {result['min_free']/gib:.1f} GiB "
            f"below target {result['reserve_target']/gib:.1f} GiB (stage={result['stage']}) -> too large"
        )

    def _probe_batch_size(self, batch_size: int, probe_path: str) -> dict:
        """Run one candidate probe in an isolated subprocess."""
        batch_size = self._round_batch_size(batch_size)
        timeout_s = float(self.config.get("auto_batch_probe_timeout_s", 300.0))

        recv_conn, send_conn = self.ctx.Pipe(duplex=False)
        process = self.ctx.Process(
            target=_probe_batch_size_worker,
            args=(self.config.to_dict(), probe_path, batch_size, send_conn),
        )
        process.start()
        send_conn.close()

        result = None
        timed_out = False
        try:
            if recv_conn.poll(timeout_s):
                result = recv_conn.recv()
            else:
                timed_out = True
        finally:
            try:
                recv_conn.close()
            except Exception:
                pass

            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join()

        if result is not None:
            return result

        error = (
            f"probe subprocess timed out after {timeout_s:.0f}s"
            if timed_out
            else f"probe subprocess exited with code {process.exitcode}"
        )
        return {
            "batch_size": batch_size,
            "safe": False,
            "oom": False,
            "probe_error": True,
            "stage": "probe_process",
            "min_free": 0.0,
            "reserve_target": 0.0,
            "free_start": 0.0,
            "total": 0.0,
            "error": error,
        }

    def _find_batch_size(self) -> int:
        """Validate candidate batch sizes with an upward search from a small known-safe start."""
        micro_batch_split = max(1, int(self.config.get("micro_batch_split", 2)))
        probe_path = self._pick_probe_particle_path()
        if probe_path is None:
            fallback = self._legacy_batch_size_hint()
            print(
                "[auto_batch] Could not pick a representative particle path; "
                f"falling back to heuristic batch_size={fallback}."
            )
            return fallback

        default_start = max(4, micro_batch_split)
        start_batch = self._round_batch_size(
            int(self.config.get("auto_batch_probe_start", default_start))
        )
        start_batch = max(micro_batch_split, start_batch)
        tested = {}

        def _probe(bs: int) -> dict:
            bs = self._round_batch_size(bs)
            result = tested.get(bs)
            if result is None:
                result = self._probe_batch_size(bs, probe_path)
                tested[bs] = result
                self._log_probe_result(result)
            return result

        lower_ok = None
        upper_bad = None

        first = _probe(start_batch)
        if first["safe"]:
            lower_ok = first["batch_size"]
        else:
            if start_batch != micro_batch_split:
                fallback = _probe(micro_batch_split)
                if fallback["safe"]:
                    lower_ok = fallback["batch_size"]
                    upper_bad = start_batch
            if lower_ok is None:
                print(
                    "[auto_batch] Even the smallest validated probe was tight or failed; "
                    f"using batch_size={micro_batch_split}."
                )
                return micro_batch_split

        if upper_bad is None:
            candidate = max(lower_ok + micro_batch_split, lower_ok * 2)
            while True:
                result = _probe(candidate)
                if result["safe"]:
                    lower_ok = result["batch_size"]
                    candidate = max(lower_ok + micro_batch_split, lower_ok * 2)
                    continue
                upper_bad = result["batch_size"]
                break

        while upper_bad - lower_ok > micro_batch_split:
            mid = self._round_batch_size((lower_ok + upper_bad) // 2)
            if mid <= lower_ok:
                break
            result = _probe(mid)
            if result["safe"]:
                lower_ok = result["batch_size"]
            else:
                upper_bad = result["batch_size"]

        print(f"[auto_batch] Selected verified batch_size = {lower_ok}")
        return lower_ok

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
