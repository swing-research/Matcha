import copy

import pandas as pd
import starfile
import torch
from ml_collections import ConfigDict

from matcha.core.CrossCorrelationMatcher import CrossCorrelationMatcher
from matcha.core.Matcha import Matcha
from matcha.core.ShiftMatcher import ShiftMatcher
from matcha.utils.io_utils import extract_subtomogram_patch_batch, load_template, mask_effective_radius, store_alignment_parameters
from matcha.utils.rotation_ops import update_rotation_estimate
from matcha.utils.setup_utils import get_prior_shifts, get_rotation_tracker, pad_data, set_random_seed, setup_mask, resolve_precision_mode
from matcha.utils.volume_rotation import rotate_volumes
import warnings

# Avoid displaying numba performance warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings(
    "ignore",
    category=NumbaPerformanceWarning,
)


class AlignmentRuntimeBuilder:
    """Build runtime-only alignment objects from an input config."""

    @staticmethod
    def _build_matcha_config(runtime_config):
        """Build Matcha search settings, using config defaults when needed."""
        if "matcha_config" in runtime_config and runtime_config.matcha_config is not None:
            cfg = dict(runtime_config.matcha_config)
            if "matcha_config_expert" in runtime_config and runtime_config.matcha_config_expert is not None:
                cfg.update(dict(runtime_config.matcha_config_expert))
            return ConfigDict(cfg)

        li = [40]

        return ConfigDict(
            {
                "Li": sorted(set(li)),
                "candidates": int(runtime_config.candidates[-1]) if "candidates" in runtime_config else 10,
                "reinits": int(runtime_config.reinits[-1]) if "reinits" in runtime_config else 1,
                "num_steps": int(runtime_config.newton_steps) if "newton_steps" in runtime_config else 5,
                "step_type": "newton",
                "stop_early": bool(runtime_config.get("newton_stop_early", False)),
                "oversampling_factor_K": 2,
                "do_random_sampling": False,
                "reinits_iterations": 0,
            }
        )

    @staticmethod
    def _resolve_device(gpu_id):
        """Normalize the configured GPU id into a torch device object."""
        if isinstance(gpu_id, str) and gpu_id.startswith("cuda"):
            return torch.device(gpu_id)
        return torch.device(f"cuda:{int(gpu_id)}")

    @staticmethod
    def _resolve_precision(precision):
        """Map precision string to complex and real torch dtypes."""
        if precision == "float32":
            return torch.complex64, torch.float32
        if precision == "float64":
            return torch.complex128, torch.float64
        raise ValueError("Unsupported precision. Use 'float32' or 'float64'.")

    @staticmethod
    def prepare_config(input_config):
        """Prepare runtime config (device, masks, matchers, correlator) from input config."""
        runtime_config = copy.deepcopy(input_config)

        device = AlignmentRuntimeBuilder._resolve_device(runtime_config.gpu_id)
        runtime_config.device = device
        torch.cuda.set_device(device)

        set_random_seed(runtime_config.random_seed)

        execution = ConfigDict()
        execution["shape"] = torch.tensor((runtime_config.N, runtime_config.N, runtime_config.N), device=device, dtype=torch.int64)
        execution["num_templates"] = 1
        execution_dtype, execution_dtype_real = AlignmentRuntimeBuilder._resolve_precision(
            runtime_config.get("precision", "float32")
        )
        execution.dtype = execution_dtype
        execution.dtype_real = execution_dtype_real
        runtime_config.execution = execution

        # setup_mask mutates runtime_config.execution in-place (kept for compatibility)
        setup_mask(runtime_config)

        n = int(runtime_config.N)
        radius = min(int(runtime_config.get("radius", n // 2)), int(n // 2 * 0.95))
        mask_path = str(runtime_config.get("path_template_mask", ""))
        if mask_path:
            radius = min(radius, int(mask_effective_radius(mask_path, n)))
        expansion_epsilon = float(runtime_config.get("expansion_epsilon", 1e-4))
        precision_mode = resolve_precision_mode(runtime_config)
        reduce_memory = bool(runtime_config.reduce_memory)
        micro_batch_split = int(getattr(runtime_config, "micro_batch_split", 2))
        num_subtomograms_per_batch = int(runtime_config.num_subtomograms_per_batch)
        num_base_shifts = 1

        matcha_batchsize = num_subtomograms_per_batch * num_base_shifts
        matcha_config = AlignmentRuntimeBuilder._build_matcha_config(runtime_config)
        search_l_max = max(matcha_config.Li)

        execution.RotationMatcher = Matcha(
            batchsize=matcha_batchsize,
            device=device,
            L_max=search_l_max,
            matcha_config=matcha_config,
        )
        execution.ShiftMatcher = ShiftMatcher(
            config=runtime_config,
            device=device,
            dtype=execution_dtype,
        )

        execution.Correlator = CrossCorrelationMatcher(
            N=n,
            device=device,
            expansion_epsilon=expansion_epsilon,
            batchsize=matcha_batchsize,
            reduce_memory=reduce_memory,
            bandlimit=search_l_max,
            micro_batch_split=micro_batch_split,
            dtype=execution_dtype_real,
            radius=radius,
            jl_zeros_path=runtime_config["jl_zeros_path"],
            cs_path=runtime_config["cs_path"],
            precision_mode=precision_mode,
        )

        return runtime_config


class BatchProcessor:
    """Handle per-batch setup and rotation/shift alternation logic."""

    def __init__(self, runtime_config, df, compute_stream, num_subtomograms_per_batch, num_base_shifts):
        """Initialize processor state used across queue batches."""
        self.runtime_config = runtime_config
        self.execution = runtime_config.execution
        self.df = df
        self.device = runtime_config["device"]
        self.dtype_real = self.execution["dtype_real"]
        self.correlator = self.execution.Correlator
        self.rotation_matcher = self.execution.RotationMatcher
        self.shift_matcher = self.execution.ShiftMatcher
        self.gpu_id = runtime_config.gpu_id
        self.path_output = runtime_config.get("path_output_tmp", runtime_config.path_output)
        self.worker_id = runtime_config.worker_id if "worker_id" in runtime_config else None
        self.num_alternations = int(runtime_config["num_alternations"])
        self.microbatch_size = int(runtime_config.num_subtomograms_per_batch // 2)
        self.compute_stream = compute_stream
        self.num_subtomograms_per_batch = int(num_subtomograms_per_batch)
        self.num_base_shifts = int(num_base_shifts)
        self.result_data = None

    @staticmethod
    def _make_result_dataframe():
        """Create the per-worker result table schema."""
        return pd.DataFrame(
            columns=[
                "path",
                "rotation_score",
                "shift_score",
                "rotation",
                "translation",
                "file_name",
                "grid_shift",
                "prior_shift",
                "alternation_index",
                "rlnTomoParticleName",
                "half",
            ]
        )

    def _ensure_template(self, ref):
        """Load and cache template-dependent matcher state for a new reference path."""
        if "ref" in self.execution and self.execution["ref"] == ref:
            return

        self.result_data = self._make_result_dataframe()
        template = load_template(
            path_template=ref,
            spherical_mask=self.execution["spherical_mask_torch"],
            dtype=self.dtype_real,
            device=self.device,
            path_template_mask=self.runtime_config.path_template_mask,
        )
        self.correlator.set_template(
            template_data=template,
            mask=self.execution["spherical_mask_torch"],
        )
        self.shift_matcher.set_reference(template_data=template)
        self.execution["ref"] = ref

    def _set_output_target(self, half):
        """Set output target fields for this batch half."""
        if self.worker_id is not None:
            self.execution.output_file_name = f"{self.path_output}_{half}_{self.worker_id}"
        else:
            self.execution.output_file_name = f"{self.path_output}_{half}"
        self.execution.half = half

    def _prepare_batch_state(self, ref, half, paths):
        """Prepare template-dependent state plus per-batch trackers and shifts."""
        self._ensure_template(ref)

        if self.result_data is None:
            self.result_data = self._make_result_dataframe()
        self._set_output_target(half)

        tomogram_file_names = paths
        base_shifts = self.shift_matcher.get_base_shifts(
            num_subtomograms_per_batch=self.num_subtomograms_per_batch,
            num_base_shifts=self.num_base_shifts,
            dtype_real=self.dtype_real,
        )

        rotation_tracker = get_rotation_tracker(
            tomogram_file_names=tomogram_file_names,
            df=self.df,
            config=self.runtime_config,
        )
        prior_shifts = get_prior_shifts(
            subtomograms=None,
            tomogram_file_names=tomogram_file_names,
            df=self.df,
            config=self.runtime_config,
        )
        local_shifts = prior_shifts.clone()
        prior_shifts.zero_()

        return tomogram_file_names, base_shifts, local_shifts, rotation_tracker, prior_shifts

    def _extract_rotated_batch(self, subtomograms, current_shift, rotation_tracker):
        """Extract shifted patches and rotate them into the current orientation estimate."""
        batched_tomogram_data = extract_subtomogram_patch_batch(
            volume=subtomograms,
            shift=current_shift,
            config=self.runtime_config,
        )
        return rotate_volumes(
            batched_tomogram_data,
            rotation_tracker,
            microbatch_size=self.microbatch_size,
            permute_before_sample=False,
        )

    def _search_rotation(self, batched_tomogram_data, rotation_tracker):
        """Run rotational search and update the cumulative rotation tracker."""
        coeffs, bh_norms = self.correlator.get_sigma(batched_tomogram_data)
        alphas, betas, gammas, rotation_score = self.rotation_matcher.search_orientations(
            sigma=coeffs,
        )
        rotation_tracker, _ = update_rotation_estimate(
            alphas=alphas.reshape(-1),
            betas=betas.reshape(-1),
            gammas=gammas.reshape(-1),
            rotation_tracker=rotation_tracker,
        )
        return rotation_tracker, rotation_score.reshape(-1), bh_norms

    def _search_shift(
        self,
        subtomos,
        ctfs,
        current_shift,
        rotation_tracker,
        alternation_index,
        ev_ctf,
    ):
        """Run shift search for the current alternation."""
        if alternation_index == 0:
            self.compute_stream.wait_event(ev_ctf)
        return self.shift_matcher.search_shifts(
            subtomos=subtomos,
            ctfs=ctfs,
            shift_zyx=current_shift,
            rotation_tracker=rotation_tracker,
        )

    def _store_batch_results(
        self,
        tomogram_file_names,
        local_shifts,
        rotation_tracker,
        prior_shifts,
        alternation_index,
        half,
        rotation_score,
        bh_norms,
        shift_scores,
    ):
        """Store one alternation's alignment results in the worker dataframe."""
        normalized_rotation_scores = rotation_score / (
            bh_norms.reshape(-1) * self.correlator.bh_norm_template
        ).clamp_min(1e-12)
        store_alignment_parameters(
            config=self.runtime_config,
            result_data=self.result_data,
            tomogram_file_names=tomogram_file_names,
            rotation_scores=normalized_rotation_scores,
            local_shifts=local_shifts,
            rotation_tracker=rotation_tracker,
            prior_shifts=prior_shifts,
            alternation_index=alternation_index,
            half=half,
            shift_scores=shift_scores.reshape(-1),
        )

    def _run_alternations(
        self,
        subtomograms,
        subtomos,
        ctfs,
        ev_ctf,
        tomogram_file_names,
        half,
        base_shifts,
        local_shifts,
        rotation_tracker,
        prior_shifts,
    ):
        """Run rotation/shift alternations for a single prepared batch."""
        for alternation_index in range(self.num_alternations):
            current_shift = prior_shifts + base_shifts + local_shifts

            batched_tomogram_data = self._extract_rotated_batch(
                subtomograms=subtomograms,
                current_shift=current_shift,
                rotation_tracker=rotation_tracker,
            )
            rotation_tracker, rotation_score, bh_norms = self._search_rotation(
                batched_tomogram_data=batched_tomogram_data,
                rotation_tracker=rotation_tracker,
            )
            shift_estimate, shift_scores = self._search_shift(
                subtomos=subtomos,
                ctfs=ctfs,
                current_shift=current_shift,
                rotation_tracker=rotation_tracker,
                alternation_index=alternation_index,
                ev_ctf=ev_ctf,
            )
            local_shifts += shift_estimate.unsqueeze(1)
            self._store_batch_results(
                tomogram_file_names=tomogram_file_names,
                local_shifts=local_shifts,
                rotation_tracker=rotation_tracker,
                prior_shifts=prior_shifts,
                alternation_index=alternation_index,
                half=half,
                rotation_score=rotation_score,
                bh_norms=bh_norms,
                shift_scores=shift_scores,
            )

    def process_batch(self, subtomos, ctfs, ev_sub, ev_ctf, ref, half, paths):
        """Process one queue batch from setup through all alternations."""
        with torch.cuda.stream(self.compute_stream):
            (
                tomogram_file_names,
                base_shifts,
                local_shifts,
                rotation_tracker,
                prior_shifts,
            ) = self._prepare_batch_state(ref=ref, half=half, paths=paths)

        self.compute_stream.wait_event(ev_sub)
        with torch.cuda.stream(self.compute_stream):
            subtomograms = subtomos.clone()

            if subtomograms.shape[0] == 0:
                print("No more subtomograms to process. Exiting.")
                return False

            if subtomograms.shape[0] < self.num_subtomograms_per_batch:
                padding = self.num_subtomograms_per_batch - subtomograms.shape[0]
                subtomograms, ctfs, subtomos, rotation_tracker = pad_data(
                    padding,
                    subtomograms,
                    ctfs,
                    subtomos,
                    rotation_tracker,
                )

            self._run_alternations(
                subtomograms=subtomograms,
                subtomos=subtomos,
                ctfs=ctfs,
                ev_ctf=ev_ctf,
                tomogram_file_names=tomogram_file_names,
                half=half,
                base_shifts=base_shifts,
                local_shifts=local_shifts,
                rotation_tracker=rotation_tracker,
                prior_shifts=prior_shifts,
            )
        return True


class AlignmentJob:
    """Umbrella object that owns queue I/O and batch orchestration."""

    def __init__(self, input_config, in_queue):
        """Create an alignment job and initialize runtime resources."""
        self.runtime_config = AlignmentRuntimeBuilder.prepare_config(input_config)
        self.in_queue = in_queue
        self.device = self.runtime_config["device"]
        self.df = self._load_particles_df()
        self.num_base_shifts = 1
        self.num_subtomograms_per_batch = int(self.runtime_config["num_subtomograms_per_batch"])
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.current_stream(self.device)
        self.batch_processor = BatchProcessor(
            runtime_config=self.runtime_config,
            df=self.df,
            compute_stream=self.compute_stream,
            num_subtomograms_per_batch=self.num_subtomograms_per_batch,
            num_base_shifts=self.num_base_shifts,
        )

    def _load_particles_df(self):
        """Load particle metadata table from the STAR file."""
        df = starfile.read(self.runtime_config.particles_starfile)
        df = df["particles"] if isinstance(df, dict) and "particles" in df else df
        df = df[2] if isinstance(df, list) else df
        return df

    @staticmethod
    def load_batch_from_queue(copy_stream, compute_stream, in_queue, device):
        """Fetch one CPU batch from queue and stage subtomos/ctfs asynchronously on GPU."""
        item = in_queue.get()
        if item is None:
            print("Received termination signal. Exiting alignment loop.")
            return None, None, None, None, None, None, None

        ref, half, paths, vols_cpu, ctfs_cpu = item
        if isinstance(vols_cpu, torch.Tensor) and not vols_cpu.is_pinned():
            vols_cpu = vols_cpu.pin_memory()
        if isinstance(ctfs_cpu, torch.Tensor) and not ctfs_cpu.is_pinned():
            ctfs_cpu = ctfs_cpu.pin_memory()

        with torch.cuda.stream(copy_stream):
            subtomos = vols_cpu.to(device=device, non_blocking=True)
            ev_sub = torch.cuda.Event(blocking=False)
            ev_sub.record(copy_stream)

            ctfs = ctfs_cpu.to(device, non_blocking=True) if ctfs_cpu is not None else None
            ev_ctf = torch.cuda.Event(blocking=False)
            ev_ctf.record(copy_stream)

        subtomos.record_stream(compute_stream)
        ctfs.record_stream(compute_stream) if ctfs is not None else None
        return subtomos, ctfs, ev_sub, ev_ctf, ref, half, paths

    def run(self):
        """Consume queue items until sentinel and process each alignment batch."""
        while True:
            subtomos, ctfs, ev_sub, ev_ctf, ref, half, paths = self.load_batch_from_queue(
                self.copy_stream,
                self.compute_stream,
                self.in_queue,
                self.device,
            )
            if subtomos is None:
                break
            if not self.batch_processor.process_batch(
                subtomos=subtomos,
                ctfs=ctfs,
                ev_sub=ev_sub,
                ev_ctf=ev_ctf,
                ref=ref,
                half=half,
                paths=paths,
            ):
                break
        return


def data_loader(copy_stream, compute_stream, in_queue, device):
    """Compatibility wrapper that delegates batch loading to AlignmentJob."""
    return AlignmentJob.load_batch_from_queue(copy_stream, compute_stream, in_queue, device)


def run_alignment(config, in_queue):
    """Run full alignment using an AlignmentJob instance."""
    AlignmentJob(input_config=config, in_queue=in_queue).run()
    return


def prepare_alignment(config):
    """Compatibility wrapper returning prepared runtime config."""
    return AlignmentRuntimeBuilder.prepare_config(config)


def align(config, in_queue):
    """Top-level worker entrypoint called by the multiprocessing launcher."""
    run_alignment(config, in_queue)
