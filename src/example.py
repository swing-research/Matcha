import torch
import numpy as np
import mrcfile
import os
import yaml
import ml_collections
import quaternionic
import time
import json
import argparse

from utils.volume_ops import get_spherical_mask, normalise
from utils.setup_utils import resolve_precision_mode
from utils.rotation_ops import sample_rotations_around, update_rotation_estimate
from core.CrossCorrelationMatcher import CrossCorrelationMatcher
from core.Matcha import Matcha
from core.SOFFT import SOFFT
from utils.volume_rotation import rotate_volumes
import warnings

# Avoid displaying numba performance warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings(
    "ignore",
    category=NumbaPerformanceWarning,
)


class SOFFTMatcher:
    """Lightweight wrapper to expose a Matcha-like `search_orientations` API."""

    def __init__(self, batchsize: int, device: torch.device, L_max: int, oversampling_factor_K: int = 2):
        self.sofft = SOFFT(
            L=L_max,
            device=device,
            batchsize=batchsize,
            oversampling_factor=oversampling_factor_K,
        )

    def search_orientations(self, sigma: torch.Tensor):
        grid_scores = self.sofft.eval(sigma)
        batchsize, num_betas, num_alphas, num_gammas = grid_scores.shape

        flat_scores = grid_scores.reshape(batchsize, -1)
        best_scores, flat_idx = torch.max(flat_scores, dim=1)

        beta_idx = flat_idx // (num_alphas * num_gammas)
        rem = flat_idx % (num_alphas * num_gammas)
        alpha_idx = rem // num_gammas
        gamma_idx = rem % num_gammas

        ids = torch.stack((beta_idx, alpha_idx, gamma_idx), dim=1).unsqueeze(1)
        alphas, betas, gammas = self.sofft.ids_to_angles(ids=ids, shape=grid_scores[0].shape)
        return alphas.squeeze(1), betas.squeeze(1), gammas.squeeze(1), best_scores


def timed_search_orientations(matcher, sigma, warmup_runs: int, timed_runs: int, device: torch.device):
    warmup_runs = max(0, int(warmup_runs))
    timed_runs = max(1, int(timed_runs))

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = matcher.search_orientations(sigma=sigma)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings_ms = []
        result = None
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            result = matcher.search_orientations(sigma=sigma)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return result, np.asarray(timings_ms, dtype=np.float64)


def add_awgn(image: torch.Tensor, snr_db, mask=None, clip: bool = True) -> torch.Tensor:
    orig_dtype = image.dtype
    device = image.device

    img = image.to(torch.float32)
    B = img.shape[0]

    if mask is not None:
        # (D,H,W) mask -> expand to (B,D,H,W)
        mask = mask.to(torch.bool)
        mask_b = mask.expand(B, *mask.shape)
        # Extract masked voxels for each sample
        signal_vals = img[mask_b].view(B, -1)
    else:
        signal_vals = img.view(B, -1)

    # Signal power (mean squared per sample)
    signal_power = (signal_vals ** 2).mean(dim=1)

    # Desired SNR
    snr_db_t = torch.as_tensor(snr_db, dtype=img.dtype, device=device)
    snr_linear = 10.0 ** (snr_db_t / 10.0)

    # Noise power per sample
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power).view(B, *([1] * (img.ndim - 1)))

    # Gaussian noise
    noise = torch.randn_like(img) * noise_std
    noisy = img + noise

    # Restore dtype
    if orig_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        if clip:
            info = torch.iinfo(orig_dtype)
            noisy = noisy.clamp(info.min, info.max)
        noisy = noisy.to(orig_dtype)
    else:
        noisy = noisy.to(orig_dtype)

    return noisy


def get_target_volumes(volume, mask, dtype, device, batchsize):
    volume = normalise(torch.from_numpy(volume.copy()).to(device = device, 
                                                                dtype=dtype), mask=mask) 
    volume = volume * mask
    return volume.unsqueeze(0).expand(batchsize, -1, -1, -1)
    

def run_alignment(config):
    # GPU available
    device = torch.device("cuda:0")
    seed = config.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batchsize = config.example_config.batchsize
    number_subtomograms = config.example_config.number_subtomograms
    timing_cfg = config.orientation_search.timing
    warmup_runs = int(timing_cfg.warmup_runs)
    timed_runs = int(timing_cfg.timed_runs)
    print_per_batch = bool(timing_cfg.print_per_batch)
    search_timings_ms = []

    # Set template data
    with mrcfile.open(config["path_template"]) as mrc:
        volume = mrc.data.copy()
    config.execution.Correlator.set_template(volume,config.execution["spherical_mask_torch"])

    # sample target rotations once
    rotation_samples = sample_rotations_around(
                    base_quat=quaternionic.array([1, 0, 0, 0]),
                    n_samples=number_subtomograms,
                    max_angle=180.0,
                )

    # create mask and move to device
    mask = get_spherical_mask(
        (config.N, config.N, config.N),
        radius=config.example_config.masking_radius
        )
    mask = torch.from_numpy(mask).to(device=device, dtype=config.execution["dtype_real"])
    
    # load target volumes
    target_volumes = get_target_volumes(
        volume = volume, 
        mask = mask, 
        dtype = config.execution["dtype_real"], 
        device = device,
        batchsize = batchsize)

    dists = []
    for batch_iter in range(0, number_subtomograms, batchsize):

        # get rotations for the batch
        rotations_batch = rotation_samples[batch_iter: batch_iter + batchsize]
        current_batch = len(rotations_batch)
        if current_batch < batchsize:
            rotations_np = np.array(rotations_batch)
            pad_np = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (batchsize - current_batch, 1))
            rotations = quaternionic.array(np.concatenate((rotations_np, pad_np), axis=0))
        else:
            rotations = rotations_batch

        # rotate volumes
        rotated_volumes = rotate_volumes(
                        target_volumes.clone(),
                        rotations,
                        microbatch_size=10,  # adjust based on your GPU memory
                        permute_before_sample=False
        )

        # add noise 
        rotated_noisy_volumes = add_awgn(
            rotated_volumes,
            snr_db=config.example_config.snr_db,
            clip=False, 
            mask=mask)
        
        # Compute correlation Coefficients \sigma_l,m,m'
        sigma, _ = config.execution.Correlator.get_sigma(rotated_noisy_volumes)
        
        # search for rotation parameters (timed)
        (alphas, betas, gammas, _), batch_times = timed_search_orientations(
            matcher=config.execution.Matcher,
            sigma=sigma,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            device=device,
        )
        search_timings_ms.extend(batch_times.tolist())
        if print_per_batch:
            print(
                f"[batch {batch_iter // batchsize:03d}] search_orientations "
                f"avg={batch_times.mean():.2f} ms over {timed_runs} timed run(s)"
            )
        
        # update rotation estimates
        rotations, _ = update_rotation_estimate(
            alphas = alphas, 
            betas = betas, 
            gammas = gammas, 
            rotation_tracker= rotations
        )

        batch_dists = np.rad2deg(
            quaternionic.distance.rotation.intrinsic(
                rotations,
                quaternionic.array([1, 0, 0, 0]),
            )
        )
        dists.append(np.asarray(batch_dists)[:current_batch])
    
    dists = np.concatenate(dists, axis=0)
    search_timings_ms = np.asarray(search_timings_ms, dtype=np.float64)
    mean_dist = float(np.mean(dists))
    median_dist = float(np.median(dists))
    p90_dist = float(np.percentile(dists, 90))
    mean_search_ms = float(search_timings_ms.mean())
    std_search_ms = float(search_timings_ms.std())
    median_search_ms = float(np.median(search_timings_ms))
    num_batches = int(search_timings_ms.size) // max(1, timed_runs)
    total_search_ms = mean_search_ms * num_batches
    search_method = str(config.orientation_search.method).lower()
    if search_method == "sofft":
        search_L_max = int(config.sofft_config.L_max)
    else:
        search_L_max = int(config.example_config.L_max)

    print(f"Results of {number_subtomograms} volumes at snr {config.example_config.snr_db}dB:")
    print(f"Mean distance (deg): {mean_dist:.2f}°")
    print(f"Median distance (deg): {median_dist:.2f}°")
    print(f"90th percentile distance (deg): {p90_dist:.2f}°")
    print(
        "search_orientations timing: "
        f"mean={mean_search_ms:.2f} ms, "
        f"std={std_search_ms:.2f} ms, "
        f"median={median_search_ms:.2f} ms, "
        f"total={total_search_ms:.2f} ms ({total_search_ms / 1000:.2f} s) "
        f"for {number_subtomograms} volumes, "
        f"runs={search_timings_ms.size}, "
        f"warmup_per_batch={warmup_runs}"
    )
    return {
        "distance_mean_deg": mean_dist,
        "distance_median_deg": median_dist,
        "distance_p90_deg": p90_dist,
        "search_time_mean_ms": mean_search_ms,
        "search_time_std_ms": std_search_ms,
        "search_time_median_ms": median_search_ms,
        "search_time_total_ms": total_search_ms,
        "search_time_runs": int(search_timings_ms.size),
        "search_time_warmup_per_batch": int(warmup_runs),
        "num_subtomograms": int(number_subtomograms),
        "batchsize": int(batchsize),
        "snr_db": float(config.example_config.snr_db),
        "search_method": str(config.orientation_search.method),
        "search_L_max": search_L_max,
    }

def prepare_alignment_example(config):
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a CUDA-capable GPU."
    device = torch.device("cuda:0")

    # Set execution parameters
    config.execution = ml_collections.ConfigDict()
    config.execution["dtype_real"] = torch.float32
    config.execution["spherical_mask_torch"] = torch.from_numpy(
        get_spherical_mask(
            (config.N, config.N, config.N),
            radius=config.example_config.masking_radius
        )).to(device=device, dtype=config.execution["dtype_real"])

    search_method = config.orientation_search.method.lower()
    if search_method == "matcha":
        search_L_max = int(config.example_config.L_max)
        config.execution.Matcher = Matcha(
            batchsize=config.example_config.batchsize,
            device=device,
            L_max=search_L_max,
            matcha_config=config.matcha_config,
        )
    elif search_method == "sofft":
        sofft_cfg = config.sofft_config
        search_L_max = int(sofft_cfg.L_max)
        config.execution.Matcher = SOFFTMatcher(
            batchsize=config.example_config.batchsize,
            device=device,
            L_max=search_L_max,
            oversampling_factor_K=int(sofft_cfg.oversampling_factor_K),
        )
    else:
        raise ValueError(
            f"Unknown orientation search method '{config.orientation_search.method}'. "
            "Use 'matcha' or 'sofft'."
        )

    # Initialize CrossCorrelationMatcher
    config.execution.Correlator = CrossCorrelationMatcher(
        N = config.N,
        device = device,
        expansion_epsilon = float(config.get("expansion_epsilon", 1e-4)),
        precision_mode = resolve_precision_mode(config),
        batchsize = config.example_config.batchsize,
        reduce_memory = True,
        bandlimit = min(search_L_max + 1, 76),
        micro_batch_split = 2, 
        dtype = torch.float32,
        radius = config.example_config.masking_radius,
        jl_zeros_path = config["jl_zeros_path"],
        cs_path = config["cs_path"]
    )
    
def main(config):

    # Prepare alignment
    prepare_alignment_example(config)

    return run_alignment(config)


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Run example alignment benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "config_example.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="",
        help="Optional path to write returned metrics as JSON.",
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as cf_file:
        config_yaml = yaml.safe_load(cf_file.read())
    config = ml_collections.ConfigDict(config_yaml, type_safe=True)


    config.execution = ml_collections.ConfigDict()
    metrics = main(config)
    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as mf:
            json.dump(metrics or {}, mf, indent=2)


if __name__ == "__main__":
    cli()
