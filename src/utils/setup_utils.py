import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def resolve_precision_mode(config) -> str:
    """Validate and return precision_mode from config.

    Returns 'fast' or 'accurate'. Defaults to 'accurate'.
    """
    mode = str(config.get("precision_mode", "accurate")).strip().lower()
    if mode not in ("fast", "accurate"):
        raise ValueError(
            f"Unknown precision_mode '{mode}'. Choose 'fast' or 'accurate'."
        )
    return mode

import numpy as np
import pandas as pd
import quaternionic
import starfile
import torch
from ml_collections import ConfigDict
from torch.nn import functional as F

from utils.volume_ops import get_spherical_mask


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_size(n: int) -> int:
    n = n - 1
    return (2 * n * (n + 1) * (2 * n + 1)) // 3 + 2 * n * (n + 1) + n + 1


def transform_filename(filename: str) -> str:
    if "subtomo" in filename:
        return filename

    parts = filename.split("_")
    if len(parts) != 3:
        raise ValueError("Filename format is not as expected.")

    prefix = "_".join(parts[:2])
    number = parts[2]
    if not number.endswith(".mrc"):
        raise ValueError("Filename does not end with '.mrc'.")

    number_core = number[:-4]
    subtomo_prefix = number_core[:3]
    subtomo_number = number_core[3:]
    return f"{prefix}_{subtomo_prefix}_subtomo{subtomo_number}.mrc"


def _read_star_raw(star_path: str) -> dict:
    """Read a RELION STAR file and return a dict with all named blocks."""
    raw = starfile.read(star_path)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        # Older starfile versions may return a list; build a best-guess dict.
        keys = ["general", "optics", "particles"]
        return {keys[i]: raw[i] for i in range(min(len(keys), len(raw)))}
    return {"particles": raw}


def _read_particles_table(star_path: str):
    raw = _read_star_raw(star_path)
    if "particles" in raw:
        return raw["particles"]
    # Fall back to last block if 'particles' key not present
    return list(raw.values())[-1]


def _validate_subtomograms_3d(star_path: str) -> None:
    """Raise if _rlnTomoSubTomosAre2DStacks != 0 (2D stacks not supported)."""
    raw = _read_star_raw(star_path)
    general = raw.get("general")
    if general is None:
        return
    col = "rlnTomoSubTomosAre2DStacks"
    val = None
    if isinstance(general, pd.DataFrame) and col in general.columns:
        val = int(general[col].iloc[0])
    elif isinstance(general, dict) and col in general:
        val = int(general[col])
    if val is not None and val != 0:
        raise ValueError(
            f"_rlnTomoSubTomosAre2DStacks = {val}. "
            "matcha requires 3D subtomograms (_rlnTomoSubTomosAre2DStacks = 0)."
        )


def _particle_token(path_like: str) -> str:
    path_norm = str(path_like).strip()
    if "@" in path_norm:
        path_norm = path_norm.split("@", 1)[1]
    path_norm = path_norm.replace("\\", "/")
    parts = Path(path_norm).parts
    token = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    token = token.replace("_weights.mrc", "").replace("_ctf.mrc", "")
    token = token.replace("_data.mrc", "").replace(".mrc", "")
    token = token.replace("_subtomo", "_")
    token = re.sub(r"_+", "_", token)
    return token


def _lookup_particles(df: pd.DataFrame, tomogram_file_names: List[str]) -> pd.DataFrame:
    token_col = "rlnTomoParticleName" if "rlnTomoParticleName" in df.columns else "rlnImageName"
    if token_col not in df.columns:
        raise ValueError("STAR particles table has neither rlnTomoParticleName nor rlnImageName.")

    df_tmp = df.copy()
    df_tmp["_token"] = df_tmp[token_col].astype(str).map(_particle_token)

    wanted_tokens = [_particle_token(p) for p in tomogram_file_names]
    indexed = df_tmp.set_index("_token")

    missing = [tok for tok in wanted_tokens if tok not in indexed.index]
    if missing:
        raise ValueError(f"Could not match {len(missing)} subtomograms in particles table.")

    subset = indexed.loc[wanted_tokens]
    if isinstance(subset, pd.Series):
        subset = subset.to_frame().T
    return subset.reset_index(drop=True)


def find_mrc_in_selected_subdirs(
    root_path: str,
    allowed_subdirs: List[str],
    path_to_input_files: str = None,
    subset_ID_path: str = None,
    subset_ID: int = None,
    tutorial: bool = False,
):
    _ = subset_ID_path, subset_ID, tutorial
    files_df = None
    if path_to_input_files:
        files_df = pd.read_pickle(path_to_input_files)

    allowed = set(allowed_subdirs or [])
    mrc_files = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        if allowed:
            dirnames[:] = [d for d in dirnames if d in allowed]

        for file in filenames:
            if not file.endswith(".mrc"):
                continue
            if "weights" in file or "div" in file or "ctf" in file:
                continue
            if "subtomo" not in file and "data" not in file:
                continue

            if files_df is not None and "file_name" in files_df.columns:
                if files_df[files_df["file_name"].str.contains(file, regex=False)].empty:
                    continue

            mrc_files.append(os.path.join(dirpath, file))

    mrc_files = sorted(mrc_files)
    print(f"Found {len(mrc_files)} MRC files in the selected subdirectories.")
    return mrc_files


def filter_by_subset(subset_ID_path: str, subset_ID: int, mrc_files: List[str]) -> List[str]:
    if not subset_ID_path or subset_ID is None:
        return list(mrc_files)

    df = _read_particles_table(subset_ID_path)
    if "rlnRandomSubset" not in df.columns:
        return list(mrc_files)

    token_col = "rlnTomoParticleName" if "rlnTomoParticleName" in df.columns else "rlnImageName"
    if token_col not in df.columns:
        return list(mrc_files)

    allowed_df = df.loc[df["rlnRandomSubset"] == subset_ID, token_col].astype(str)
    allowed_tokens = set(allowed_df.map(_particle_token))
    return [p for p in mrc_files if _particle_token(p) in allowed_tokens]


def _resolve_particles_image_path(raw: str, star_parent: Path) -> Path:
    value = str(raw).strip()
    if "@" in value:
        value = value.split("@", 1)[1]
    value = value.replace("\\", "/")
    path = Path(value)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        star_parent / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def get_subtomogram_paths_from_particles(star_path: str) -> List[str]:
    _validate_subtomograms_3d(star_path)
    df = _read_particles_table(star_path)
    if "rlnImageName" not in df.columns:
        raise ValueError("Input particles STAR is missing required column 'rlnImageName'.")

    star_parent = Path(star_path).resolve().parent
    seen = set()
    paths = []
    missing = []

    for raw in df["rlnImageName"].astype(str).tolist():
        full_path = _resolve_particles_image_path(raw, star_parent)
        full_str = str(full_path)
        if full_str in seen:
            continue
        seen.add(full_str)
        if not full_path.is_file():
            missing.append(full_str)
            continue
        paths.append(full_str)

    if not paths:
        raise ValueError("No subtomogram files from rlnImageName could be resolved on disk.")
    if missing:
        print(
            f"[WARN] {len(missing)} particle image paths from STAR were not found on disk and were skipped."
        )
    return sorted(paths)


def _get_mask_full(shape: Tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = shape
    center_z, center_y = nz // 2, ny // 2

    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    kp = np.where(z < center_z, z, z - nz)
    ip = np.where(y < center_y, y, y - ny)
    jp = np.where(x < center_y, x, x - nx)

    r2 = kp**2 + ip**2 + jp**2
    return r2 <= (nz // 2) * (nz // 2)


def _compute_masks(shape: Tuple[int, int, int], radius: float, cosine_width: float = 5.0):
    z, y, x = shape
    if radius < 0:
        radius = x / 2.0
    radius_p = radius + cosine_width

    zz, yy, xx = np.meshgrid(
        np.arange(-z // 2, z // 2),
        np.arange(-y // 2, y // 2),
        np.arange(-x // 2, x // 2),
        indexing="ij",
    )
    r = np.sqrt(xx**2 + yy**2 + zz**2)

    mask_inner = r < radius
    mask_outer = r > radius_p
    mask_cosine = (~mask_inner) & (~mask_outer)
    raisedcos = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r[mask_cosine]) / cosine_width)
    return mask_outer, mask_cosine, raisedcos


def setup_mask(config: ConfigDict, rotation_only: bool = False) -> ConfigDict:
    n = int(config["N"])
    radius = float(config.get("radius", n // 2))

    config.execution["spherical_mask"] = get_spherical_mask((n, n, n), radius)
    config.execution["spherical_mask_torch"] = torch.tensor(
        config.execution["spherical_mask"],
        device=config["device"],
        dtype=torch.int32,
    )

    if not rotation_only:
        mask_full = _get_mask_full((n, n, int(n // 2) + 1))
        config.execution.mask_full_t = torch.from_numpy(mask_full).unsqueeze(0).to(config["device"], dtype=torch.bool)

        box_size = float(config.get("box_size", n))
        voxel_size = float(config.get("voxel_size", 1.0))
        mask_radius = box_size / (2.0 * voxel_size)
        mask_outer, mask_cosine, raisedcos = _compute_masks((n, n, n), mask_radius, 5.0)

        config.execution.mask_outer = torch.tensor(mask_outer, device=config["device"], dtype=torch.bool)
        config.execution.mask_cosine = torch.tensor(mask_cosine, device=config["device"], dtype=torch.bool)
        config.execution.raisedcos = torch.tensor(
            raisedcos,
            device=config["device"],
            dtype=config.execution["dtype_real"],
        )

    return config


def get_prior_shifts(
    subtomograms: torch.Tensor,
    tomogram_file_names: List[str],
    df: pd.DataFrame,
    config: ConfigDict,
) -> torch.Tensor:
    _ = subtomograms
    subset = _lookup_particles(df, tomogram_file_names)

    if all(c in subset.columns for c in ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]):
        shifts = -subset[["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]].to_numpy(dtype=np.float32)
        shifts = shifts / float(config.voxel_size)
    elif all(c in subset.columns for c in ["rlnOriginX", "rlnOriginY", "rlnOriginZ"]):
        shifts = -subset[["rlnOriginX", "rlnOriginY", "rlnOriginZ"]].to_numpy(dtype=np.float32)
    else:
        shifts = np.zeros((len(tomogram_file_names), 3), dtype=np.float32)

    prior_shifts = torch.from_numpy(shifts).to(device=config["device"], dtype=config.execution["dtype_real"])
    prior_shifts = prior_shifts.unsqueeze(1).expand(-1, 1, -1)

    pad_count = int(config["num_subtomograms_per_batch"]) - prior_shifts.shape[0]
    if pad_count > 0:
        prior_shifts = F.pad(prior_shifts, (0, 0, 0, 0, 0, pad_count), value=0.0)
    return prior_shifts


def get_rotation_tracker(tomogram_file_names: List[str], df: pd.DataFrame, config: ConfigDict):
    batch_size = int(config.num_subtomograms_per_batch)

    if not bool(config.get("use_prior_rotation", False)):
        base = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        tiled = np.tile(base, (batch_size, 1))
        return quaternionic.array(tiled)

    subset = _lookup_particles(df, tomogram_file_names)
    if not all(c in subset.columns for c in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]):
        raise ValueError("Missing Euler angle columns in particles table for prior rotation.")

    eulers_deg = subset[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(dtype=np.float64)
    return quaternionic.array.from_euler_angles(np.deg2rad(eulers_deg)).conj()


def pad_data(padding_size: int, subtomograms, ctfs, subtomos, rotation_tracker):
    subtomograms = F.pad(subtomograms, (0, 0, 0, 0, 0, 0, 0, padding_size), "constant", 0)
    ctfs = F.pad(ctfs, (0, 0, 0, 0, 0, 0, 0, padding_size), "constant", 0) if ctfs is not None else None
    subtomos = F.pad(subtomos, (0, 0, 0, 0, 0, 0, 0, padding_size), "constant", 0)

    if rotation_tracker.shape[0] < subtomos.shape[0]:
        arr = np.pad(rotation_tracker.ndarray, ((0, padding_size), (0, 0)), "constant", constant_values=1)
        rotation_tracker = quaternionic.array(arr).normalized

    return subtomograms, ctfs, subtomos, rotation_tracker


def assert_inputs(config: ConfigDict) -> None:
    assert isinstance(config.gpu_ids, list) and len(config.gpu_ids) > 0, "gpu_ids must be a non-empty list."
    assert isinstance(config.path_templates, list) and len(config.path_templates) > 0, "path_templates must be a non-empty list."
    random_split = bool(config.get("random_half_split", False))

    if random_split:
        assert len(config.path_templates) == 2, "random_half_split requires exactly two templates."
    else:
        assert isinstance(config.subset_IDs, list) and len(config.subset_IDs) > 0, "subset_IDs must be a non-empty list."
        assert len(config.subset_IDs) in [1, 2], "subset_IDs must have length 1 or 2."
        assert len(config.path_templates) == len(config.subset_IDs), "path_templates and subset_IDs length mismatch."

    missing_templates = [p for p in config.path_templates if not Path(p).is_file()]
    assert not missing_templates, f"Missing template files: {missing_templates}"

    if not Path(config.particles_starfile).is_file():
        raise AssertionError(f"particles_starfile does not exist: {config.particles_starfile}")


def _random_subset_lookup(star_path: str, seed: int) -> Dict[str, int]:
    df = _read_particles_table(star_path)
    token_col = "rlnTomoParticleName" if "rlnTomoParticleName" in df.columns else "rlnImageName"
    if token_col not in df.columns:
        raise ValueError("STAR particles table has neither rlnTomoParticleName nor rlnImageName.")

    tokens = list(dict.fromkeys(df[token_col].astype(str).map(_particle_token).tolist()))
    if not tokens:
        raise ValueError("Input particles STAR did not contain any particle entries.")

    rng = np.random.default_rng(seed)
    shuffled = tokens[:]
    rng.shuffle(shuffled)

    half1_count = (len(shuffled) + 1) // 2
    subset_lookup = {}
    for idx, token in enumerate(shuffled):
        subset_lookup[token] = 1 if idx < half1_count else 2
    return subset_lookup


def setup_data_splits(config: ConfigDict, subtomograms: List[str]):
    if bool(config.get("random_half_split", False)):
        particles_df = _read_particles_table(config.particles_starfile)
        if "rlnRandomSubset" in particles_df.columns:
            subset_1 = filter_by_subset(config.particles_starfile, 1, subtomograms)
            subset_2 = filter_by_subset(config.particles_starfile, 2, subtomograms)
            print(
                "Using existing rlnRandomSubset from input particles STAR: "
                f"half1={len(subset_1)}, half2={len(subset_2)}."
            )
        else:
            subset_lookup = _random_subset_lookup(
                star_path=config.particles_starfile,
                seed=int(config.get("random_seed", 0)),
            )
            subset_1 = [p for p in subtomograms if subset_lookup.get(_particle_token(p)) == 1]
            subset_2 = [p for p in subtomograms if subset_lookup.get(_particle_token(p)) == 2]
            print(
                "rlnRandomSubset missing in input STAR; using deterministic random split: "
                f"half1={len(subset_1)}, half2={len(subset_2)}."
            )

        run_data = {
            "half1": {
                "ref": config.path_templates[0],
                "subset_ID": 1,
                "subtomogram_paths": subset_1,
            },
            "half2": {
                "ref": config.path_templates[1],
                "subset_ID": 2,
                "subtomogram_paths": subset_2,
            },
        }
        return [run_data]

    num_halves = len(config.path_templates)

    subset_1 = filter_by_subset(config.particles_starfile, config.subset_IDs[0], subtomograms)

    run_data = {
        "half1": {
            "ref": config.path_templates[0],
            "subset_ID": config.subset_IDs[0],
            "subtomogram_paths": subset_1,
        }
    }

    if num_halves == 2:
        subset_2 = filter_by_subset(config.particles_starfile, config.subset_IDs[1], subtomograms)
        run_data["half2"] = {
            "ref": config.path_templates[1],
            "subset_ID": config.subset_IDs[1],
            "subtomogram_paths": subset_2,
        }

    return [run_data]
