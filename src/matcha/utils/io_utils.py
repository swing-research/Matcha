import torch
from torch.nn import functional as F
import quaternionic
import numpy as np
import mrcfile
import pandas as pd
from ml_collections import ConfigDict
import starfile 
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from matcha.utils.volume_ops import normalise


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
    token = token.replace("//", "/")
    return token


def _worker_pickle_prefix(config: ConfigDict) -> str:
    if config is None:
        raise ValueError("config is required")

    tmp_prefix = config.get("path_output_tmp", "")
    if str(tmp_prefix).strip():
        return str(tmp_prefix)

    return str(config.path_output)


def _center_crop_or_pad(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop or zero-pad a 3D array to target_shape."""
    result = np.zeros(target_shape, dtype=arr.dtype)
    slices_src, slices_dst = [], []
    for s, t in zip(arr.shape, target_shape):
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(None))
        else:
            start = (t - s) // 2
            slices_src.append(slice(None))
            slices_dst.append(slice(start, start + s))
    result[tuple(slices_dst)] = arr[tuple(slices_src)]
    return result


def mask_effective_radius(mask_path: str, N: int) -> float:
    """Return the maximum voxel distance from the center to any non-zero voxel in the mask."""
    mask = np.asarray(mrcfile.open(mask_path, permissive=True).data, dtype=np.float32)
    if mask.shape != (N, N, N):
        mask = _center_crop_or_pad(mask, (N, N, N))
    c = N / 2.0
    z, y, x = np.ogrid[:N, :N, :N]
    dist = np.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2)
    nonzero = mask > 0.5
    return float(dist[nonzero].max()) if nonzero.any() else N / 2.0


def load_template(path_template: str,
                  spherical_mask: torch.Tensor,
                  dtype: torch.dtype,
                  device: torch.device,
                  path_template_mask: str="",) -> torch.Tensor:
    """
    Load the template volume from an MRC file and normalize it under a spherical mask.
    Parameters:
    - config: ConfigDict, configuration containing the path to the template and execution parameters.
    Returns:
    - None, modifies the config in place to include the template data.
    """

    # Load template volume from MRC file
    template = mrcfile.open(path_template).data

    if path_template_mask != "":
        template_mask = mrcfile.open(path_template_mask).data
        if template_mask.shape != template.shape:
            template_mask = _center_crop_or_pad(template_mask, template.shape)
        template = template * template_mask

    # Normalise template under mask
    #template = normalise(template, spherical_mask) * spherical_mask
    return template

def _get_base_coords_zyx(shape: tuple, center: tuple, device: torch.device) -> torch.Tensor:
    """
    Generate base coordinates for a patch centered around a specified point.
    Parameters:
    - shape: tuple of int, the shape of the patch (depth, height, width).
    - center: tuple of int, the center coordinates (cz, cy, cx).
    - device: torch.device, the device to create the coordinates on.
    Returns:
    - torch.Tensor, a tensor of shape (D, H, W, 3) containing the base coordinates.
    """
    dz, dy, dx = shape
    cz, cy, cx = center
    z = torch.arange(dz, device=device) + cz - dz // 2
    y = torch.arange(dy, device=device) + cy - dy // 2
    x = torch.arange(dx, device=device) + cx - dx // 2
    X,Y,Z = torch.meshgrid(z, y, x, indexing='ij')
    base_coords = torch.stack((Z,Y,X), dim=-1)  # shape (D, H, W, 3)
    return base_coords

def extract_subtomogram_patch_batch(volume: torch.Tensor, 
                              shift: torch.Tensor, 
                              config: ConfigDict, 
                              normalize: bool = True, 
                              set_coords_to_zero:bool = False, 
                              dtype = "complex") -> torch.Tensor:
    """
    Extract patches from a 3D volume based on specified shifts.
    Parameters:
    - volume: torch.Tensor, the input 3D volume with shape (B, D, H, W).
    - shift: torch.Tensor, the shifts to apply, shape (B, num_shifts, 3).
    - config: ConfigDict, configuration containing device, dtype, and patch shape.
    Returns:
    - torch.Tensor, a tensor of shape (B * num_shifts, D, H, W) containing the extracted patches.
    """

    # Ensure tomogram has a batch and channel dimension
    patch_shape = config["execution"]["shape"]
    batchsize = shift.shape[0]
    num_shifts = shift.shape[1]
    batched_tomogram_data = torch.zeros(batchsize, num_shifts, *patch_shape, device=config["device"], dtype=config["execution"]["dtype"] if dtype == "complex" else torch.float32)
    
    volume = volume.unsqueeze(1)  # Shape: (1, 1, D, H, W)
   
    #tomogram = F.pad(tomogram, pad_size, mode='constant', value=0)
    vol_shape = volume.shape[2:]  # D, H, W
    center = [s // 2 for s in vol_shape]  # voxel center of padded tomogram
    base_coords = _get_base_coords_zyx(patch_shape, 
                                  center=center, 
                                  device=volume.device
                                  )
    
    size = torch.tensor(volume.shape[2:], device=volume.device) - 1  # D_vol, H_vol, W_vol

    for b in range(batchsize):
        single_tomo = volume[b:b+1]
        for s in range(num_shifts):
            # Extract the patch using the provided coordinates            
            grid = base_coords + shift[b, s].view(1, 1, 1, 1, 1, 3)
            
            norm_grid = 2 * grid / size.flip(-1).view(1, 1, 1, 1, 1, 3) - 1  # shape (1, 1, D, H, W, 3)
            norm_grid = norm_grid.squeeze(0)  # shape (1, D, H, W, 3) for grid_sample
           
            # Sample using grid_sample
            patch = F.grid_sample(single_tomo, norm_grid, mode='bilinear', align_corners=True)[0, 0]

            # Set specific coordinate axis to zero. This is useful for 
            if set_coords_to_zero:
                abs_shift = torch.abs(shift[b, s])
                sign = torch.sign(shift[b, s])
                floor = torch.floor(abs_shift).to(torch.int64)
                ceil= torch.ceil(abs_shift).to(torch.int64)
                floating = (abs_shift - floor) > 0
                
                for dim in range(3):
                    if floating[dim]:
                        # Create slice: [:, :, :] with dim-th index set to floor[dim]
                        idx = [slice(None)] * 3
                        idx[dim] = floor[dim] if sign[dim] < 0 else -ceil[dim]
                       
                        patch[tuple(idx)] = 0

            if normalize:
                patch = normalise(patch, config["execution"]["spherical_mask_torch"]) * config["execution"]["spherical_mask_torch"]
            batched_tomogram_data[b, s] = patch
    return batched_tomogram_data.view(-1, *config["execution"]["shape" ])


def store_alignment_parameters(config: ConfigDict,
                                result_data: pd.DataFrame,
                                tomogram_file_names: list,
                                rotation_scores: torch.Tensor,
                                shift_scores: torch.Tensor,
                                local_shifts: torch.Tensor,
                                rotation_tracker: torch.Tensor,
                                prior_shifts: torch.Tensor,
                                half: str, 
                                alternation_index: int = None):
    """
    Store the alignment parameters in a DataFrame and save it to a pickle file.
    Parameters:
    - config: ConfigDict, configuration containing execution parameters.
    - result_data: pd.DataFrame, DataFrame to store the results.
    - tomogram_file_names: list of str, list of tomogram file names.
    - rotation_scores: torch.Tensor, tensor of rotation scores.
    - shift_scores: torch.Tensor, tensor of shift scores 
    - local_shifts: torch.Tensor, tensor of local shifts.
    - rotation_tracker: torch.Tensor, tensor of rotation parameters.
    - prior_shifts: torch.Tensor, tensor of prior shifts.
    - alternation_index: int, index of the current alternation (optional).
    """
    # Initialize result DataFrame
    num_subtomograms_per_batch = config["num_subtomograms_per_batch"]

    # Store highest scoring alignment
    max_ids = torch.argmax(rotation_scores.view(num_subtomograms_per_batch, 1), dim=1)

    for i,tomogram_file_name in enumerate(tomogram_file_names):

        max_id = max_ids[i]
        result_data.loc[len(result_data)] ={"path": tomogram_file_name,
                                            "rotation_score": rotation_scores.view(num_subtomograms_per_batch, 1)[i,max_id].item(),
                                            "rotation": quaternionic.array(rotation_tracker.reshape(num_subtomograms_per_batch, 1, -1)[i, max_id]).to_euler_angles,
                                            "translation": local_shifts[i,max_id].cpu().numpy(),
                                            "file_name": tomogram_file_name.split("/")[-1],
                                            "prior_shift": prior_shifts[i, max_id].cpu().numpy(),
                                            "alternation_index": alternation_index if alternation_index is not None else 0,
                                            "shift_score": shift_scores.view(num_subtomograms_per_batch, 1)[i,max_id].item(),
                                            "rlnTomoParticleName": _particle_token(tomogram_file_name),
                                            "half": half,
                                            }
    # store alignment parameters
    result_data.to_pickle(f"{config.execution.output_file_name}.pkl")



def load_subtomogram_cpu(path: str, dtype=np.float32) -> torch.Tensor:
    """
    CPU-only, zero-copy (where possible), memmapped MRC → torch.Tensor on CPU.
    No FFTs, no device transfers here.
    """
    # memmap avoids reading whole file into RAM at once
    with mrcfile.open(path, permissive=True) as m:
        # Ensure C-order float32 once (cheap if already float32)
        arr = np.asarray(m.data, dtype=dtype, order="C", copy=True)
    return arr

def load_ctf_relion5_cpu(filepath: str, transpose: bool = False) -> np.ndarray:
    """
    CPU-only CTF load. Keep float32. Avoid double opens and float64.
    """
    
    with mrcfile.mmap(filepath, permissive=True) as m:
        ctf = m.data
    
        # Ensure C-order float32
        ctf = np.asarray(ctf, dtype=np.float32, order="C")
    if transpose:
        ctf = np.transpose(ctf, (2, 1, 0))  # still a view when possible
    # Keep only the rfft half: first N//2 slices in z, N//2+1 in x (last axis)
    ctf = ctf[: ctf.shape[0] // 2, :, : ctf.shape[0] // 2 + 1]
    return ctf

def join_data(star_path: str, workers: int = 1, config=None):

    pkl_prefix = _worker_pickle_prefix(config)

    # Load all STAR blocks in a single read
    raw = starfile.read(star_path)
    if isinstance(raw, dict):
        df_base = raw.get("particles", list(raw.values())[-1])
        optics   = raw.get("optics")
        general  = raw.get("general")
    elif isinstance(raw, list):
        df_base  = raw[2] if len(raw) > 2 else raw[-1]
        optics   = raw[1] if len(raw) > 1 else None
        general  = raw[0] if len(raw) > 0 else None
    else:
        df_base = raw
        optics = general = None

    pixel_size = optics["rlnImagePixelSize"].iloc[0]

    # Load all worker results
    parts = []
    for half_id in [1,2]:
        for worker_id in range(workers):
            file_name = f"{pkl_prefix}_half{half_id}_{worker_id}.pkl"
            if Path(file_name).exists():
                df = pd.read_pickle(file_name)
                parts.append(df)
            else:
                print(f"Warning: {file_name} not found")
    if not parts:
        print("No worker files found.")
        return False

    df_result = pd.concat(parts, ignore_index=True)

    base_join_col = "rlnTomoParticleName" if "rlnTomoParticleName" in df_base.columns else "rlnImageName"
    if base_join_col not in df_base.columns:
        raise KeyError("Base particles table must contain 'rlnTomoParticleName' or 'rlnImageName'.")

    result_join_col = "rlnTomoParticleName" if "rlnTomoParticleName" in df_result.columns else "path"
    if result_join_col not in df_result.columns:
        raise KeyError("Result table must contain 'rlnTomoParticleName' or 'path'.")

    df_base["_match_token"] = df_base[base_join_col].astype(str).map(_particle_token)
    df_result["_match_token"] = df_result[result_join_col].astype(str).map(_particle_token)
    df_result = df_result.drop_duplicates(subset="_match_token", keep="last")

    # --- merge on normalized particle token ---
    dfm = df_base.merge(
        df_result,
        on="_match_token",
        how="left",
        suffixes=("", "_res"),
        indicator=True
    )
    dfm["fastAlignment"] = (dfm["_merge"] == "both").astype(np.int8)
    dfm = dfm.drop(columns=["_merge"])

    # --- compute eulers only for matching rows ---
    if "rotation" in dfm.columns:
        mask = dfm["rotation"].notna()
        if mask.any():
            rot_arr = np.stack(dfm.loc[mask, "rotation"].to_numpy())
            eulers = R.from_euler("zyz", rot_arr, degrees=False).inv().as_euler("ZYZ", degrees=True)
            dfm.loc[mask, ["rlnAnglePsi","rlnAngleTilt","rlnAngleRot"]] = eulers

    if "translation" in dfm.columns:
        mask = dfm["translation"].notna()
        if mask.any():
            shifts = np.stack(-1*dfm.loc[mask, "translation"].to_numpy())
            dfm.loc[mask, "rlnOriginXAngst"] = shifts[:, 0] * pixel_size
            dfm.loc[mask, "rlnOriginYAngst"] = shifts[:, 1] * pixel_size
            dfm.loc[mask, "rlnOriginZAngst"] = shifts[:, 2] * pixel_size

    if "alternation_index" in dfm.columns:
        final_alternation = int(config.get("num_alternations", 1)) - 1
        dfm = dfm[
            dfm["alternation_index"].isna()
            | (dfm["alternation_index"] == final_alternation)
        ]

    if "half" in dfm.columns:
        half_to_subset = {"half1": 1, "half2": 2}
        mapped = dfm["half"].map(half_to_subset)
        if "rlnRandomSubset" in dfm.columns:
            dfm.loc[mapped.notna(), "rlnRandomSubset"] = mapped[mapped.notna()].astype(np.int32)
        elif bool(config.get("random_half_split", False)):
            dfm["rlnRandomSubset"] = mapped.fillna(0).astype(np.int32)
    
    # remove all columns from df_result
    cols_to_remove = [col for col in df_result.columns if col != "rlnTomoParticleName"]
    cols_to_remove += [f"{col}_res" for col in cols_to_remove]

    dfm = dfm.drop(columns=cols_to_remove, errors="ignore")
    if "rlnTomoParticleName" in dfm.columns:
        dfm = dfm.drop_duplicates(subset="rlnTomoParticleName")
    else:
        dfm = dfm.drop_duplicates(subset="_match_token")
    dfm = dfm.drop(columns=["_match_token"], errors="ignore")

    out_blocks = {}
    if general is not None:
        out_blocks["general"] = general
    if optics is not None:
        out_blocks["optics"] = optics
    out_blocks["particles"] = dfm
    starfile.write(
        out_blocks,
        f"{config.path_output}.star",
        overwrite=True,
    )
    return True
