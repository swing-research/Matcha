import numpy as np
import quaternionic
import torch
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F


def rotate_volumes(
    batched_tomogram_data,
    rotation_tracker,
    microbatch_size=10,
    permute_before_sample=False,
):
    """
    Rotate subtomograms in micro-batches to reduce memory usage.

    Parameters
    ----------
    batched_tomogram_data : torch.Tensor
        Input tensor with shape ``(B, D, H, W)`` or ``(B, C, D, H, W)``.
    rotation_tracker : array-like
        Quaternion rotations for each entry in the batch.
    microbatch_size : int, default=10
        Number of volumes to process at once.
    permute_before_sample : bool, default=False
        Whether to permute spatial order before interpolation.

    Returns
    -------
    torch.Tensor
        Rotated tensor with the same shape as the input.
    """
    euler_angles_deg = np.rad2deg(quaternionic.array(rotation_tracker).to_euler_angles)

    for start in range(0, len(batched_tomogram_data), microbatch_size):
        end = start + microbatch_size
        batch_data = batched_tomogram_data[start:end]
        batch_angles = euler_angles_deg[start:end]
        rotated_batch = rotate_volumes_in_batches(
            batch_data,
            batch_angles,
            permute_before_sample=permute_before_sample,
        )
        batched_tomogram_data[start:end].copy_(rotated_batch)

    return batched_tomogram_data


def rotate_volumes_in_batches(
    volume: torch.Tensor,
    euler_angles: torch.Tensor,
    permute_before_sample: bool = False,
) -> torch.Tensor:
    """
    Rotate a batch of 3D volumes using ZYZ Euler angles.

    Parameters
    ----------
    volume : torch.Tensor
        Tensor of shape ``(B, D, H, W)`` or ``(B, C, D, H, W)``.
    euler_angles : torch.Tensor
        Euler angles in degrees with shape ``(B, 3)``.
    permute_before_sample : bool, default=False
        Whether to permute tensor dimensions before sampling.

    Returns
    -------
    torch.Tensor
        Rotated volume tensor.
    """
    assert volume.ndim in [4, 5], "Input volume must be 3D (N,B,D, H, W) or (B, D, H, W)"

    added_batch = False
    if volume.ndim == 4:
        volume = volume.unsqueeze(1)
        added_batch = True
    assert volume.ndim == 5, "vol must be (B,N, D, H, W) or (B,1, D, H, W)"

    b, _, d, h, w = volume.shape
    device = volume.device
    dtype = volume.dtype
    real_dtype = torch.float32
    if dtype == torch.complex128:
        real_dtype = torch.float64

    pad_d = 1 if (d % 2 == 0) else 0
    pad_h = 1 if (h % 2 == 0) else 0
    pad_w = 1 if (w % 2 == 0) else 0
    pad = (0, pad_w, 0, pad_h, 0, pad_d)

    volume = F.pad(volume, pad, mode='constant', value=0)
    dp, hp, wp = volume.shape[2:]

    main_rot = R.from_euler('ZYZ', euler_angles, degrees=True).as_matrix()
    rot_matrix = torch.tensor(main_rot, dtype=real_dtype, device=device)

    d_range = torch.linspace(-1, 1, dp, device=device, dtype=real_dtype)
    h_range = torch.linspace(-1, 1, hp, device=device, dtype=real_dtype)
    w_range = torch.linspace(-1, 1, wp, device=device, dtype=real_dtype)
    w_grid, h_grid, d_grid = torch.meshgrid(d_range, h_range, w_range, indexing='ij')

    coords = torch.stack([d_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=0)
    rotated_coords = rot_matrix @ coords
    rotated_coords = rotated_coords.permute(0, 2, 1)
    rotated_coords = rotated_coords.view(b, dp, hp, wp, 3)

    if permute_before_sample:
        volume = torch.permute(volume, (0, 1, 4, 3, 2))

    vol_out = F.grid_sample(volume.real, rotated_coords, mode='bilinear', align_corners=True)

    if volume.dtype.is_complex:
        rotated_volume_imag = F.grid_sample(volume.imag, rotated_coords, mode='bilinear', align_corners=True)
        vol_out = vol_out + rotated_volume_imag * 1j

    if pad_d:
        vol_out = vol_out[:, :, :d, :, :]
    if pad_h:
        vol_out = vol_out[:, :, :, :h, :]
    if pad_w:
        vol_out = vol_out[:, :, :, :, :w]

    if added_batch:
        vol_out = vol_out.squeeze(1)
    return vol_out
