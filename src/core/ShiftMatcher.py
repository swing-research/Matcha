import math

import numpy as np
import quaternionic
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R

from ml_collections import ConfigDict


@torch.jit.script
def _mean_torch(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return masked mean; if mask is empty, return zero."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=vol.device, dtype=vol.dtype)
    return (vol * mask).sum() / mask.sum()


@torch.jit.script
def _std_torch(vol: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Return masked standard deviation from a precomputed mean."""
    return torch.sqrt(_mean_torch(vol**2, mask) - mean**2)


def normalise_torch(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize a volume using masked mean/std."""
    if mask.sum() == 0:
        return vol
    mean = _mean_torch(vol, mask)
    std = _std_torch(vol, mask, mean)
    return (vol - mean) / std


def get_base_coords(shape: tuple, center: tuple, device: torch.device) -> torch.Tensor:
    """Build an absolute coordinate grid centered at the requested voxel."""
    dz, dy, dx = int(shape[0]), int(shape[1]), int(shape[2])
    cz, cy, cx = int(center[0]), int(center[1]), int(center[2])
    z = torch.arange(dz, device=device) + cz - dz // 2
    y = torch.arange(dy, device=device) + cy - dy // 2
    x = torch.arange(dx, device=device) + cx - dx // 2
    X, Y, Z = torch.meshgrid(z, y, x, indexing='ij')
    return torch.stack((X, Y, Z), dim=-1)


def _upsampled_dft_torch(
    data: torch.Tensor,
    upsampled_region_size: torch.Tensor,
    upsample_factor: float = 1.0,
    axis_offsets: torch.Tensor = None,
):
    """Evaluate a localized upsampled DFT around per-batch offsets."""
    if not torch.is_complex(data):
        raise ValueError("Input `data` must be complex.")

    ndim = data.ndim - 1
    dtype = data.dtype
    device = data.device
    batch_size = data.shape[0] if data.ndim > 1 else 1

    if upsampled_region_size.numel() == 1:
        upsampled_region_size = upsampled_region_size.expand(ndim)
    elif upsampled_region_size.numel() != ndim:
        raise ValueError("`upsampled_region_size` must match data.ndim")
    upsampled_region_size = upsampled_region_size.reshape(ndim)

    if axis_offsets is None:
        axis_offsets = torch.zeros((batch_size, ndim), device=device, dtype=torch.float32)
    else:
        axis_offsets = torch.as_tensor(axis_offsets, device=device, dtype=torch.float32)
        if axis_offsets.ndim == 1:
            axis_offsets = axis_offsets.unsqueeze(0).expand(batch_size, -1)

    im2pi = 2j * np.pi

    for dim in reversed(range(ndim)):
        dim_size = data.shape[dim + 1]
        ups_size = upsampled_region_size[dim]
        offset = axis_offsets[:, dim]

        row_coords = torch.arange(ups_size, device=device, dtype=torch.float32).view(1, ups_size, 1)
        row_coords = row_coords - offset.view(batch_size, 1, 1)

        freqs = torch.fft.fftfreq(dim_size, d=upsample_factor).to(device).view(1, 1, dim_size)
        kernel = torch.exp(-im2pi * row_coords * freqs).to(dtype)

        data = data.movedim(dim + 1, -1)
        data = torch.einsum("bum,b...m->b...u", kernel, data)
        data = data.movedim(-1, dim + 1)

    return data


def compute_shift(
    reference_image: torch.Tensor,
    moving_image: torch.Tensor,
    upsample_factor: int = 1,
    space: str = "real",
    normalization: str = "phase",
    score_normalization: bool = False,
    max_shift: torch.Tensor = None,
    mode: str = "local",
):
    """Estimate translation between reference and moving volumes.

    Parameters
    ----------
    reference_image : torch.Tensor
        Reference volume or batch of volumes with shape ``(D, H, W)`` or
        ``(B, D, H, W)``.
    moving_image : torch.Tensor
        Moving volume or batch of volumes with shape ``(D, H, W)`` or
        ``(B, D, H, W)``.
    upsample_factor : int, default=1
        DFT upsampling factor used for sub-voxel refinement.
    space : {"real", "complex"}, default="real"
        Input domain of ``reference_image`` and ``moving_image``.
    normalization : {"phase", None}, default="phase"
        Correlation normalization strategy.
    score_normalization : bool, default=False
        If ``True``, normalize correlation scores by input norms.
    max_shift : torch.Tensor or None, default=None
        Optional per-axis bound for allowed coarse shifts.
    mode : {"local", "global"}, default="local"
        Refinement mode. ``"local"`` preserves legacy behavior by refining
        around zero. ``"global"`` refines around the coarse peak.

    Returns
    -------
    shift : torch.Tensor
        Estimated shift tensor of shape ``(B, 3)``.
    cross_correlation : torch.Tensor
        Complex-valued cross-correlation volume of shape ``(B, D, H, W)``.
    max_score : torch.Tensor
        Maximum score per batch element after optional refinement.

    Raises
    ------
    ValueError
        If batch sizes are incompatible or if ``mode`` is unsupported.
    """
    assert moving_image.ndim >= 3 and moving_image.ndim <= 4

    if reference_image.ndim == 3:
        reference_image = reference_image.unsqueeze(0)
    if moving_image.ndim == 3:
        moving_image = moving_image.unsqueeze(0)

    b_ref = reference_image.shape[0]
    b_mov = moving_image.shape[0]
    size = reference_image.shape[1]

    if b_ref == 1 and b_mov > 1:
        reference_image = reference_image.expand(b_mov, -1, -1, -1)
    elif b_ref != b_mov:
        raise ValueError(f"Batch sizes must match or reference batch must be 1: got {b_ref} vs {b_mov}")

    b = moving_image.shape[0]

    if not torch.is_complex(reference_image):
        reference_image = reference_image.to(torch.get_default_dtype())
    if not torch.is_complex(moving_image):
        moving_image = moving_image.to(torch.get_default_dtype())

    dims = tuple(range(-reference_image.ndim + 1, 0))
    src_freq = torch.fft.fftn(reference_image.to(torch.float64), dim=dims) if space == "real" else reference_image
    target_freq = torch.fft.fftn(moving_image.to(torch.float64), dim=dims) if space == "real" else moving_image

    image_product = torch.conj(src_freq) * target_freq

    if normalization == "phase":
        eps = torch.finfo(image_product.real.dtype).eps
        image_product /= torch.clamp(image_product.abs(), min=100 * eps)

    cross_correlation = torch.fft.ifftn(image_product, dim=dims, norm="forward", s=(size, size, size))

    abs_corr = cross_correlation.real.contiguous()
    if score_normalization:
        abs_corr /= (
            torch.linalg.norm(target_freq.flatten(start_dim=1), dim=1)
            * torch.linalg.norm(src_freq.flatten(start_dim=1), dim=1)
        )[:, None, None, None]

    b, *spatial_shape = abs_corr.shape

    abs_corr_flat = abs_corr.view(b, -1)
    max_indices = torch.argmax(abs_corr_flat, dim=1)

    coords = []
    for dim_size in reversed(spatial_shape):
        coords.append(max_indices % dim_size)
        max_indices = max_indices // dim_size
    maxima = torch.stack(coords[::-1], dim=1)

    shape = torch.tensor(spatial_shape, dtype=torch.float32, device=abs_corr.device)
    midpoint = torch.floor(shape / 2)[None, :]
    shift = maxima.to(torch.float32)
    shift = torch.where(shift > midpoint, shift - shape, shift)

    if max_shift is not None:
        over = (shift.abs() > max_shift).any(dim=1)
        batch_ids = torch.where(over)[0]
        if batch_ids.numel() > 0:
            k = batch_ids.numel()
            abs_corr_sub = abs_corr[batch_ids]
            pooled = F.max_pool3d(abs_corr_sub, kernel_size=7, stride=1, padding=3)
            candidate_mask = abs_corr_sub == pooled

            topk_lin = (candidate_mask * abs_corr_sub).view(k, -1).topk(k=20, dim=1).indices

            idx = topk_lin
            coords = []
            for dim in reversed(spatial_shape):
                coords.append(idx % dim)
                idx = idx // dim
            coords = torch.stack(coords[::-1], dim=-1).to(torch.float32)

            shape = torch.tensor(spatial_shape, dtype=torch.float32, device=abs_corr.device)
            midpoint = torch.floor(shape / 2)[None, None, :]
            cand_shifts = torch.where(coords > midpoint, coords - shape, coords)

            within = (cand_shifts.abs() <= max_shift).all(dim=2)
            has_valid = within.any(dim=1)
            first_idx = within.float().argmax(dim=1)

            selected = cand_shifts[torch.arange(k, device=shift.device), first_idx]
            zero_shift = torch.zeros(3, device=shift.device, dtype=shift.dtype)
            selected = torch.where(has_valid[:, None], selected.to(shift.dtype), zero_shift)

            new_shift = shift.clone()
            new_shift[batch_ids] = selected
            shift = new_shift

    float_dtype = src_freq.real.dtype

    if upsample_factor > 1:
        if mode not in ("local", "global"):
            raise ValueError(f"Unsupported mode: {mode}. Use 'local' or 'global'.")
        if mode == "local":
            shift = shift * 0.0
        upsample_factor_tensor = torch.tensor(upsample_factor, dtype=float_dtype, device=abs_corr.device)
        shift = torch.round(shift * upsample_factor_tensor) / upsample_factor_tensor
        upsampled_region_size = torch.ceil(upsample_factor_tensor * 5).to(torch.int64)

        dftshift = torch.floor(upsampled_region_size.to(float_dtype) / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor_tensor

        refined_cc = _upsampled_dft_torch(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor=upsample_factor,
            axis_offsets=sample_region_offset,
        ).conj()

        abs_refined_cc = refined_cc.real.contiguous()

        if score_normalization:
            abs_refined_cc /= (
                torch.linalg.norm(target_freq.flatten(start_dim=1), dim=1)
                * torch.linalg.norm(src_freq.flatten(start_dim=1), dim=1)
            )[:, None, None, None]

        flat_idx = torch.argmax(abs_refined_cc.view(b, -1), dim=1)
        maxima = torch.unravel_index(flat_idx, abs_refined_cc.shape[1:])
        maxima = torch.stack(maxima, dim=1).to(torch.float32)
        maxima -= dftshift
        shift += maxima / upsample_factor_tensor

    for dim in range(len(shape)):
        if shape[dim] == 1:
            shift[:, dim] = 0

    return shift, cross_correlation, (
        torch.max(abs_corr.view(b, -1), dim=1)[0]
        if upsample_factor == 1
        else torch.max(abs_refined_cc.view(b, -1), dim=1)[0]
    )


def compute_shift_local(
    reference_image: torch.Tensor,
    moving_image: torch.Tensor,
    upsample_factor: int = 1,
    space: str = "real",
    normalization: str = "phase",
    score_normalization: bool = False,
    max_shift: torch.Tensor = None,
):
    """Estimate shifts using local (legacy) refinement.

    Parameters
    ----------
    reference_image : torch.Tensor
        Reference volume(s), shape ``(D, H, W)`` or ``(B, D, H, W)``.
    moving_image : torch.Tensor
        Moving volume(s), shape ``(D, H, W)`` or ``(B, D, H, W)``.
    upsample_factor : int, default=1
        DFT upsampling factor for sub-voxel refinement.
    space : {"real", "complex"}, default="real"
        Input domain of ``reference_image`` and ``moving_image``.
    normalization : {"phase", None}, default="phase"
        Correlation normalization strategy.
    score_normalization : bool, default=False
        If ``True``, normalize correlation scores by input norms.
    max_shift : torch.Tensor or None, default=None
        Optional per-axis bound for allowed coarse shifts.

    Returns
    -------
    tuple
        Same return tuple as :func:`compute_shift`.
    """
    return compute_shift(
        reference_image=reference_image,
        moving_image=moving_image,
        upsample_factor=upsample_factor,
        space=space,
        normalization=normalization,
        score_normalization=score_normalization,
        max_shift=max_shift,
        mode="local",
    )


def compute_shift_global(
    reference_image: torch.Tensor,
    moving_image: torch.Tensor,
    upsample_factor: int = 1,
    space: str = "real",
    normalization: str = "phase",
    score_normalization: bool = False,
    max_shift: torch.Tensor = None,
):
    """Estimate shifts using global refinement from coarse peak.

    Parameters
    ----------
    reference_image : torch.Tensor
        Reference volume(s), shape ``(D, H, W)`` or ``(B, D, H, W)``.
    moving_image : torch.Tensor
        Moving volume(s), shape ``(D, H, W)`` or ``(B, D, H, W)``.
    upsample_factor : int, default=1
        DFT upsampling factor for sub-voxel refinement.
    space : {"real", "complex"}, default="real"
        Input domain of ``reference_image`` and ``moving_image``.
    normalization : {"phase", None}, default="phase"
        Correlation normalization strategy.
    score_normalization : bool, default=False
        If ``True``, normalize correlation scores by input norms.
    max_shift : torch.Tensor or None, default=None
        Optional per-axis bound for allowed coarse shifts.

    Returns
    -------
    tuple
        Same return tuple as :func:`compute_shift`.
    """
    return compute_shift(
        reference_image=reference_image,
        moving_image=moving_image,
        upsample_factor=upsample_factor,
        space=space,
        normalization=normalization,
        score_normalization=score_normalization,
        max_shift=max_shift,
        mode="global",
    )


def apply_soft_mask(Irefs, ori_size, pixel_size, ini_high, width_fmask_edge=6.0):
    """Low-pass filter references in Fourier space with a cosine edge mask."""
    if ini_high <= 0:
        return Irefs

    radius = ori_size * pixel_size / ini_high
    radius -= width_fmask_edge / 2.0
    radius_p = radius + width_fmask_edge

    filtered_Irefs = []
    for vol in Irefs:
        vol_ft = np.fft.fftn(vol, norm="forward")
        vol_ft = np.fft.fftshift(vol_ft)

        zdim, ydim, xdim = vol.shape
        center = np.array([zdim // 2, ydim // 2, xdim // 2])
        zz, yy, xx = np.indices(vol.shape)
        dz = zz - center[0]
        dy = yy - center[1]
        dx = xx - center[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        mask = np.ones_like(r)
        mask[r > radius_p] = 0
        transition_zone = (r >= radius) & (r <= radius_p)
        mask[transition_zone] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[transition_zone]) / width_fmask_edge)

        vol_ft *= mask
        vol_filtered = np.fft.ifftn(np.fft.ifftshift(vol_ft), norm="forward").real
        filtered_Irefs.append(vol_filtered.astype(np.float64))

    return filtered_Irefs


def gridding_correct(vol_in, ori_size, padding_factor, interpolator, r_min_nn=0):
    """Apply sinc/sinc^2 gridding correction used by RELION preprocessing."""
    assert vol_in.ndim == 3, "Input volume must be 3D"
    z, y, x = vol_in.shape

    kk, ii, jj = np.meshgrid(
        np.arange(-z // 2, z // 2),
        np.arange(-y // 2, y // 2),
        np.arange(-x // 2, x // 2),
        indexing='ij',
    )
    r = np.sqrt(kk**2 + ii**2 + jj**2)
    rval = r / (ori_size * padding_factor)

    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.ones_like(rval)
        nonzero = rval > 0
        sinc[nonzero] = np.sin(np.pi * rval[nonzero]) / (np.pi * rval[nonzero])

    if interpolator == 'nearest' and r_min_nn == 0:
        correction = sinc
    elif interpolator == 'trilinear' or (interpolator == 'nearest' and r_min_nn > 0):
        correction = sinc**2
    else:
        raise ValueError("Unrecognized interpolator: must be 'nearest' or 'trilinear'")

    correction[correction == 0] = 1.0
    vol_in /= correction
    return vol_in


def get_mask_full(shape):
    """Return a centered spherical mask in Fourier index coordinates."""
    nz, ny, nx = shape
    center_z, center_y = nz // 2, ny // 2

    Z, Y, X = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    kp = np.where(Z < center_z, Z, Z - nz)
    ip = np.where(Y < center_y, Y, Y - ny)
    jp = np.where(X < center_y, X, X - nx)

    r2 = kp**2 + ip**2 + jp**2
    return r2 <= nz // 2 * nz // 2


def load_reference_full(ref, volume_size, voxel_size):
    """Build Fourier-domain reference used by shift matching."""
    ref = ref.astype(np.float64)

    iref = apply_soft_mask([ref], ori_size=volume_size, pixel_size=voxel_size, ini_high=4.0, width_fmask_edge=2.0)
    iref_gridded = gridding_correct(iref[0], ori_size=volume_size, padding_factor=1.0, interpolator='trilinear', r_min_nn=0)

    half_size = int(volume_size // 2)
    centered = np.roll(iref_gridded, (-half_size, -half_size, -half_size), axis=(0, 1, 2))

    faux = np.fft.fftn(centered, norm="forward")
    mask = get_mask_full(faux.shape)

    mask_shifted = np.fft.fftshift(mask, axes=(0, 1, 2))
    faux_shifted = np.fft.fftshift(faux, axes=(0, 1, 2)) * mask_shifted

    return faux_shifted


def _grid_sample_complex(x, grid, *, mode='bilinear', align_corners=True, padding_mode='zeros'):
    """Complex-valued wrapper around torch.grid_sample."""
    if not torch.is_complex(x):
        return F.grid_sample(x, grid, mode=mode, align_corners=align_corners, padding_mode=padding_mode)

    xr = torch.view_as_real(x)
    b, c, d, h, w, _ = xr.shape
    xr = xr.permute(0, 5, 1, 2, 3, 4).reshape(b, 2 * c, d, h, w)

    y = F.grid_sample(xr, grid, mode=mode, align_corners=align_corners, padding_mode=padding_mode)

    y_real = y[:, 0::2, ...]
    y_imag = y[:, 1::2, ...]
    return torch.complex(y_real, y_imag)


def rotate_complex_volume(volume: torch.Tensor, euler_angles: torch.Tensor, pad_for_rfft_slicing: bool = False) -> torch.Tensor:
    """Rotate a batch of complex volumes with Euler ZYZ angles (degrees)."""
    assert volume.ndim in (4, 5)
    added_chan = False
    if volume.ndim == 4:
        volume = volume.unsqueeze(1)
        added_chan = True

    b, _, d, h, w = volume.shape
    dev = volume.device
    rdt = torch.float64 if volume.dtype == torch.complex128 else torch.float32

    rb_np = R.from_euler('ZYZ', euler_angles, degrees=True).as_matrix()
    rb = torch.from_numpy(rb_np).to(dev, rdt)

    de = d + ((d + 1) % 2)
    he = h + ((h + 1) % 2)
    we = w + ((w + 1) % 2)
    if pad_for_rfft_slicing:
        w += 1
    cz, cy, cx = (de - 1) / 2, (he - 1) / 2, (we - 1) / 2

    z = torch.arange(d, device=dev, dtype=rdt)
    y = torch.arange(h, device=dev, dtype=rdt)
    x = torch.arange(w, device=dev, dtype=rdt)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    base = torch.stack([xx - cx, yy - cy, zz - cz], dim=0).reshape(3, -1)

    rot = torch.einsum('bij,jn->bin', rb, base)
    rot = rot + torch.tensor([[cx], [cy], [cz]], device=dev, dtype=rdt)
    rot = rot.transpose(1, 2).reshape(b, d, h, w, 3)

    rot = torch.stack(
        [
            rot[..., 0] * (2.0 / (we - 2)) - 1.0,
            rot[..., 1] * (2.0 / (he - 2)) - 1.0,
            rot[..., 2] * (2.0 / (de - 2)) - 1.0,
        ],
        dim=-1,
    )

    with torch.no_grad():
        volume = _grid_sample_complex(volume, rot, mode='bilinear', align_corners=True, padding_mode='zeros')
    if added_chan:
        volume = volume.squeeze(1)
    return volume


def apply_cosine_mask(vol, mask_outer, mask_cosine, raisedcos):
    """Blend outer-region voxels towards background using a raised-cosine edge."""
    b = vol.shape[0]

    sum_bg = vol[:, mask_outer].view(b, -1).sum(dim=1) + (raisedcos * vol[:, mask_cosine]).view(b, -1).sum(dim=1)
    sum_weights = mask_outer.sum() + raisedcos.sum()

    avg_bg = sum_bg / sum_weights if sum_weights > 0 else 0.0

    n = vol[:, mask_outer].shape[1]
    vol[:, mask_outer] = avg_bg[:, None].expand(-1, n)

    n_cosine = vol[:, mask_cosine].shape[1]
    vol[:, mask_cosine] = (1 - raisedcos) * vol[:, mask_cosine] + raisedcos * avg_bg[:, None].expand(-1, n_cosine)

    return vol


def extract_patch_torch_batch(
    volume: torch.Tensor,
    shift: torch.Tensor,
    config: ConfigDict,
    normalize: bool = True,
    set_coords_to_zero: bool = False,
    dtype="complex",
) -> torch.Tensor:
    """Extract shifted patches for each particle/grid shift combination."""
    patch_shape = config["execution"]["shape"]
    batchsize = shift.shape[0]
    num_shifts = shift.shape[1]
    batched_tomogram_data = torch.zeros(
        batchsize,
        num_shifts,
        int(patch_shape[0]),
        int(patch_shape[1]),
        int(patch_shape[2]),
        device=config["device"],
        dtype=config["execution"]["dtype"] if dtype == "complex" else torch.float32,
    )

    volume = volume.permute(0, 3, 2, 1)
    volume = volume.unsqueeze(1)

    vol_shape = volume.shape[2:]
    center = [s // 2 for s in vol_shape]
    base_coords = get_base_coords(patch_shape, center=center, device=volume.device)

    size = torch.tensor(volume.shape[2:], device=volume.device) - 1

    for b in range(batchsize):
        single_tomo = volume[b : b + 1]
        for s in range(num_shifts):
            grid = base_coords + shift[b, s].view(1, 1, 1, 3)

            norm_grid = 2 * grid / size.flip(-1).view(1, 1, 1, 3) - 1
            norm_grid = norm_grid.unsqueeze(0)

            patch = F.grid_sample(single_tomo, norm_grid, mode='bilinear', align_corners=True)[0, 0]

            if set_coords_to_zero:
                abs_shift = torch.abs(shift[b, s])
                sign = torch.sign(shift[b, s])
                floor = torch.floor(abs_shift).to(torch.int64)
                ceil = torch.ceil(abs_shift).to(torch.int64)
                floating = (abs_shift - floor) > 0

                for dim in range(3):
                    if floating[dim]:
                        idx = [slice(None)] * 3
                        idx[dim] = floor[dim] if sign[dim] < 0 else -ceil[dim]
                        patch[tuple(idx)] = 0

            if normalize:
                patch = normalise_torch(patch, config["execution"]["spherical_mask_torch"]) * config["execution"]["spherical_mask_torch"]
            batched_tomogram_data[b, s] = patch

    return batched_tomogram_data.view(-1, *config["execution"]["shape"])


def estimate_shift_like_relion(
    reference_data_fourier,
    subtomos,
    ctfs,
    shift_zyx,
    angles,
    mask_full,
    mask_outer,
    mask_cosine,
    raisedcos,
    config,
    upsample_factor,
):
    """Estimate local shifts using the RELION-style masked correlation pipeline."""
    _ = ctfs  # kept for API compatibility
    rotated_reference = rotate_complex_volume(reference_data_fourier, angles, pad_for_rfft_slicing=True)
    rotated_reference = rotated_reference[:, :, :, int(config.N // 2) :]

    rotated_reference = torch.fft.ifftshift(rotated_reference, dim=(1, 2)) * mask_full

    shifted_subtomos = extract_patch_torch_batch(
        subtomos,
        torch.flip(shift_zyx, dims=[-1]),
        config,
        set_coords_to_zero=True,
        dtype="real",
        normalize=False,
    )

    shifted_subtomos = apply_cosine_mask(shifted_subtomos, mask_outer, mask_cosine, raisedcos)

    half_size = int(config.N // 2)
    subtomo_rffts = torch.fft.rfftn(
        torch.roll(shifted_subtomos, shifts=(-half_size, -half_size, -half_size), dims=(1, 2, 3)),
        norm="forward",
        dim=(1, 2, 3),
    )

    local_shift, _, max_corr = compute_shift_local(
        rotated_reference,
        subtomo_rffts,
        upsample_factor=16,  # keep fixed value for parity with prior behavior
        space="complex",
        normalization=None,
        score_normalization=True,
        max_shift=torch.tensor(8, device=rotated_reference.device),
    )

    total_shift = torch.flip(local_shift, dims=[-1])
    return total_shift, max_corr


@torch.no_grad()
def estimate_shift(
    *,
    reference_data_fourier,
    subtomos,
    ctfs,
    shift_zyx,
    angles,
    mask_full,
    mask_outer,
    mask_cosine,
    raisedcos,
    config,
    upsample_factor,
    micro_batch_size: int = 10,
    device=None,
):
    """Estimate shifts in micro-batches using RELION-style preprocessing.

    Parameters
    ----------
    reference_data_fourier : torch.Tensor
        Fourier-domain reference, shape ``(D, H, Wf)`` or ``(B, D, H, Wf)``.
    subtomos : torch.Tensor
        Subtomogram volume batch.
    ctfs : torch.Tensor or None
        CTF batch (currently kept for API compatibility).
    shift_zyx : torch.Tensor
        Current shift estimates.
    angles : array-like
        Euler angles in degrees (ZYZ convention) per subtomogram.
    mask_full, mask_outer, mask_cosine, raisedcos : torch.Tensor
        Masks used in RELION-style shift scoring.
    config : ConfigDict
        Runtime configuration with volume/mask settings.
    upsample_factor : int
        Requested upsampling factor (the inner RELION-like call currently uses
        its fixed parity-preserving behavior).
    micro_batch_size : int, default=10
        Batch size for chunked processing to control memory usage.
    device : torch.device or None, default=None
        Device override; defaults to ``subtomos.device``.

    Returns
    -------
    shift_estimate : torch.Tensor
        Estimated shifts for all subtomograms.
    shift_scores : torch.Tensor
        Corresponding correlation scores.
    """
    device = device or subtomos.device

    b = subtomos.shape[0]
    n_chunks = math.ceil(b / micro_batch_size)

    est_list, score_list = [], []

    if reference_data_fourier.ndim == 3:
        ref = reference_data_fourier.unsqueeze(0)
    else:
        ref = reference_data_fourier

    for k in range(n_chunks):
        s = k * micro_batch_size
        e = min(b, (k + 1) * micro_batch_size)

        subtomos_mb = subtomos[s:e].to(device, non_blocking=True)
        ctfs_mb = ctfs[s:e] if isinstance(ctfs, torch.Tensor) else None
        shift_zyx_mb = shift_zyx[s:e] if isinstance(shift_zyx, torch.Tensor) else shift_zyx[s:e]
        angles_mb = angles[s:e]

        ref_mb = ref.expand(e - s, *ref.shape[1:])
        shift_estimate_mb, shift_scores_mb = estimate_shift_like_relion(
            reference_data_fourier=ref_mb,
            subtomos=subtomos_mb,
            ctfs=ctfs_mb,
            shift_zyx=shift_zyx_mb,
            angles=angles_mb,
            mask_full=mask_full,
            mask_outer=mask_outer,
            mask_cosine=mask_cosine,
            raisedcos=raisedcos,
            config=config,
            upsample_factor=upsample_factor,
        )

        est_list.append(shift_estimate_mb.detach() if shift_estimate_mb.requires_grad else shift_estimate_mb)
        score_list.append(shift_scores_mb.detach() if shift_scores_mb.requires_grad else shift_scores_mb)

    shift_estimate = torch.cat(est_list, dim=0)
    shift_scores = torch.cat(score_list, dim=0)
    return shift_estimate, shift_scores


class ShiftMatcher:
    """Stateful wrapper for shift-search setup and execution.

    Notes
    -----
    This class owns reference preprocessing and provides a stable API for
    per-batch shift estimation during alignment.
    """

    def __init__(
        self,
        config: ConfigDict,
        device: torch.device,
        dtype: torch.dtype,
        micro_batch_size: int = None,
    ):
        """Initialize matcher resources.

        Parameters
        ----------
        config : ConfigDict
            Runtime configuration containing masks, geometry, and search grid.
        device : torch.device
            Device used for reference and shift computations.
        dtype : torch.dtype
            Complex dtype for stored reference data.
        micro_batch_size : int or None, default=None
            Optional override for shift estimation micro-batch size.
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self.micro_batch_size = (
            int(micro_batch_size)
            if micro_batch_size is not None
            else max(1, int(config.num_subtomograms_per_batch // 2))
        )
        self.reference_data_fourier = None

    @torch.no_grad()
    def set_reference(self, template_data: np.ndarray):
        """Precompute and cache Fourier-domain reference from template data.

        Parameters
        ----------
        template_data : np.ndarray
            Real-space template volume.
        """
        reference_data_fourier = load_reference_full(
            template_data,
            volume_size=int(self.config.N),
            voxel_size=float(self.config.voxel_size),
        )
        self.reference_data_fourier = torch.from_numpy(reference_data_fourier).unsqueeze(0).to(
            self.device,
            dtype=self.dtype,
        )

    def get_base_shifts(
        self,
        num_subtomograms_per_batch: int,
        num_base_shifts: int,
        dtype_real: torch.dtype,
    ) -> torch.Tensor:
        """Build base translation grid for the current batch.

        Parameters
        ----------
        num_subtomograms_per_batch : int
            Batch size ``B``.
        num_base_shifts : int
            Number of base-grid shifts ``S``.
        dtype_real : torch.dtype
            Real dtype for returned tensor.

        Returns
        -------
        torch.Tensor
            Base shifts with shape ``(B, S, 3)``.
        """
        translational_search_grid = self.config.execution.get("translational_search_grid", None)
        if translational_search_grid is None:
            base_grid = torch.zeros(
                (num_base_shifts, 3),
                device=self.device,
                dtype=dtype_real,
            )
        else:
            base_grid = torch.as_tensor(
                translational_search_grid[0],
                device=self.device,
                dtype=dtype_real,
            )
            if base_grid.ndim == 1:
                base_grid = base_grid.unsqueeze(0)
            if base_grid.shape[0] != num_base_shifts:
                if base_grid.shape[0] > num_base_shifts:
                    base_grid = base_grid[:num_base_shifts]
                else:
                    pad = torch.zeros(
                        (num_base_shifts - base_grid.shape[0], 3),
                        device=self.device,
                        dtype=dtype_real,
                    )
                    base_grid = torch.cat((base_grid, pad), dim=0)
        return base_grid.unsqueeze(0).expand(num_subtomograms_per_batch, -1, -1)

    @torch.no_grad()
    def search_shifts(
        self,
        *,
        subtomos: torch.Tensor,
        ctfs: torch.Tensor,
        shift_zyx: torch.Tensor,
        rotation_tracker,
        upsample_factor: int,
        micro_batch_size: int = None,
    ):
        """Estimate shifts for a batch at current orientation estimates.

        Parameters
        ----------
        subtomos : torch.Tensor
            Subtomogram batch.
        ctfs : torch.Tensor
            CTF batch (kept for compatibility with RELION-style API).
        shift_zyx : torch.Tensor
            Current shift estimates in ``(z, y, x)`` order.
        rotation_tracker : quaternionic.array-like
            Current cumulative rotation estimate per particle/grid candidate.
        upsample_factor : int
            Shift refinement upsampling factor argument.
        micro_batch_size : int or None, default=None
            Optional override for chunk size during shift estimation.

        Returns
        -------
        shift_estimate : torch.Tensor
            Estimated shift update.
        shift_scores : torch.Tensor
            Correlation score associated with each estimate.

        Raises
        ------
        RuntimeError
            If reference data has not been set via :meth:`set_reference`.
        """
        if self.reference_data_fourier is None:
            raise RuntimeError("ShiftMatcher reference is not initialized. Call `set_reference` first.")

        angles = np.rad2deg(quaternionic.array(rotation_tracker.conj()).to_euler_angles)
        return estimate_shift(
            reference_data_fourier=self.reference_data_fourier,
            subtomos=subtomos,
            ctfs=ctfs,
            shift_zyx=shift_zyx,
            angles=angles,
            mask_full=self.config.execution.mask_full_t,
            mask_outer=self.config.execution.mask_outer,
            mask_cosine=self.config.execution.mask_cosine,
            raisedcos=self.config.execution.raisedcos,
            config=self.config,
            upsample_factor=upsample_factor,
            micro_batch_size=(
                int(micro_batch_size) if micro_batch_size is not None else self.micro_batch_size
            ),
        )
