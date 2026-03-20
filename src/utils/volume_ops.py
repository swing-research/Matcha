import numpy as np
import scipy.ndimage
import torch


def get_spherical_mask(size: tuple, radius: float, sigma: float = 0):
    """Create a spherical mask with a given radius and size."""
    center = np.array(size) // 2
    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    distance = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    spherical_mask = (distance < radius).astype(np.float32)
    soft_mask = scipy.ndimage.gaussian_filter(spherical_mask.astype(float), sigma=sigma)
    soft_mask /= soft_mask.max()
    return soft_mask


def mean(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate the mean of a volume using a mask."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=vol.device, dtype=vol.dtype)
    return (vol * mask).sum() / mask.sum()


def std(vol: torch.Tensor, mask: torch.Tensor, mean_val: torch.Tensor) -> torch.Tensor:
    """Calculate the standard deviation of a volume using a mask and a precomputed mean."""
    return torch.sqrt(mean(vol**2, mask) - mean_val**2)


def normalise(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize a volume using a mask."""
    if mask.sum() == 0:
        return vol
    mean_val = mean(vol, mask)
    std_val = std(vol, mask, mean_val)
    return (vol - mean_val) / std_val
