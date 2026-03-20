import numpy as np
import quaternionic
import torch
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def sample_rotations_around(base_quat: np.ndarray, n_samples: int = 100, max_angle: float = 0.05) -> np.ndarray:
    """Sample rotations around a base quaternion within an angular radius."""
    base = R.from_quat(base_quat)
    samples = []

    for _ in range(n_samples):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0, max_angle)
        delta = R.from_rotvec(angle * axis)
        samples.append((delta * base).as_quat())

    return quaternionic.array(np.stack(samples, axis=-2))


def update_rotation_estimate(
    alphas: torch.Tensor,
    betas: torch.Tensor,
    gammas: torch.Tensor,
    rotation_tracker: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update the rotation tracker with new Euler angles."""
    rotations = torch.stack([alphas, betas, gammas], dim=1)
    current_rotation = quaternionic.array.from_euler_angles(rotations.cpu().numpy()).ndarray
    current_rotation[:, 1], current_rotation[:, 3] = current_rotation[:, 3], current_rotation[:, 1].copy()
    rotation_tracker = (rotation_tracker * quaternionic.array(current_rotation).conj()).normalized
    return rotation_tracker, current_rotation


def compute_quat(row) -> np.ndarray:
    """Convert RELION ZYZ Euler angles from a row into (w, x, y, z) quaternion order."""
    quat = R.from_euler('ZYZ', row[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].values, degrees=True).as_quat()
    return np.roll(quat, shift=1)
