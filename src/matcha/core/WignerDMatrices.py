"""
Wigner D-matrix computation for SO(3).

Adapted from the reference implementation by Michael Boyle:
  https://github.com/moble/spherical
"""

import numpy as np
from numba import cuda
import numba 
import math
import torch
import functools
import time
def choose_dims(B, D, prefer_bx=256, max_by=4):
    dev = cuda.get_current_device()
    # x = loop (contiguous)
    bx = min(prefer_bx, dev.MAX_THREADS_PER_BLOCK)
    bx = min(bx, max(32, 1 << (int(math.log2(max(1, min(D, prefer_bx))))))
             )  # power-of-two up to prefer_bx
    # y = batch lanes per block
    by = min(max_by, max(1, dev.MAX_THREADS_PER_BLOCK // bx))
    # grid
    gx = (D + bx - 1) // bx
    gy = (B + by - 1) // by
    # dynamic shared memory: one float per (x thread) per (y lane)
    shmem = bx * by * np.dtype(np.float32).itemsize
    return (gx, gy), (bx, by), shmem

jit = njit = functools.partial(numba.njit, cache=True)


@jit
def nabsm_index(n, absm):
    """Return flat index into arrray of [n, abs(m)] pairs

    Assumes array is ordered as

        [
            [n, m]
            for n in range(n_max+1)
            for m in range(n+1)
        ]

    """
    return absm + (n * (n + 1)) // 2

def WignerDsize(ell_min, mp_max, ell_max=-1):
    """Compute total size of Wigner 𝔇 matrix

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    mp_max : int, optional
        Integer satisfying 0 <= mp_max.  Defaults to ell_max.
    ell_max : int
        Integer satisfying 0 <= ell_min <= ell_max

    Returns
    -------
    i : int
        Total size of Wigner 𝔇 matrix arranged as described below

    See Also
    --------
    WignerDrange : Array of (ℓ, m', m) indices corresponding to the 𝔇 matrix
    WignerDindex : Index of a particular element of the 𝔇 matrix

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_min, ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    if ell_max < 0:
        ell_max = mp_max
    if mp_max >= ell_max:
        return (
            ell_max * (ell_max * (4 * ell_max + 12) + 11)
            + ell_min * (1 - 4 * ell_min**2)
            + 3
        ) // 3
    if mp_max > ell_min:
        return (
            3 * ell_max * (ell_max + 2)
            + ell_min * (1 - 4 * ell_min**2)
            + mp_max * (
                3 * ell_max * (2 * ell_max + 4)
                + mp_max * (-2 * mp_max - 3) + 5
            )
            + 3
        ) // 3
    else:
        return (ell_max * (ell_max + 2) - ell_min**2) * (1 + 2 * mp_max) + 2 * mp_max + 1


def WignerDrange(ell_min, mp_max, ell_max=-1):
    """Create an array of (ℓ, m', m) indices as in 𝔇 array

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    mp_max : int, optional
        Integer satisfying 0 <= mp_max.  Default is ell_max.
    ell_max : int
        Integer satisfying 0 <= ell_min <= ell_max

    See Also
    --------
    WignerDsize : Total size of 𝔇 array
    WignerDindex : Index inside these wedges

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_min, ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    if ell_max < 0:
        ell_max = mp_max
    r = np.zeros((WignerDsize(ell_min, mp_max, ell_max), 3), dtype=np.int32)
    i = 0
    for ell in range(ell_min, ell_max+1):
        for mp in range(-min(ell, mp_max), min(ell, mp_max)+1):
            for m in range(-ell, ell+1):
                r[i, 0] = ell
                r[i, 1] = mp
                r[i, 2] = m
                i += 1
    return r

def WignerHindex(ell, mp, m, mp_max=None):
    """Index to "wedge" arrays

    Parameters
    ----------
    ell : int
    mp : int
    m : int
    mp_max : int, optional
        If None, it is assumed to be at least ell

    See Also
    --------
    WignerHsize : Total size of wedge array
    WignerHrange : Array of (ℓ, m', m) indices corresponding to this wedge

    Notes
    -----
    Here, it is assumed that only data with m≥|m'| are stored, and only
    corresponding values are passed.  We also assume |m|≤ell and |m'|≤ell.  Neither
    of these are checked.  The wedge array that this function indexes is ordered as

        [
            H(ell, mp, m) for ell in range(ell_max+1)
            for mp in range(-min(ell, mp_max), min(ell, mp_max)+1)
            for m in range(abs(mp), ell+1)
        ]

    """
    if ell == 0:
        return 0
    mpmax = ell
    if mp_max is not None:
        mpmax = min(mp_max, mpmax)
    if m < -mp:
        if m < mp:
            return _WignerHindex(ell, -mp, -m, mpmax)
        else:
            return _WignerHindex(ell, -m, -mp, mpmax)
    else:
        if m < mp:
            return _WignerHindex(ell, m, mp, mpmax)
        else:
            return _WignerHindex(ell, mp, m, mpmax)


def WignerHsize(mp_max, ell_max=-2):
    """Total size of array of wedges of width mp_max up to ell_max

    Parameters
    ----------
    ell_max : int
    mp_max : int, optional
        If None, it is assumed to be at least ell

    See Also
    --------
    WignerHrange : Array of (ℓ, m', m) indices corresponding to this wedge
    WignerHindex : Index inside these wedges

    Notes
    -----
    Here, it is assumed that only data with m≥|m'| are stored, and only
    corresponding values are passed.  We also assume |m|≤ell and |m'|≤ell.  Neither
    of these are checked.  The wedge array that this function indexes is ordered as

        [
            H(ell, mp, m) for ell in range(ell_max+1)
            for mp in range(-min(ell, mp_max), min(ell, mp_max)+1)
            for m in range(abs(mp), ell+1)
        ]

    """
    if ell_max == -2:
        ell_max = mp_max
    elif ell_max < 0:
        return 0
    if mp_max is None or mp_max >= ell_max:
        return (ell_max+1) * (ell_max+2) * (2*ell_max+3) // 6
    else:
        return ((ell_max+1) * (ell_max+2) * (2*ell_max+3) - 2*(ell_max-mp_max)*(ell_max-mp_max+1)*(ell_max-mp_max+2)) // 6


def _WignerHindex(ell, mp, m, mp_max):
    """Helper function for `WignerHindex`"""
    mp_max = min(mp_max, ell)
    i = WignerHsize(mp_max, ell-1)  # total size of everything with smaller ell
    if mp<1:
        i += (mp_max + mp) * (2*ell - mp_max + mp + 1) // 2  # size of wedge to the left of m'
    else:
        i += (mp_max + 1) * (2*ell - mp_max + 2) // 2  # size of entire left half of wedge
        i += (mp - 1) * (2*ell - mp + 2) // 2  # size of right half of wedge to the left of m'
    i += m - abs(mp)  # size of column in wedge between m and |m'|
    return i

def precompute_indices(n_max, mp_max):
    """Precompute required indices for WignerHindex and nm_index before CUDA execution."""
    WignerH_indices = np.zeros(n_max + 2, dtype=np.int32)
    nm_indices = np.zeros(n_max + 2, dtype=np.int32)

    for n in range(1, n_max + 2):
        WignerH_indices[n] = WignerHindex(n, 0, n, mp_max)  # Equivalent to WignerHindex(n, 0, n, mp_max)
        if n == n_max+1: #last element
            WignerH_indices[n] = n_max+1
    for n in range(1, n_max + 2):
        nm_indices[n] = n + n * (n + 1)  # Equivalent to nm_index(n, n)

    return WignerH_indices, nm_indices


def _fill_wigner_D_numba(Hwedge, 
                        z_apowers_real, z_apowers_imag, 
                        z_ypowers_real, z_ypowers_imag, 
                        res,
                        H_indices, z_a_id, z_y_id, z_a_conj_id, z_y_conj_id, 
                        out_real, out_imag, truncate):
    
    threads_per_block = (16, 64)
    batch_size = Hwedge.shape[0]
    loop_size = z_a_id.shape[0]
    #blocks_per_grid = (res.shape[0] + threads_per_block - 1) // threads_per_block
    blocks_per_grid = ((batch_size + threads_per_block[0] - 1) // threads_per_block[0], 
                     (loop_size + threads_per_block[1] - 1) // threads_per_block[1])
    
    fill_wigner_D_numba[blocks_per_grid, threads_per_block](
        Hwedge, 
        z_apowers_real, z_apowers_imag, 
        z_ypowers_real, z_ypowers_imag, 
        res,
        H_indices, z_a_id, z_y_id, z_a_conj_id, z_y_conj_id, 
        out_real, out_imag, 
        truncate
    )
    return out_real, out_imag 

@cuda.jit 
def fill_wigner_D_numba(Hwedge, 
                        z_apowers_real, z_apowers_imag, 
                        z_ypowers_real, z_ypowers_imag, 
                        res,
                        H_indices, z_a_id, z_y_id, z_a_conj_id, z_y_conj_id, 
                        out_real, out_imag, truncate):
    batch_idx, loop_idx = cuda.grid(2)
    if batch_idx < Hwedge.shape[0] and loop_idx < truncate:
        #for loop_idx in range(z_a_id.shape[0]):
            # Load values for real and imaginary parts
        a_real = z_apowers_real[batch_idx, z_a_id[loop_idx], z_a_conj_id[loop_idx]]
        a_imag = z_apowers_imag[batch_idx, z_a_id[loop_idx], z_a_conj_id[loop_idx]]
        
        b_real = z_ypowers_real[batch_idx, z_y_id[loop_idx], z_y_conj_id[loop_idx]]
        b_imag = z_ypowers_imag[batch_idx, z_y_id[loop_idx], z_y_conj_id[loop_idx]]

        h = Hwedge[batch_idx, H_indices[loop_idx]]

        # Complex multiplication: (a + ib) * (c + id)
        real_part = (a_real * b_real - a_imag * b_imag)  # ac - bd
        imag_part = (a_real * b_imag + a_imag * b_real)  # ad + bc
        
        # Multiply with `res` and `Hwedge`
        out_real[batch_idx,loop_idx] = res[loop_idx] * h * real_part 
        out_imag[batch_idx,loop_idx] = res[loop_idx] * h * imag_part 


def _fill_wigner_D_numba_eval(Hwedge, 
                        z_apowers_real, z_apowers_imag, 
                        z_ypowers_real, z_ypowers_imag, 
                        res,
                        H_indices, z_a_id, z_y_id, z_a_conj_id, z_y_conj_id, 
                        coeffs, result, num_candidates):
    
    batch_size = Hwedge.shape[0]
    loop_size = z_a_id.shape[0]
    (grid, block, shmem) = choose_dims(batch_size, loop_size, prefer_bx=256, max_by=2)
    def _ensure_dev_arr(name, a):
        assert a is not None, f"{name} is None"
        import numpy as np
        from numba import cuda
        # Accept NumPy on host or Numba device array
        assert isinstance(a, (np.ndarray, cuda.cudadrv.devicearray.DeviceNDArray)), \
            f"{name} has wrong type: {type(a)}"
        # (optional) sanity checks
        assert a.size > 0, f"{name} is empty"
        return True

    _ensure_dev_arr("Hwedge", Hwedge)
    _ensure_dev_arr("z_apowers_real", z_apowers_real)
    _ensure_dev_arr("z_apowers_imag", z_apowers_imag)
    _ensure_dev_arr("z_ypowers_real", z_ypowers_real)
    _ensure_dev_arr("z_ypowers_imag", z_ypowers_imag)
    _ensure_dev_arr("res", res)
    _ensure_dev_arr("H_indices", H_indices)  # ← likely!
    _ensure_dev_arr("z_a_id", z_a_id)
    _ensure_dev_arr("z_y_id", z_y_id)
    _ensure_dev_arr("z_a_conj_id", z_a_conj_id)
    _ensure_dev_arr("z_y_conj_id", z_y_conj_id)
    _ensure_dev_arr("coeffs", coeffs)
    _ensure_dev_arr("result", result)
    assert isinstance(num_candidates, (int, np.integer)), "num_candidates must be int"
    threads_per_block = (16, 64)
    batch_size = Hwedge.shape[0]
    loop_size = z_a_id.shape[0]
    #blocks_per_grid = (res.shape[0] + threads_per_block - 1) // threads_per_block
    blocks_per_grid = ((batch_size + threads_per_block[0] - 1) // threads_per_block[0], 
                     (loop_size + threads_per_block[1] - 1) // threads_per_block[1])
    fill_wigner_D_numba_eval[blocks_per_grid, threads_per_block, 0,shmem](
        Hwedge, 
        z_apowers_real, z_apowers_imag, 
        z_ypowers_real, z_ypowers_imag, 
        res,
        H_indices, z_a_id, z_y_id, z_a_conj_id, z_y_conj_id, 
        coeffs, result,num_candidates
    )
    return result

@cuda.jit(fastmath=True)
def fill_wigner_D_numba_eval(Hwedge, 
                        z_apowers_real, z_apowers_imag, 
                        z_ypowers_real, z_ypowers_imag, 
                        res,
                        H_indices, z_a_id, z_y_id, z_a_conj_id, z_y_conj_id, 
                        coeffs, result, num_candidates):
    batch_idx,loop_idx = cuda.grid(2)
    if batch_idx >= result.shape[0]:
        return
    # bx = cuda.blockDim.x
    # by = cuda.blockDim.y
    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    # # per-(y lane) segment in shared memory; size = bx*by floats
    # smem = cuda.shared.array(0, dtype=numba.float32)
    # base = ty * bx          # offset for this y-lane
    
    logical_batch = batch_idx // num_candidates

    val = 0.0
    if loop_idx < z_a_id.shape[0]:
        a_real = z_apowers_real[batch_idx, z_a_id[loop_idx], z_a_conj_id[loop_idx]]
        a_imag = z_apowers_imag[batch_idx, z_a_id[loop_idx], z_a_conj_id[loop_idx]]
        
        b_real = z_ypowers_real[batch_idx, z_y_id[loop_idx], z_y_conj_id[loop_idx]]
        b_imag = z_ypowers_imag[batch_idx, z_y_id[loop_idx], z_y_conj_id[loop_idx]]

        h = Hwedge[batch_idx, H_indices[loop_idx]]
        s = res[loop_idx] * h
        # Complex multiplication: D = s* (a*b)
        real_part = s * (a_real * b_real - a_imag * b_imag)  # ac - bd
        imag_part = s * (a_real * b_imag + a_imag * b_real)  # ad + bc
        
        #complex multiply with coeffs and accumulate to result
        val = real_part * coeffs[logical_batch,loop_idx, 0]  \
            - imag_part * coeffs[logical_batch,loop_idx,1]
        cuda.atomic.add(result, batch_idx, val)
    # smem[base + tx] = val
    # cuda.syncthreads()
    # # Reduction in shared memory
    # stride = bx // 2
    # while stride > 0:
    #     if tx < stride:
    #         smem[base + tx] += smem[base + tx + stride]
    #     cuda.syncthreads()
    #     stride //= 2
    # # one atomic per batch per y-lane
    # if tx == 0:
    #     cuda.atomic.add(result, batch_idx, smem[base])

@cuda.jit
def stack_and_conjugate(z_real, z_imag, truncate):
    """Copy real and negative imaginary parts to the second channel

    Args:
        z_real (_type_): _description_
        z_imag (_type_): _description_
    """
    batch_idx, z_idx = cuda.grid(2)
    if batch_idx < z_real.shape[0] and z_idx < truncate:
        z_real[batch_idx, z_idx, 1] = z_real[batch_idx, z_idx, 0]
        z_imag[batch_idx, z_idx, 1] = -z_imag[batch_idx, z_idx, 0]


@cuda.jit
def _complex_powers_cuda(zravel_real, zravel_imag, M, zpowers_real, zpowers_imag):
    """CUDA version of _complex_powers that computes complex powers in parallel."""

    batch_idx = cuda.grid(1)  # Get thread index
    if batch_idx >= zravel_real.shape[0]:  # Bounds check
        return

    # Initialize first column to 1 + 0j
    zpowers_real[batch_idx, 0, 0] = 1.0
    zpowers_imag[batch_idx, 0, 0] = 0.0

    if M > 0:
        # Load zravel as a complex number (split into real and imag)
        z_real = zravel_real[batch_idx]
        z_imag = zravel_imag[batch_idx]

        θ_real = 1.0  # Real part of θ
        θ_imag = 0.0  # Imaginary part of θ

        # Mask condition (z.real < 0) | (z.imag < 0)
        while z_real < 0.0 or z_imag < 0.0:
            # Multiply θ by i (θ *= complex_i)
            new_θ_real = -θ_imag
            new_θ_imag = θ_real
            θ_real = new_θ_real
            θ_imag = new_θ_imag

            # Divide z by i (z /= complex_i)
            new_z_real = z_imag
            new_z_imag = -z_real
            z_real = new_z_real
            z_imag = new_z_imag

        # Store z into zpowers[:, 1]
        zpowers_real[batch_idx, 1, 0] = z_real
        zpowers_imag[batch_idx, 1, 0] = z_imag

        # Initialize clock = θ
        clock_real = θ_real
        clock_imag = θ_imag

        # Compute `dc = -2 * np.sqrt(z).imag ** 2`
        #sqrt_real = math.sqrt(0.5 * (math.sqrt(z_real**2 + z_imag**2) + z_real))  # Real part of sqrt(z)
        sqrt_imag = ((z_imag > 0) - (z_imag < 0)) * math.sqrt(0.5 * (math.sqrt(z_real**2 + z_imag**2) - z_real))  # Imag part

        dc = -2 * (sqrt_imag ** 2)
        t = 2 * dc

        # Compute dz = dc * (1 + 2 * zpowers[:, 1]) + 1j * sqrt(-dc * (2 + dc))
        dz_real = dc * (1 + 2 * z_real)
        dz_imag = dc * (2 * z_imag) + math.sqrt(-dc * (2 + dc))  # Complex sqrt component

        for m in range(2, M + 1):
            # zpowers[:, m] = zpowers[:, m - 1] + dz
            zpowers_real[batch_idx, m, 0] = zpowers_real[batch_idx, m - 1, 0] + dz_real
            zpowers_imag[batch_idx, m, 0] = zpowers_imag[batch_idx, m - 1, 0] + dz_imag

            # dz += t * zpowers[:, m]
            dz_real += t * zpowers_real[batch_idx, m, 0]
            dz_imag += t * zpowers_imag[batch_idx, m, 0]

            # zpowers[:, m - 1] *= clock
            temp_real = zpowers_real[batch_idx, m - 1, 0] * clock_real - zpowers_imag[batch_idx, m - 1, 0] * clock_imag
            temp_imag = zpowers_real[batch_idx, m - 1, 0] * clock_imag + zpowers_imag[batch_idx, m - 1, 0] * clock_real
            zpowers_real[batch_idx, m - 1, 0] = temp_real
            zpowers_imag[batch_idx, m - 1, 0] = temp_imag

            # clock *= θ
            temp_clock_real = clock_real * θ_real - clock_imag * θ_imag
            temp_clock_imag = clock_real * θ_imag + clock_imag * θ_real
            clock_real = temp_clock_real
            clock_imag = temp_clock_imag

        # zpowers[:, M] *= clock
        temp_real = zpowers_real[batch_idx, M, 0] * clock_real - zpowers_imag[batch_idx, M, 0] * clock_imag
        temp_imag = zpowers_real[batch_idx, M, 0] * clock_imag + zpowers_imag[batch_idx, M, 0] * clock_real
        zpowers_real[batch_idx, M, 0] = temp_real
        zpowers_imag[batch_idx, M, 0] = temp_imag


@cuda.jit
def _complex_powers_and_conjugate_cuda(zravel_real, zravel_imag, M, truncate, zpowers_real, zpowers_imag):
    """Compute complex powers and stack conjugates into channel 1."""
    batch_idx = cuda.grid(1)
    if batch_idx >= zravel_real.shape[0]:
        return

    # Initialize first column to 1 + 0j
    zpowers_real[batch_idx, 0, 0] = 1.0
    zpowers_imag[batch_idx, 0, 0] = 0.0

    if M > 0:
        z_real = zravel_real[batch_idx]
        z_imag = zravel_imag[batch_idx]

        θ_real = 1.0
        θ_imag = 0.0

        # Mask condition (z.real < 0) | (z.imag < 0)
        while z_real < 0.0 or z_imag < 0.0:
            # Multiply θ by i (θ *= complex_i)
            new_θ_real = -θ_imag
            new_θ_imag = θ_real
            θ_real = new_θ_real
            θ_imag = new_θ_imag

            # Divide z by i (z /= complex_i)
            new_z_real = z_imag
            new_z_imag = -z_real
            z_real = new_z_real
            z_imag = new_z_imag

        # Store z into zpowers[:, 1]
        zpowers_real[batch_idx, 1, 0] = z_real
        zpowers_imag[batch_idx, 1, 0] = z_imag

        # Initialize clock = θ
        clock_real = θ_real
        clock_imag = θ_imag

        # Compute `dc = -2 * np.sqrt(z).imag ** 2`
        sqrt_imag = ((z_imag > 0) - (z_imag < 0)) * math.sqrt(0.5 * (math.sqrt(z_real**2 + z_imag**2) - z_real))

        dc = -2 * (sqrt_imag ** 2)
        t = 2 * dc

        # Compute dz = dc * (1 + 2 * zpowers[:, 1]) + 1j * sqrt(-dc * (2 + dc))
        dz_real = dc * (1 + 2 * z_real)
        dz_imag = dc * (2 * z_imag) + math.sqrt(-dc * (2 + dc))

        for m in range(2, M + 1):
            # zpowers[:, m] = zpowers[:, m - 1] + dz
            zpowers_real[batch_idx, m, 0] = zpowers_real[batch_idx, m - 1, 0] + dz_real
            zpowers_imag[batch_idx, m, 0] = zpowers_imag[batch_idx, m - 1, 0] + dz_imag

            # dz += t * zpowers[:, m]
            dz_real += t * zpowers_real[batch_idx, m, 0]
            dz_imag += t * zpowers_imag[batch_idx, m, 0]

            # zpowers[:, m - 1] *= clock
            temp_real = zpowers_real[batch_idx, m - 1, 0] * clock_real - zpowers_imag[batch_idx, m - 1, 0] * clock_imag
            temp_imag = zpowers_real[batch_idx, m - 1, 0] * clock_imag + zpowers_imag[batch_idx, m - 1, 0] * clock_real
            zpowers_real[batch_idx, m - 1, 0] = temp_real
            zpowers_imag[batch_idx, m - 1, 0] = temp_imag

            # clock *= θ
            temp_clock_real = clock_real * θ_real - clock_imag * θ_imag
            temp_clock_imag = clock_real * θ_imag + clock_imag * θ_real
            clock_real = temp_clock_real
            clock_imag = temp_clock_imag

        # zpowers[:, M] *= clock
        temp_real = zpowers_real[batch_idx, M, 0] * clock_real - zpowers_imag[batch_idx, M, 0] * clock_imag
        temp_imag = zpowers_real[batch_idx, M, 0] * clock_imag + zpowers_imag[batch_idx, M, 0] * clock_real
        zpowers_real[batch_idx, M, 0] = temp_real
        zpowers_imag[batch_idx, M, 0] = temp_imag

    # Stack/conjugate into channel 1 for z_idx in [0, truncate)
    for z_idx in range(truncate):
        zpowers_real[batch_idx, z_idx, 1] = zpowers_real[batch_idx, z_idx, 0]
        zpowers_imag[batch_idx, z_idx, 1] = -zpowers_imag[batch_idx, z_idx, 0]

@cuda.jit
def _angles_to_phases_cuda(alpha, beta, gamma, a_r, a_i, b_r, b_i, g_r, g_i):
    idx = cuda.grid(1)
    if idx >= alpha.shape[0]:
        return
    a = alpha[idx]
    b = beta[idx]
    g = gamma[idx]
    a_r[idx] = math.cos(a)
    a_i[idx] = math.sin(a)
    b_r[idx] = math.cos(b)
    b_i[idx] = math.sin(b)
    g_r[idx] = math.cos(g)
    g_i[idx] = math.sin(g)


@cuda.jit
def to_euler_phases_cuda(R, z_alphas_real, z_alphas_imag, z_betas_real, z_betas_imag, z_gammas_real, z_gammas_imag):
    """
    CUDA implementation of `to_euler_phases` for batch processing.

    Parameters:
    - R: (N, 4) CUDA device array, each row contains [R[0], R[1], R[2], R[3]]
    - z_real: (N, 3) Real part of the output
    - z_imag: (N, 3) Imaginary part of the output
    """
    i = cuda.grid(1)  # Get thread index
    if i >= R.shape[0]:  # Ensure within bounds
        return

    # Compute a and b
   
    r0, r1, r2, r3 = R[i, 0], R[i, 1], R[i, 2], R[i, 3]
    a = r0**2 + r3**2
    b = r1**2 + r2**2

    # Compute square roots
    sqrta = math.sqrt(a)
    sqrtb = math.sqrt(b)

    # Compute exp[iβ] = ((a - b) + 2j * sqrta * sqrtb) / (a + b)
    denominator = a + b
    if denominator != 0:
        z_betas_real[i] = (a - b) / denominator
        z_betas_imag[i] = (2 * sqrta * sqrtb) / denominator
    else:
        z_betas_real[i] = 1.0  # Default to 1+0j
        z_betas_imag[i] = 0.0

    # Compute exp[i(α+γ)/2]
    if sqrta > 0.0:
        zp_real = R[i, 0] / sqrta
        zp_imag = R[i, 3] / sqrta
    else:
        zp_real = 1.0
        zp_imag = 0.0

    # Compute exp[i(α-γ)/2]
    if sqrtb > 0.0:
        zm_real = R[i, 2] / sqrtb
        zm_imag = -R[i, 1] / sqrtb
    else:
        zm_real = 1.0
        zm_imag = 0.0

    # Compute z[0] = zp * zm (Complex multiplication)
    z_alphas_real[i] = zp_real * zm_real - zp_imag * zm_imag
    z_alphas_imag[i] = zp_real * zm_imag + zp_imag * zm_real

    # Compute z[2] = zp * conj(zm) (Complex multiplication with conjugate)
    z_gammas_real[i] = zp_real * zm_real + zp_imag * zm_imag
    z_gammas_imag[i] = zp_real * (-zm_imag) + zp_imag * zm_real

@jit
def ϵ(m):
    if m <= 0:
        return 1
    elif m%2:
        return -1
    else:
        return 1



@cuda.jit
def step_1_2_cuda(g, h, n_max, mp_max, Hwedge, Hextra, expiβ_real, expiβ_imag, WignerH_indices, nm_indices):
    """CUDA implementation of _2 with precomputed indices."""
    batch_idx = cuda.grid(1)  # Each thread processes one batch index

    if batch_idx >= Hwedge.shape[0]:  # Avoid out-of-bounds
        return
    #step 1 
    Hwedge[batch_idx, 0] = 1.0

    cosβ = expiβ_real[batch_idx]
    sinβ = expiβ_imag[batch_idx]

    sinβ_squared = sinβ * sinβ

    if n_max > 0:
        # n = 1
        n0n_index = WignerH_indices[1]
        nn_index = nm_indices[1]
        Hwedge[batch_idx, n0n_index] = math.sqrt(3) # sqrt(3), un-normalized
        Hwedge[batch_idx, n0n_index - 1] = (g[nn_index - 1] * cosβ) * 1.0/math.sqrt(2)  # inverse_sqrt2
        
        # n = 2, ..., n_max+1
        for n in range(2, n_max + 2):
            if n <= n_max:
                n0n_index = WignerH_indices[n]
                H = Hwedge
            else:
                n0n_index = n
                H = Hextra

            nm10nm1_index = WignerH_indices[n - 1]
            nn_index = nm_indices[n]
            const = math.sqrt(1.0 + 0.5 / n)

            gi = g[nn_index - 1]

            # m = n
            H[batch_idx, n0n_index] = const * Hwedge[batch_idx, nm10nm1_index]

            # m = n-1
            H[batch_idx, n0n_index - 1] = gi * cosβ * H[batch_idx, n0n_index]

            # m = n-2, ..., 1
            for i in range(2, n):
                gi = g[nn_index - i]
                hi = h[nn_index - i]
                H[batch_idx, n0n_index - i] = gi * cosβ * H[batch_idx, n0n_index - i + 1] - hi * sinβ_squared * H[batch_idx, n0n_index - i + 2]

            # m = 0, with normalization
            const = 1.0 / math.sqrt(4 * n + 2)
            gi = g[nn_index - n]
            hi = h[nn_index - n]
            H[batch_idx, n0n_index - n] = (gi * cosβ * H[batch_idx, n0n_index - n + 1] - hi * sinβ_squared * H[batch_idx, n0n_index - n + 2]) * const

            # Now loop back and correct normalization
            prefactor = const
            for i in range(1, n):
                prefactor *= sinβ
                H[batch_idx, n0n_index - n + i] *= prefactor

        # Correct normalization of m=n elements
        prefactor = 1.0
        for n in range(1, n_max + 1):
            prefactor *= sinβ
            Hwedge[batch_idx, WignerH_indices[n]] *= prefactor / math.sqrt(4 * n + 2)
        for n in [n_max + 1]:
            prefactor *= sinβ
            Hextra[batch_idx, n] *= prefactor / math.sqrt(4 * n + 2)



@cuda.jit
def _step_3_test_cuda(b6s, b7s, a8s, inds0, inds1, 
                      b6s_extra, b7s_extra, a8s_extra, inds0_extra, inds1_extra, 
                      expiβ_real, expiβ_imag, Hwedge, Hextra, truncate, truncate_extra):
    """CUDA implementation of _step_3_test with β as a vector."""

    # Compute global indices
    batch_idx, loop_idx = cuda.grid(2)  # 2D grid: (batch index, i index)

    if batch_idx < Hwedge.shape[0]:  # Ensure valid batch index
        cosβ = expiβ_real[batch_idx]  # Get β value for this batch
        sinβ = expiβ_imag[batch_idx]
        one_plus_cosβ = 1 + cosβ
        one_minus_cosβ = 1 - cosβ

        # # First loop: Parallelized
        if loop_idx < truncate:
            Hwedge[batch_idx, inds0[loop_idx]] = (
                b6s[loop_idx] * one_minus_cosβ * Hwedge[batch_idx, inds1[loop_idx] + 2]
                - b7s[loop_idx] * one_plus_cosβ * Hwedge[batch_idx, inds1[loop_idx]]
                - a8s[loop_idx] * sinβ * Hwedge[batch_idx, inds1[loop_idx] + 1]
            )

        # Second loop: Parallelized
        if loop_idx < truncate_extra:
                Hwedge[batch_idx, inds0_extra[loop_idx]] = (
                    b6s_extra[loop_idx] * one_minus_cosβ * Hextra[batch_idx, inds1_extra[loop_idx] + 2]
                    - b7s_extra[loop_idx] * one_plus_cosβ * Hextra[batch_idx, inds1_extra[loop_idx]]
                    - a8s_extra[loop_idx] * sinβ * Hextra[batch_idx, inds1_extra[loop_idx] + 1]
                )

@cuda.jit
def _step4_test_cuda(Hwedge, d6s, d7s, d8s,inds1, inds2, inds3, inds4,  unique_ns, start_indices, end_indices, truncate):
    batch_idx, ns_idx = cuda.grid(2)  # 2D grid: (batch, identifier)
    
    if batch_idx < Hwedge.shape[0] and ns_idx < truncate:
        start = start_indices[ns_idx]
        end = end_indices[ns_idx]

        for i in range(start, end):  # Only loop over relevant indices
            Hwedge[batch_idx, inds1[i]] = (
                d6s[i] * Hwedge[batch_idx, inds2[i]]
                - d7s[i] * Hwedge[batch_idx, inds3[i]]
                + d8s[i] * Hwedge[batch_idx, inds4[i]]
            )
@cuda.jit
def _step5_test_cuda(Hwedge, d6s, d7s, d8s, inds1, inds2, inds3, inds4,  unique_ns, start_indices, end_indices, truncate):
    batch_idx, ns_idx = cuda.grid(2)  # 2D grid: (batch, identifier)

    if batch_idx < Hwedge.shape[0] and ns_idx < truncate:
        start = start_indices[ns_idx]
        end = end_indices[ns_idx]

        for i in range(start, end): # Only loop over relevant indices
            Hwedge[batch_idx,inds1[i]] = (d6s[i] * Hwedge[batch_idx,inds2[i]]
                                    + d7s[i] * Hwedge[batch_idx,inds3[i]]
                                    - d8s[i] * Hwedge[batch_idx,inds4[i]])
            
def precompute_indices_and_values_step3(a, b, n_max ,mp_max):

    # Preallocate arrays
    inds0, inds1 = [], []
    inds0_extra, inds1_extra = [], []
    b6s, b7s, a8s = [], [], []
    b6s_extra, b7s_extra, a8s_extra = [], [], []
    for n in range(1, n_max + 1):
        i1 = WignerHindex(n, 1, 1, mp_max)
        i2 = WignerHindex(n + 1, 0, 0, mp_max) if n+1 <= n_max else 0
        i3 = nm_index(n + 1, 0)
        i4 = nabsm_index(n, 1)
        inverse_b5 = 1.0 / b[i3]
        if n+1 <= n_max:
            for i in range(n):
                inds0.append(i+i1)
                inds1.append(i+i2)
                b6s.append(b[-i+i3-2]* inverse_b5* 0.5)
                b7s.append(b[i+i3] * inverse_b5 * 0.5)
                a8s.append(a[i+i4] * inverse_b5)
        else:
            for i in range(n):
                inds0_extra.append(i+i1)
                inds1_extra.append(i)
                b6s_extra.append(b[-i+i3-2]* inverse_b5* 0.5)
                b7s_extra.append(b[i+i3] * inverse_b5 * 0.5)
                a8s_extra.append(a[i+i4] * inverse_b5)
                

    b6s = cuda.to_device(np.array(b6s))
    b7s = cuda.to_device(np.array(b7s))
    a8s = cuda.to_device(np.array(a8s))
    inds0 = cuda.to_device(np.array(inds0))
    inds1 = cuda.to_device(np.array(inds1))
    b6s_extra = cuda.to_device(np.array(b6s_extra))
    b7s_extra = cuda.to_device(np.array(b7s_extra))
    a8s_extra = cuda.to_device(np.array(a8s_extra))
    inds0_extra = cuda.to_device(np.array(inds0_extra))
    inds1_extra = cuda.to_device(np.array(inds1_extra))
    

    return  inds0, inds1, b6s, b7s, a8s, b6s_extra, b7s_extra, a8s_extra, inds0_extra, inds1_extra

def precompute_indices_and_values_step4(n_max, mp_max, d):
    d6_temp, d7_temp, d8_temp = [], [], []
    inds1, inds2, inds3, inds4 = [], [], [], []
    ns = []
    for n in range(2, n_max + 1):
        for mp in range(1, min(n, mp_max)):
            # Precompute indices
            i1 = WignerHindex(n, mp+1, mp+1, mp_max) - 1
            i2 = WignerHindex(n, mp-1, mp, mp_max)
            # i3 = WignerHindex(n, mp, mp-1, mp_max)
            i3 = WignerHindex(n, mp, mp, mp_max) - 1
            i4 = WignerHindex(n, mp, mp+1, mp_max)
            i5 = nm_index(n, mp)
            i6 = nm_index(n, mp-1)

            # Precompute constants
            inverse_d5 = 1.0 / d[i5]

            for i in range(1, n - mp):
                inds1.append(i+i1)
                inds2.append(i+i2)
                inds3.append(i+i3)
                inds4.append(i+i4)
                d6_temp.append(d[i6]*inverse_d5)
                d7_temp.append(d[i+i6]*inverse_d5)
                d8_temp.append(d[i+i5]*inverse_d5)
                ns.append(n)
            i = n-mp
            d6_temp.append(d[i6]*inverse_d5)
            d7_temp.append(d[i + i6]*inverse_d5)
            d8_temp.append(0.0)
            inds1.append(i+i1)
            inds2.append(i+i2)
            inds3.append(i+i3)
            inds4.append(i+i4)
            ns.append(n)
    inds1 = cuda.to_device(np.array(inds1))
    inds2 = cuda.to_device(np.array(inds2))
    inds3 = cuda.to_device(np.array(inds3))
    inds4 = cuda.to_device(np.array(inds4))
    d6_temp = cuda.to_device(np.array(d6_temp))
    d7_temp = cuda.to_device(np.array(d7_temp))
    d8_temp = cuda.to_device(np.array(d8_temp))
    ns = np.array(ns)
    
    return inds1, inds2, inds3, inds4, d6_temp, d7_temp, d8_temp, ns


@jit
def nm_index(n, m):
    """Return flat index into arrray of [n, m] pairs.

    Assumes array is ordered as

        [
            [n, m]
            for n in range(n_max+1)
            for m in range(-n, n+1)
        ]

    """
    return m + n * (n + 1)

def precompute_indices_and_values_step5(n_max, mp_max, d):
    # Determine maximum possible length for d7 and d8
    max_len = n_max + mp_max + 1
    count = sum(min(n, mp_max) for n in range(0, n_max + 1))

    # Preallocate arrays
    
    inds1, inds2, inds3, inds4 = [], [], [], []
    d6_temp, d7_temp, d8_temp = [], [], []
    ns = []
    for n in range(0, n_max + 1):
        for mp in range(0, -min(n, mp_max), -1):
            # Precompute indices
            i1 = WignerHindex(n, mp - 1, -mp + 1, mp_max) - 1
            i2 = WignerHindex(n, mp + 1, -mp + 1, mp_max) - 1
            i3 = WignerHindex(n, mp, -mp, mp_max) - 1
            i4 = WignerHindex(n, mp, -mp + 1, mp_max)
            i5 = nm_index(n, mp - 1)
            i6 = nm_index(n, mp)
            i7 = nm_index(n, -mp - 1)
            i8 = nm_index(n, -mp)

            # Precompute constants
            inverse_d5 = 1.0 / d[i5]
            d6 = d[i6]

            
            for i in range(1, n + mp):
                inds1.append(i+i1)
                inds2.append(i+i2)
                inds3.append(i+i3)
                inds4.append(i+i4)
                d6_temp.append(d[i6]*inverse_d5)
                d7_temp.append(d[i+i7]*inverse_d5)
                d8_temp.append(d[i+i8]*inverse_d5)
                ns.append(n)
            i = n+mp
            d6_temp.append(d[i6]*inverse_d5)
            d7_temp.append(d[i + i7]*inverse_d5)
            d8_temp.append(0.0)
            inds1.append(i+i1)
            inds2.append(i+i2)
            inds3.append(i+i3)
            inds4.append(i+i4)
            ns.append(n)

            
            #print(np.intersect1d(i_range_i1, i_range_i2), np.intersect1d(i_range_i1, i_range_i3), np.intersect1d(i_range_i1, i_range_i4))
    inds1 = cuda.to_device(np.array(inds1))
    inds2 = cuda.to_device(np.array(inds2))
    inds3 = cuda.to_device(np.array(inds3))
    inds4 = cuda.to_device(np.array(inds4))
    d6_temp = cuda.to_device(np.array(d6_temp))
    d7_temp = cuda.to_device(np.array(d7_temp))
    d8_temp = cuda.to_device(np.array(d8_temp))
    ns = np.array(ns)
    return inds1, inds2, inds3, inds4, d6_temp, d7_temp, d8_temp, ns



def WignerDindex(ell, mp, m, ell_min=0, mp_max=-1):
    """Compute index into Wigner 𝔇 matrix

    Parameters
    ----------
    ell : int
        Integer satisfying ell_min <= ell <= ell_max
    mp : int
        Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
    m : int
        Integer satisfying -ell <= m <= ell
    ell_min : int, optional
        Integer satisfying 0 <= ell_min <= ell_max.  Defaults to 0.
    mp_max : int, optional
        Integer satisfying 0 <= mp_max.  Defaults to ell.

    Returns
    -------
    i : int
        Index into Wigner 𝔇 matrix arranged as described below

    See Also
    --------
    WignerDsize : Total size of the 𝔇 matrix
    WignerDrange : Array of (ℓ, m', m) indices corresponding to the 𝔇 matrix

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_min, ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    if mp_max < 0:
        mp_max = ell
    i = (mp + min(mp_max, ell)) * (2 * ell + 1) + m + ell
    if ell > ell_min:
        i += WignerDsize(ell_min, mp_max, ell-1)
    return i

import os 
from pathlib import Path

def load_precomputed_indices(ell_max, mp_max, Dsize):
    """Load precomputed indices from a .npz file."""
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    out_path = out_dir / f"wigner_matrix_ellmax{ell_max}.npz"
    if os.path.exists(out_path): 
        arrs = np.load(out_path, allow_pickle=False)
        res = arrs["res"]
        z_a_conj_id = arrs["z_a_conj_id"]
        z_a_id = arrs["z_a_id"]
        z_y_conj_id = arrs["z_y_conj_id"]
        z_y_id = arrs["z_y_id"]
        H_indices = arrs["H_indices"]
        
    else:
        print(f"Pre-computing Wigner matrix indices (ell_max={ell_max}) — this only happens once and is cached to disk.")
        indices = WignerDrange(0, ell_max)
        # precomp
        res = np.zeros(Dsize, dtype=np.float32)
        z_a_conj_id = np.zeros(Dsize, dtype = np.int32) # index into the correct zₐpowers
        z_a_id = np.zeros(Dsize, dtype = np.int32) # index into the correct zₐpowers
        z_y_conj_id = np.zeros(Dsize, dtype = np.int32) # index into the correct zᵧpowers
        z_y_id = np.zeros(Dsize, dtype = np.int32) # index into the correct zᵧpowers
        H_indices = np.zeros(Dsize, dtype = np.int32) # index into the correct H
        for i, (ell, mp, m) in enumerate(indices):
            res[i] = ϵ(mp) * ϵ(-m)
            H_indices[i] = WignerHindex(ell, mp, m, mp_max)
            if mp < 0:
                z_a_conj_id[i] = 1
                z_a_id[i] = -mp
            else:
                z_a_conj_id[i] = 0
                z_a_id[i] = mp 
            if m < 0:
                z_y_conj_id[i] = 1
                z_y_id[i] = -m
            else:
                z_y_conj_id[i] = 0
                z_y_id[i] = m
        np.savez_compressed(
            out_path,
            res=res,
            z_a_conj_id=z_a_conj_id,
            z_a_id=z_a_id,
            z_y_conj_id=z_y_conj_id,
            z_y_id=z_y_id,
            H_indices=H_indices,
        )
    return (cuda.to_device(res.astype(np.float32)),
            cuda.to_device(z_a_conj_id.astype(np.int32)),
            cuda.to_device(z_a_id.astype(np.int32)),
            cuda.to_device(z_y_conj_id.astype(np.int32)),
            cuda.to_device(z_y_id.astype(np.int32)),
            cuda.to_device(H_indices.astype(np.int32))
    )
        


class WignerDMatrices():
    def __init__(
            self, 
            ell_max, 
            batchsize, 
            num_candidates=1, 
            device=torch.device("cuda:0"), 
            mode=None
            ):
        # Initialize variables
        self.ell_max = int(ell_max)
        self.mp_max =  self.ell_max
        self.device = device
        self.mode = mode

        self.num_candidates = num_candidates
        self.batchsize = batchsize
        self.total_batch_size = self.batchsize * self.num_candidates
        self.Hsize = WignerHsize(self.mp_max, self.ell_max)
        
        self.Dsize = WignerDsize(0,self.mp_max, self.ell_max)
        n = np.array([n for n in range(self.ell_max+2) for m in range(-n, n+1)])
        m = np.array([m for n in range(self.ell_max+2) for m in range(-n, n+1)])
        absn = np.array([n for n in range(self.ell_max+2) for m in range(n+1)])
        absm = np.array([m for n in range(self.ell_max+2) for m in range(n+1)])
        self._a = np.sqrt((absn+1+absm) * (absn+1-absm) / ((2*absn+1)*(2*absn+3)))
        self._b = np.sqrt((n-m-1) * (n-m) / ((2*n-1)*(2*n+1)))
        self._b[m<0] *= -1
        self._d = 0.5 * np.sqrt((n-m) * (n+m+1))
        self._d[m<0] *= -1
        with np.errstate(divide='ignore', invalid='ignore'):
            self._g = cuda.to_device(2*(m+1) / np.sqrt((n-m)*(n+m+1)))
            self._h = cuda.to_device(np.sqrt((n+m+2)*(n-m-1) / ((n-m)*(n+m+1))))
        
        self.inds1_5, self.inds2_5, self.inds3_5, self.inds4_5, self.d6_5, self.d7_5, self.d8_5, self.ns_5 = precompute_indices_and_values_step5(self.ell_max, self.mp_max, self._d)
        self.inds1_4, self.inds2_4, self.inds3_4, self.inds4_4, self.d6_4, self.d7_4, self.d8_4, self.ns_4 = precompute_indices_and_values_step4(self.ell_max, self.mp_max, self._d)
        self.inds0, self.inds1, self.b6s,self.b7s, self.a8s, self.b6s_extra,self.b7s_extra, self.a8s_extra, self.inds0_extra, self.inds1_extra= precompute_indices_and_values_step3(self._a, self._b, self.ell_max, self.mp_max)

        unique_ns, start_indices = np.unique(self.ns_4, return_index=True)
        self.unique_ns_4 = cuda.to_device(unique_ns)
        self.start_indices_4 = cuda.to_device(start_indices)
        _, end_indices = np.unique(self.ns_4[::-1], return_index=True)
        end_indices_4 = self.ns_4.shape[0] - end_indices  # Convert from reversed index to normal index
        self.end_indices_4 = cuda.to_device(end_indices_4)
       

        unique_ns, start_indices = np.unique(self.ns_5, return_index=True)
        self.unique_ns_5 = cuda.to_device(unique_ns)
        self.start_indices_5 = cuda.to_device(start_indices)
        _, end_indices = np.unique(self.ns_5[::-1], return_index=True)
        end_indices_5 = self.ns_5.shape[0] - end_indices  # Convert from reversed index to normal index
        self.end_indices_5 = cuda.to_device(end_indices_5)


        B = self.total_batch_size
        D = self.Dsize
        H = self.Hsize
        L = self.ell_max + 1
        dev = self.device

        if self.mode == "eval":
             # eval accumulator (real only) as torch:
            self.result_real_t = torch.zeros((B,), device=dev, dtype=torch.float32)
            self.result_real_v = cuda.as_cuda_array(self.result_real_t)
        else:
            # --- OUTPUTS (torch) ---
            self.out_real_t = torch.empty((B, D), device=dev, dtype=torch.float32)
            self.out_imag_t = torch.empty_like(self.out_real_t)
            self.out_real_v = cuda.as_cuda_array(self.out_real_t)  # Numba view
            self.out_imag_v = cuda.as_cuda_array(self.out_imag_t)


        # --- INTERMEDIATES (Numba only) ---
        self.Hwedge  = cuda.device_array((B, H), dtype=np.float32)
        self.Hextra  = cuda.device_array((B, self.ell_max+2), dtype=np.float32)

        self.z_alphas_real = cuda.device_array((B,), dtype=np.float32)
        self.z_alphas_imag = cuda.device_array((B,), dtype=np.float32)
        self.z_betas_real  = cuda.device_array((B,), dtype=np.float32)
        self.z_betas_imag  = cuda.device_array((B,), dtype=np.float32)
        self.z_gammas_real = cuda.device_array((B,), dtype=np.float32)
        self.z_gammas_imag = cuda.device_array((B,), dtype=np.float32)

        self.z_alpha_powers_real = cuda.device_array((B, L, 2), dtype=np.float32)
        self.z_alpha_powers_imag = cuda.device_array((B, L, 2), dtype=np.float32)
        self.z_gamma_powers_real = cuda.device_array((B, L, 2), dtype=np.float32)
        self.z_gamma_powers_imag = cuda.device_array((B, L, 2), dtype=np.float32)
    

        # Precompute indices on CPU
        WignerH_indices, nm_indices = precompute_indices(self.ell_max, self.mp_max)
        self.d_WignerH_indices = cuda.to_device(WignerH_indices)
        self.d_nm_indices = cuda.to_device(nm_indices)
        self.res, self.z_a_conj_id, self.z_a_id, self.z_y_conj_id, self.z_y_id, self.H_indices = load_precomputed_indices(self.ell_max, self.mp_max, self.Dsize)

        # Set up CUDA kernel parameters
        self.threads_per_block_step2 = 128
        n = self.b6s.shape[0]
        self.blocks_per_grid_step2 = (self.total_batch_size + self.threads_per_block_step2 - 1) // self.threads_per_block_step2
        self.threads_per_block_step3 = (16, 16)  # Tune for performance
        self.blocks_per_grid_step3 = ((self.total_batch_size + self.threads_per_block_step3[0] - 1) // self.threads_per_block_step3[0], 
                    (n + self.threads_per_block_step3[1] - 1) // self.threads_per_block_step3[1])


    def _set_truncates(self, truncate):
        # assert  self.b6s_extra.shape[0] == ell_max
        # assert  self.b6s.shape[0] == 0.5*ell_max * (ell_max-1), ""
        # assert self.inds1_4.shape[0] == ell_max*(ell_max+1)*(ell_max-1)//6
        # assert self.inds2_5.shape[0] == ell_max*(ell_max+1)*(ell_max+2)//6
        if truncate is None:
            truncate = self.ell_max + 1
        else:
            truncate = int(truncate) + 1
        self.truncate_ell_max = min(truncate, self.ell_max)
        self.step3_truncate = min(int(0.5*truncate * (truncate-1)), self.b6s.shape[0])
        self.step3_extra_truncate = min(truncate, self.b6s_extra.shape[0])
        self.step4_truncate = min(truncate*(truncate-1)*(truncate+1)//6, self.inds1_4.shape[0])
        self.step5_truncate = min(truncate*(truncate+1)*(truncate+2)//6, self.inds1_5.shape[0])
        
    def get_Dsize(self, ell):
        return WignerDsize(0, ell, ell)

    def Dindex(self, ell, mp, m):
        """Compute index into Wigner 𝔇 matrix

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        mp : int
            Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index into Wigner 𝔇 matrix arranged as described below

        See Also
        --------
        Dsize : Total size of the 𝔇 matrix

        Notes
        -----
        This assumes that the Wigner 𝔇 matrix is arranged as

            [
                𝔇(ℓ, mp, m)
                for ℓ in range(ell_min, ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ℓ, ℓ+1)
            ]

        """
        return WignerDindex(ell, mp, m, 0, self.mp_max)
    
    def H(self,expibeta_real, expibeta_imag, Hwedge, Hextra):
        
        step_1_2_cuda[self.blocks_per_grid_step2, self.threads_per_block_step2](self._g, 
                                                                                self._h, 
                                                                                self.truncate_ell_max, 
                                                                                self.truncate_ell_max, 
                                                                                Hwedge, 
                                                                                Hextra, 
                                                                                expibeta_real, 
                                                                                expibeta_imag, 
                                                                                self.d_WignerH_indices, 
                                                                                self.d_nm_indices,
                                                                                #self.sqrt_4n2,
            )
        
        _step_3_test_cuda[self.blocks_per_grid_step3, self.threads_per_block_step3](self.b6s, 
                                                                                    self.b7s, 
                                                                                    self.a8s, 
                                                                                    self.inds0, 
                                                                                    self.inds1, 
                                                                                    self.b6s_extra, 
                                                                                    self.b7s_extra, 
                                                                                    self.a8s_extra, 
                                                                                    self.inds0_extra, 
                                                                                    self.inds1_extra, 
                                                                                    expibeta_real, 
                                                                                    expibeta_imag, 
                                                                                    Hwedge,
                                                                                    Hextra, 
                                                                                    self.step3_truncate,
                                                                                    self.step3_extra_truncate)
        
        _step4_test_cuda[self.blocks_per_grid_step3, self.threads_per_block_step3](Hwedge, 
                                                                                   self.d6_4, 
                                                                                   self.d7_4, 
                                                                                   self.d8_4, 
                                                                                   self.inds1_4, 
                                                                                   self.inds2_4, 
                                                                                   self.inds3_4, 
                                                                                   self.inds4_4,
                                                                                   self.unique_ns_4,
                                                                                   self.start_indices_4,
                                                                                   self.end_indices_4, 
                                                                                   self.truncate_ell_max)
        
        _step5_test_cuda[self.blocks_per_grid_step3, self.threads_per_block_step3](Hwedge, 
                                                                                   self.d6_5, 
                                                                                   self.d7_5, 
                                                                                   self.d8_5, 
                                                                                   self.inds1_5, 
                                                                                   self.inds2_5, 
                                                                                   self.inds3_5, 
                                                                                   self.inds4_5, 
                                                                                   self.unique_ns_5,
                                                                                   self.start_indices_5,
                                                                                   self.end_indices_5, 
                                                                                   self.truncate_ell_max)
        return Hwedge
    
    def D(self,q =None, alpha=None, beta=None, gamma=None, truncate=None):
        assert self.mode != "eval", "Use D_eval for eval mode"
        threads_per_block = 128
        blocks_per_grid= max((self.total_batch_size  + threads_per_block - 1) // threads_per_block, 432)
        self.out_real_t.zero_()
        self.out_imag_t.zero_()
        self.Hwedge[:] = 0
        if q is not None:
            
            #1) Reshape input quaternions
            q = q.reshape(-1, 4)
            q = cuda.to_device(q)

            #2) Compute Euler phases
            to_euler_phases_cuda[blocks_per_grid, threads_per_block](q, 
                                                                     self.z_alphas_real, 
                                                                     self.z_alphas_imag, 
                                                                     self.z_betas_real, 
                                                                     self.z_betas_imag, 
                                                                     self.z_gammas_real, 
                                                                     self.z_gammas_imag)
        elif alpha is not None and beta is not None and gamma is not None:

            a = alpha.view(-1).to(self.device, dtype=torch.float32).contiguous()
            b = beta.view(-1).to(self.device, dtype=torch.float32).contiguous()
            g = gamma.view(-1).to(self.device, dtype=torch.float32).contiguous()
            _angles_to_phases_cuda[blocks_per_grid, threads_per_block](
                cuda.as_cuda_array(a),
                cuda.as_cuda_array(b),
                cuda.as_cuda_array(g),
                self.z_alphas_real,
                self.z_alphas_imag,
                self.z_betas_real,
                self.z_betas_imag,
                self.z_gammas_real,
                self.z_gammas_imag,
            )
        else:
            raise ValueError("Either q or alpha, beta, gamma must be provided")
        
        self._set_truncates(truncate)
       
        #3) Compute H matrix
        self.Hwedge = self.H(self.z_betas_real, self.z_betas_imag, self.Hwedge, self.Hextra)
        
        
        
        #4) Compute complex powers + conjugates (fused)
        _complex_powers_and_conjugate_cuda[blocks_per_grid, threads_per_block](self.z_alphas_real.ravel(), 
                                                                               self.z_alphas_imag.ravel(), 
                                                                               self.truncate_ell_max, 
                                                                               self.truncate_ell_max,
                                                                               self.z_alpha_powers_real, 
                                                                               self.z_alpha_powers_imag)
        
        _complex_powers_and_conjugate_cuda[blocks_per_grid, threads_per_block](self.z_gammas_real.ravel(), 
                                                                               self.z_gammas_imag.ravel(), 
                                                                               self.truncate_ell_max,
                                                                               self.truncate_ell_max,
                                                                               self.z_gamma_powers_real, 
                                                                               self.z_gamma_powers_imag)
        
        #6) Compute Wigner D matrix
        _fill_wigner_D_numba(self.Hwedge, self.z_alpha_powers_real, 
                             self.z_alpha_powers_imag, 
                             self.z_gamma_powers_real, 
                             self.z_gamma_powers_imag, 
                             self.res, 
                             self.H_indices, 
                             self.z_a_id, 
                             self.z_y_id, 
                             self.z_a_conj_id, 
                             self.z_y_conj_id, 
                             self.out_real_v, 
                             self.out_imag_v, 
                             WignerDsize(0,self.truncate_ell_max, self.truncate_ell_max),)
        
        #7) Return the output
        if truncate == self.ell_max or truncate is None:
            D_complex = torch.complex(self.out_real_t, self.out_imag_t)
            return D_complex.view(self.batchsize, self.num_candidates, -1).to(dtype=torch.complex64)
        else:
            D_complex = torch.complex(self.out_real_t[:, :WignerDsize(0,self.truncate_ell_max-1, self.truncate_ell_max-1)], 
                                  self.out_imag_t[:, :WignerDsize(0,self.truncate_ell_max-1, self.truncate_ell_max-1)])
        return D_complex.view(self.batchsize, self.num_candidates, -1).to(dtype=torch.complex64)


    def D_eval(self,q = None, alpha=None, beta=None, gamma=None, coeffs=None):
        assert self.mode == "eval", "D_eval can only be called in eval mode"
        threads_per_block = 128
        blocks_per_grid= max((self.total_batch_size  + threads_per_block - 1) // threads_per_block, 432)
        
        if q is not None:
            
            #1) Reshape input quaternions
            q = q.reshape(-1, 4)
            q = cuda.to_device(q)
        
            #2) Compute Euler phases
            to_euler_phases_cuda[blocks_per_grid, threads_per_block](q, 
                                                                     self.z_alphas_real, 
                                                                     self.z_alphas_imag, 
                                                                     self.z_betas_real, 
                                                                     self.z_betas_imag, 
                                                                     self.z_gammas_real, 
                                                                     self.z_gammas_imag)
        elif alpha is not None and beta is not None and gamma is not None:

            a_r = torch.cos(alpha.view(-1)).to(self.device, dtype=torch.float32).contiguous()
            a_i = torch.sin(alpha.view(-1)).to(self.device, dtype=torch.float32).contiguous()
            b_r = torch.cos(beta.view(-1)).to(self.device, dtype=torch.float32).contiguous()
            b_i = torch.sin(beta.view(-1)).to(self.device, dtype=torch.float32).contiguous()
            g_r = torch.cos(gamma.view(-1)).to(self.device, dtype=torch.float32).contiguous()
            g_i = torch.sin(gamma.view(-1)).to(self.device, dtype=torch.float32).contiguous()
          
            self.z_alphas_real.copy_to_device(cuda.as_cuda_array(a_r))
            self.z_alphas_imag.copy_to_device(cuda.as_cuda_array(a_i))
            self.z_betas_real.copy_to_device(cuda.as_cuda_array(b_r))
            self.z_betas_imag.copy_to_device(cuda.as_cuda_array(b_i))
            self.z_gammas_real.copy_to_device(cuda.as_cuda_array(g_r))
            self.z_gammas_imag.copy_to_device(cuda.as_cuda_array(g_i))

           
        else:
            raise ValueError("Either q or alpha, beta, gamma must be provided")
        
        self.result_real_t.zero_()
        
        #3) Compute H matrix
        self.Hwedge = self.H(self.z_betas_real, self.z_betas_imag, self.Hwedge, self.Hextra)

        #4) Compute complex powers + conjugates (fused)
        _complex_powers_and_conjugate_cuda[blocks_per_grid, threads_per_block](self.z_alphas_real.ravel(), 
                                                                               self.z_alphas_imag.ravel(), 
                                                                               self.ell_max, 
                                                                               self.ell_max,
                                                                               self.z_alpha_powers_real, 
                                                                               self.z_alpha_powers_imag)
        
        _complex_powers_and_conjugate_cuda[blocks_per_grid, threads_per_block](self.z_gammas_real.ravel(), 
                                                                               self.z_gammas_imag.ravel(), 
                                                                               self.ell_max, 
                                                                               self.ell_max,
                                                                               self.z_gamma_powers_real, 
                                                                               self.z_gamma_powers_imag)

        _fill_wigner_D_numba_eval(self.Hwedge, self.z_alpha_powers_real, 
                             self.z_alpha_powers_imag, 
                             self.z_gamma_powers_real, 
                             self.z_gamma_powers_imag, 
                             self.res, 
                             self.H_indices, 
                             self.z_a_id, 
                             self.z_y_id, 
                             self.z_a_conj_id, 
                             self.z_y_conj_id, 
                             coeffs, 
                             self.result_real_v, self.num_candidates)        
        
        return self.result_real_t