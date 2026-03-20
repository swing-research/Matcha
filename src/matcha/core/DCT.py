## This code is based on the implementation of the DCT in this repo https://github.com/zh217/torch-dct

# Adjustemts were made to handle batches and data types. 

import numpy as np
import torch

def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def idct_irfft_impl(V):
    
    V = V.contiguous()
    
    V = torch.view_as_complex(V)
    
    V = torch.fft.irfft(V, n=V.shape[1], dim=1)
    
    return V
    

def dct(x:torch.Tensor, norm=None, dtype = torch.float64):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter norm, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    Parameters:
    - m: torch.Tensor, input tensor.
    - norm: str or None, normalization mode.
    - dtype: torch.dtype, data type for computations.
    Returns:
    - V: torch.Tensor, DCT-II of the input tensor.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X:torch.Tensor, norm=None):
    """
    Memory-efficient Inverse DCT-II (Type III).
    Parameters:
    - X: torch.Tensor, input tensor in the DCT domain.
    - norm: str or None, normalization mode.
    Returns:
    - x: torch.Tensor, inverse DCT of the input tensor.
    """
    x_shape = X.shape
    N = x_shape[-1]
    
    # Reshape to 2D (avoid unnecessary copies)
    X_v = X.contiguous().view(-1, N)
    X_v.div_(2)  # In-place division
   
    # Apply normalization
    if norm == 'ortho':
        X_v[:, 0].mul_(np.sqrt(N) * 2)  # In-place multiplication
        X_v[:, 1:].mul_(np.sqrt(N / 2) * 2)
    
    # Compute DCT basis (reuse the same tensor to avoid new allocations)
    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * (np.pi / (2 * N))
    W_r = torch.cos(k)  # Real part
    W_i = torch.sin(k)  # Imaginary part
    
    # Construct complex V_t (avoid unnecessary cat operation)
    V_t_r = X_v
    V_t_i = torch.zeros_like(V_t_r)  # Allocate only onc
    V_t_i[:, 1:].sub_(X_v.flip([1])[:, :-1])  # In-place subtraction
   
    # Combine into complex representation for FFT
    V = torch.stack([torch.mul(V_t_r, W_r).sub_(torch.mul(V_t_i, W_i)), torch.mul(V_t_r, W_i).add_(torch.mul(V_t_i, W_r))], dim=2)  # Efficient complex representation
    
    # Compute iFFT (assuming `idct_irfft_impl` is optimized)
    v = idct_irfft_impl(V)
    
    # Allocate result tensor in-place
    x = torch.zeros_like(v)  # No need for `new_zeros`
   
    # Reconstruct original signal (avoiding redundant flips)
    half_N = N // 2
    x[:, ::2].add_(v[:, :N - half_N])  # In-place addition
    x[:, 1::2].add_(v.flip([1])[:, :half_N])  # In-place addition
    
    return x.view(*x_shape)  # Reshape back to original shape
