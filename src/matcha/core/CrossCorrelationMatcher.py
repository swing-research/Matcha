import numpy as np
import spherical
import torch
import torch.nn.functional as F

from matcha.utils.volume_ops import normalise
from matcha.utils.setup_utils import compute_size
from matcha.core.FLEBasis3D import FLEBasis3D


def _to_complex(dtype):
    if dtype == torch.float32:
        return torch.complex64
    elif dtype == torch.float64:
        return torch.complex128
    else:
        raise ValueError(f"Unsupported dtype {dtype} for conversion to complex.")

def get_sht_index(m : int):
    """
    Get the Spherical Harmonics Transform (SHT) index for a given order m.
    Parameters:
    - m: int, the order of the spherical harmonic.
    Returns:
    - int, the SHT index corresponding to the order m.
    """
    # Input: m = -l, -l+1, ..., l-1, l 
    # Output: index to m = 0,-1,1,-2,2,-3,3,...,-l,l
    if m == 0:
        return 0
    if m > 0:
        return 2 * m
    if m < 0:
        return -2 * m - 1


class CrossCorrelationMatcher():
    """
    Class to compute the cross-correlation coefficients between a set of tomograms and a set of templates.
    Uses the FLEBasis3D class to compute the ball harmonics expansion and the correlation coefficients.
    """
    
    def __init__(
            self, 
            N: int,
            device : torch.device,
            expansion_epsilon: float = 1e-4,
            batchsize: int = 1,
            reduce_memory: bool = True,
            bandlimit: int = None,
            micro_batch_split: int = 2,
            dtype=torch.float32,
            radius: int = None,
            jl_zeros_path: str = None,
            cs_path: str = None,
            precision_mode: str = "accurate",
    ):
        """
        Initialize the CrossCorrelationMatcher with the given arguments.
        """
        
        # Set FLE parameters
        self.N = N
        self.fle_basis_eps = expansion_epsilon
        self.batchsize = batchsize
        self.reduce_memory = reduce_memory
        self.bandlimit = bandlimit
   
        # Determine micro-batch split based on batch size
        self.micro_batch_split = int(micro_batch_split)
        if self.micro_batch_split <= 0:
            raise ValueError("micro_batch_split must be a positive integer.")
        if self.batchsize == 1:
            self.micro_batch_split = 1
        self.micro_batch_size = max(
            1,
            min(self.batchsize, (self.batchsize + self.micro_batch_split - 1) // self.micro_batch_split),
        )
        
        # Set device and data types
        self.device = device
        self.dtype_real = dtype
        self.dtype = _to_complex(self.dtype_real)

        self.radius = radius
        
        # Initialize FLE basis
        self.fle = FLEBasis3D(
            N = self.N,
            bandlimit  = self.bandlimit,
            eps = self.fle_basis_eps,
            reduce_memory = self.reduce_memory,
            batchsize = self.micro_batch_size,
            radius = self.radius,
            device = self.device,
            dtype = self.dtype,
            jl_zeros_path = jl_zeros_path,
            cs_path = cs_path,
            precision_mode = precision_mode,
        )
    
        # Store maximum degree lmax
        self.lmax = self.fle.lmax
        
        # Precompute indices and phase factors for correlation coefficient computation
        self.alpha1_indices, self.alpha2_indices, self.phase_factors, self.wigner_indices = self._precompute_indices()
        
        # Initialize output tensors
        self.sigma = torch.zeros((self.batchsize, compute_size(self.lmax + 1)), dtype=self.dtype, device=self.device) # 3 for the 3 normalization terms
        self.bh_norms = torch.zeros((self.batchsize), dtype=self.dtype_real, device=self.device)

        # Initialize tomogram coefficients tensor, acts as a buffer
        # +1 comes from padding
        self.bh_volumes_buffer = torch.zeros((self.micro_batch_size, self.fle.ne + 1), dtype=self.dtype, device=self.device)
        

    def set_template(self, template_data: np.array, mask:torch.Tensor = None):
        """
        Set the template data and compute the spherical expansion coefficients for the template.
        Parameters: 
        - template_data: np.array, the template data to be used for the spherical expansion.
        - D: torch.Tensor, optional, the D-matrix to be used for the correlation coefficient computation.
        - cut_off_id: int, optional, the cut-off index for the correlation coefficient computation.
        """
        template_data = torch.from_numpy(template_data.copy()).to(dtype=self.dtype_real, device=self.device) # shape (N, N, N)
        #template_data = normalise_torch(template_data, mask=mask) * mask if mask is not None else template_data

        bh_template = self.fle.evaluate_t(template_data.unsqueeze(0))[0] # shape (ne)
        self.template_coefficients_alphas_torch =  F.pad(bh_template, (0, 1)) # pad with one zero entry for the last index
        self.bh_norm_template = torch.linalg.norm(bh_template) # norm over ne dimension


    def _precompute_indices(self):
        """Precompute all indices and phases for correlation coefficients."""
        # Wigner D-matrix to compute indices
        wigner = spherical.Wigner(self.lmax+1)

        # Preallocate lists to hold precomputed data
        alpha1_indices = []
        alpha2_indices = []
        phase_factors = []
        wigner_indices = []  

        # Precompute everything for all l, m, and mp
        for l in range(self.lmax + 1):
            alpha1_l = []
            alpha2_l = []
            wigner_l = []
            max_len_l = 0
            
            # Loop over m and mp to gather indices and phases
            for m in range(-l, l + 1):
                alpha1_id = self.fle.idlm_list[l][get_sht_index(-m)]
                max_len_l = max(max_len_l, len(alpha1_id))
                alpha1_l.append(alpha1_id)

            for mp in range(-l, l + 1):
                alpha2_id = self.fle.idlm_list[l][get_sht_index(-mp)]
                max_len_l = max(max_len_l, len(alpha2_id))
                alpha2_l.append(alpha2_id)

            # Precompute the Wigner Dindex for the valid range of m and mp for this l
            wigner_l = [wigner.Dindex(l, m, mp) for m in range(-l, l + 1) for mp in range(-l, l + 1)]
    
            # Pad the lists to the maximum length for uniformity
            alpha1_l = [np.pad(a, (0, max_len_l - len(a)), 'constant', constant_values=-1) for a in alpha1_l]
            alpha2_l = [np.pad(a, (0, max_len_l - len(a)), 'constant', constant_values=-1) for a in alpha2_l]

            # Precompute the phase factors for the valid range of m and mp for this l
            phase_l = np.zeros((2*l + 1, 2*l + 1), dtype=complex)
            for m_idx, m in enumerate(range(-l, l + 1)):
                for mp_idx, mp in enumerate(range(-l, l + 1)):
                    phase_l[m_idx, mp_idx] = (-1)**m * (-1)**mp

            # Store precomputed data
            alpha1_indices.append(torch.from_numpy(np.array(alpha1_l)).to(torch.long).to(self.device))
            alpha2_indices.append(torch.from_numpy(np.array(alpha2_l)).to(torch.long).to(self.device))
            phase_factors.append(torch.from_numpy(phase_l.flatten()).to(self.dtype).to(self.device))
            wigner_indices.append(torch.from_numpy(np.array(wigner_l)).to(torch.long).to(self.device))

        return alpha1_indices, alpha2_indices, phase_factors, wigner_indices
    
    def compute_sigma(self, bh_volumes: torch.Tensor, i_from:int, i_to:int):
        """
        Compute the correlation coefficient, supporting batched inputs.
        
        Parameters:
        - tomogram_coefficients_alphas: torch.Tensor, the spherical expansion coefficients of the tomograms.
        - i_from: int, starting index for the batch.
        - i_to: int, ending index for the batch."""
        # Conjugate tomogram coefficients
        bh_template = self.template_coefficients_alphas_torch
        current_batch_size = bh_volumes.shape[0]
        self.bh_volumes_buffer[:current_batch_size].zero_()
        self.bh_volumes_buffer[:current_batch_size, :-1] = torch.conj(bh_volumes)

        # Iterate over l
        for l in range(self.lmax + 1):
            # Fetch precomputed alpha1, alpha2 indices, phase factors, and Wigner indices for this l
            alpha1_ids_l = self.alpha1_indices[l]  # Indices for template coefficients
            alpha2_ids_l = self.alpha2_indices[l]  # Indices for tomogram coefficients
            phase_factors_l = self.phase_factors[l]  # Precomputed phase factors (tensor)
            wigner_indices_l = self.wigner_indices[l]  # Wigner indices for Clmm'
            
            template_coeffs = bh_template[alpha1_ids_l]  # Shape: (len(alpha1_ids_l))
            tomogram_coeffs = self.bh_volumes_buffer[:current_batch_size, alpha2_ids_l]  # Shape: (batch_size, len(alpha2_ids_l))

            # Compute the correlation coefficients for all m and mp in one go
            alp_matrix = torch.einsum('cb,mab->mac', template_coeffs, tomogram_coeffs)  # Shape: (batch_size, len(alpha1_ids_l), len(alpha2_ids_l))
        
            # Apply precomputed phase factors and store the result in Clmm_prime
            self.sigma[i_from:i_to,wigner_indices_l] = (phase_factors_l * alp_matrix.flatten(start_dim=1))
    
    def get_sigma(self, volumes: torch.Tensor):
        """
        Compute the cross-correlation coefficients sigma for a batch of volumes.
        Parameters:
        - volumes: torch.Tensor, the input tomograms to compute the coefficients for.
        Returns:
        - sigma: torch.Tensor, the computed cross-correlation coefficients.
        - bh_volumes: torch.Tensor, the ball harmonics expansion of the input volumes.
        """
        
        num_volumes = volumes.shape[0]
        if num_volumes > self.batchsize:
            raise ValueError(
                f"Input batch size {num_volumes} exceeds configured batchsize {self.batchsize}."
            )

        # Reset outputs to avoid stale values across repeated calls.
        self.sigma.zero_()
        self.bh_norms.zero_()

        #Micro-batching to reduce memory consumption
        for i in range(0, num_volumes, self.micro_batch_size):
            i_to = min(i + self.micro_batch_size, num_volumes)
            # Slice the input volumes for the current micro-batch
            batch_volumes = volumes[i:i_to]

            # Compute ball harmonics expansion for the current micro-batch
            bh_volumes = self.fle.evaluate_t(batch_volumes) 

            # Compute norms of the ball harmonics volumes
            self.bh_norms[i:i_to] = torch.linalg.norm(bh_volumes, dim=1)
            
            # Compute correlation coefficient vector for the current micro-batch
            self.compute_sigma(bh_volumes, i, i_to)

        return self.sigma, self.bh_norms 
