import torch
import numpy as np
import quaternionic
import spherical

class SOFFT:
    """
    Class to perform SO3 FFT using precomputed Wigner D-matrices and PyTorch's FFT.
    """
    
    def __init__(self, L:int,
                 device: torch.device, 
                 batchsize: int,
                 oversampling_factor=1):
        
        # Define the number of samples in each Euler angle
        self.num_betas = 2*L*oversampling_factor
        self.num_alphas = 2*L*oversampling_factor
        self.num_gammas = 2*L*oversampling_factor
       
        # Store parameters
        self.batch_size = batchsize
        self.device = device
        self.L = L

        # Precompute Wigner D-matrices
        self._precompute()


    def _precompute(self):
        wigner = spherical.Wigner(self.L)
        self.Dsize = wigner.Dsize

        betas = np.linspace(0, np.pi, self.num_betas, endpoint=True) # beta in [0, np.pi]
        ds_precomputed = []
        for beta in betas:
            exp_beta = np.exp(1j*beta)
            d = wigner.d(exp_beta)
            ds_precomputed.append(d)
        ds_precomputed = np.array(ds_precomputed)

        m_list = []
        n_list = []
        
        idx = 0
        for l in range(self.L):
            for m in range(-l, l+1):
                for n in range(-l, l+1):
                    m_list.append(m % self.num_alphas)
                    n_list.append(n % self.num_gammas)
                    idx += 1

        m_fft_idx = torch.tensor(m_list, dtype=torch.long)   # (K,)
        n_fft_idx = torch.tensor(n_list, dtype=torch.long)   # (K,)
        self.ds_precomputed_t = torch.tensor(ds_precomputed).to(self.device).to(torch.complex64)  # (B, K)
        self.B = self.ds_precomputed_t.shape[0]

        lin_idx = m_fft_idx * self.num_gammas + n_fft_idx                

        self.F_flat = torch.zeros((self.batch_size,self.num_betas, self.num_alphas*self.num_gammas),
                            device=self.device, dtype=torch.complex64)             # (B, A*G)

        # Expand lin_idx to match batch dimension
        self.lin_idx_expanded = lin_idx.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.B, -1).to(self.device)        # (B, K)
    

    def eval(self, flmn_coeffs_t: torch.Tensor):
        """
        Evaluate the SO3 FFT given the FLMN coefficients.
        Parameters: 
        - flmn_coeffs_t: torch.Tensor, shape (batch_size, Dsize)
            The FLMN coefficients for each item in the batch.
        Returns: 
        - f: torch.Tensor, shape (batch_size, num_betas, num_alphas, num_gammas)
            The evaluated function on the SO3 grid.
        """
        
        # Validate input dimensions
        assert flmn_coeffs_t.ndim == 2

        # Truncate flmn_coeffs_t to Dsize if necessary
        flmn_coeffs_t = flmn_coeffs_t[:, :self.Dsize]

        # 1) Multiply ds_precomputed and flmn_coeffs over the shared index using einsum
        #    Computes beta-dependent contributions to F
        temp = torch.einsum('bk,Bk->Bbk', self.ds_precomputed_t, flmn_coeffs_t)   # (B, K)

        # 2) Scatter-add into the flat accumulator
        self.F_flat.zero_().view(self.batch_size,self.num_betas, self.num_alphas*self.num_gammas) 
        self.F_flat.scatter_add_(2, self.lin_idx_expanded, temp)               # accumulate over k

        # 3) Perform ifft2 over last two dims for alpha and gamma
        f = torch.fft.ifft2(self.F_flat.view(self.batch_size, 
                                             self.B, 
                                             self.num_alphas, 
                                             self.num_gammas), dim=(2,3)).real
        
        return f
        
    def ids_to_angles(self, ids: torch.Tensor, shape):
        """
        Convert discrete indices to Euler angles.
        Parameters:
        - ids: torch.Tensor, shape (..., 3)
            The discrete indices for beta, alpha, and gamma.
        - shape: tuple
            The shape containing (num_betas, num_alphas, num_gammas).
        Returns:
        - alphas: torch.Tensor
            The alpha angles.   
        - betas: torch.Tensor
            The beta angles.
        - gammas: torch.Tensor
            The gamma angles.
        """

        
        num_betas, num_alphas, num_gammas = shape

        betas = ids[:,:,0] * (np.pi / (num_betas-1)) #beta in [0, np.pi]
        alphas = ids[:,:,1] * (2 * np.pi / (num_alphas)) # alpha in [0, 2pi)
        gammas = ids[:,:,2] * (2 * np.pi / (num_gammas)) # gamma in [0, 2pi)

        return alphas, betas, gammas

    
    