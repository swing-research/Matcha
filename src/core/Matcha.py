import torch
from torch.nn import functional as F
from core.SOFFT import SOFFT
from core.WignerDMatrices import WignerDMatrices
from core.WignerDGradients import WignerDGradients

from ml_collections import ConfigDict
from typing import Tuple    

# Helper functions for optimization steps
def _get_newton_step(gradient, hessian, lr = 1):
    return lr * torch.linalg.solve(hessian, gradient)

def _get_gradient_step(gradient, lr = 1e-2):
    return -lr * gradient


# Reduce over candidates to get the best orientation per batch
def reduce_over_candidates(alphas, betas, gammas, scores):
    scores, idx = torch.max(scores, dim=-1)
    alphas = torch.gather(alphas, 1, idx.unsqueeze(-1)).squeeze(-1)
    betas = torch.gather(betas, 1, idx.unsqueeze(-1)).squeeze(-1)
    gammas = torch.gather(gammas, 1, idx.unsqueeze(-1)).squeeze(-1)
    return alphas, betas, gammas, scores


@torch.jit.script
def sparse_topk_per_batch_jit(
    z_scores: torch.Tensor,
    max_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes top-k coordinates for each batch in a 4D tensor of z-scores.
    Parameters:
    - z_scores: torch.Tensor, shape (B, X, Y, Z), tensor of z-scores
    - max_k: int, maximum number of top-k coordinates to return
    Returns:
    - topk_coords: torch.Tensor, shape (B, max_k, 3), coordinates of the top-k candidates
    """
    assert z_scores.ndim == 4, "z_scores must be a 4D tensor"
    B, X, Y, Z = z_scores.shape
    device = z_scores.device
    flat_scores = z_scores.reshape(B, -1)

    k = min(max_k, flat_scores.shape[1])
    topk_values, topk_idx = torch.topk(flat_scores, k, dim=1)

    topk_coords = torch.full((B, max_k, 3), -1, dtype=torch.long, device=device)
    topk_scores = torch.full((B, max_k), -1.0, dtype=flat_scores.dtype, device=device)

    z_idx = topk_idx % Z
    y_idx = (topk_idx // Z) % Y
    x_idx = topk_idx // (Y * Z)
    coords = torch.stack((x_idx, y_idx, z_idx), dim=-1)

    topk_coords[:, :k] = coords
    topk_scores[:, :k] = topk_values
    return topk_coords, topk_scores



class Matcha:
    
    def __init__(
            self, 
            batchsize: int, 
            device: torch.device,
            L_max: int,
            matcha_config:ConfigDict = None
            ):

        matcha_config = self._assert_config(matcha_config)
        self.device = device

        self.Lmax = L_max    
        self.Li = sorted(matcha_config.Li) + [self.Lmax]
        self.L0 = min(self.Li)
        

        self.do_random_sampling = matcha_config.do_random_sampling
        self.batchsize = batchsize
        self.candidates = matcha_config.candidates

        self.reinits = matcha_config.reinits
        self.reinits_iterations = matcha_config.reinits_iterations
        self.num_steps = matcha_config.num_steps
        self.stop_early = matcha_config.stop_early
        self.num_templates = 1

        self.step_type = matcha_config.step_type  # 'gradient' or 'newton'
        self.base_lr = 1 if self.step_type == 'newton' else 5e-7
        

        self.SOFFT = SOFFT(
            L = self.L0,
            device = self.device,
            batchsize = self.batchsize,
            oversampling_factor=matcha_config.oversampling_factor_K
        )

        self.WignerDMatrices = WignerDMatrices(
            ell_max= self.Lmax,
            batchsize=self.batchsize,
            device = self.device,
            num_candidates= self.candidates * self.reinits
        )

        self.WignerDGradients = WignerDGradients(
            wigner = self.WignerDMatrices,
            device = self.device
        )
        self._truncate_dsize_cache = {
            int(Li): int(self.WignerDMatrices.get_Dsize(int(Li)))
            for Li in set(self.Li)
        }
        self._printed_newton_singular_warn = True

        pass

    def _assert_config(self, matcha_config: ConfigDict):
        if matcha_config is None:
            matcha_config = ConfigDict()

        # Keys with that throw error if missing
        required_keys = [
            'Li',
        ]
        for key in required_keys:
            if key not in matcha_config:
                raise ValueError(f"Matcha config missing required key: '{key}'")


        # Check nexesaary keys in matcha_config and set defaults
        necessary_keys = {
            'candidates': 10,
            'reinits': 1,
            'num_steps': 5,
            'step_type': 'newton',
            'stop_early': False,
            "do_random_sampling": False,
            "reinits_iterations": 0,
            "oversampling_factor_K": 2,

        }
        for key, default_value in necessary_keys.items():
            if key not in matcha_config:
                matcha_config[key] = default_value
        return matcha_config

    @torch.no_grad()
    def _take_step(
            self,
            D_tensor: torch.Tensor,
            alphas: torch.Tensor,
            betas: torch.Tensor,
            gammas: torch.Tensor,
            step: torch.Tensor,
            coeffs: torch.Tensor,
            curr_max_val: torch.Tensor,
            curr_max_alpha: torch.Tensor,
            curr_max_beta: torch.Tensor,
            curr_max_gamma: torch.Tensor,
            truncate_ell: int,
            stop_early: bool = False,
            lr: torch.Tensor = None,
        ):

        # Apply the computed step to the current angles
        target_alphas = alphas - step[..., 0]
        target_betas = betas - step[..., 1]
        target_gammas = gammas - step[..., 2]

        # Compute Wigner-D and pad
        D_target = self.WignerDMatrices.D(
            alpha=target_alphas,
            beta=target_betas,
            gamma=target_gammas,
            truncate=truncate_ell
        ).to(dtype=coeffs.dtype)

        D_target = F.pad(D_target, (0, 1), value=0)

        # Perform the batched matrix multiplication
        evaled = torch.einsum('nbc,nc->nb', D_target, coeffs).real

        # Update curr_max and curr_max_angle conditionally
        update_mask = evaled > curr_max_val

        # Rescale learning rate per candidate
        lr[~update_mask] *= 0.5
        lr[update_mask] *= 1.1

        target_alphas[~update_mask] = alphas[~update_mask]
        target_betas[~update_mask] = betas[~update_mask]
        target_gammas[~update_mask] = gammas[~update_mask]
        D_target[~update_mask] = D_tensor[~update_mask]

        if stop_early and update_mask.sum().item() == 0:
            return D_target, target_alphas, target_betas, target_gammas, True, lr

        curr_max_val.masked_scatter_(update_mask, evaled[update_mask])
        curr_max_alpha.masked_scatter_(update_mask, target_alphas[update_mask])
        curr_max_beta.masked_scatter_(update_mask, target_betas[update_mask])
        curr_max_gamma.masked_scatter_(update_mask, target_gammas[update_mask])
        return D_target, target_alphas, target_betas, target_gammas, False, lr

    @torch.no_grad()
    def _grid_search_so3fft(self,sigma): 
        """
        Perform a grid search over Euler angles to find the best candidates.
        Parameters:
        - sigma: torch.Tensor, coefficients for the spherical harmonics
        Returns:
        - alphas: torch.Tensor, alpha angles of the best candidates
        - betas: torch.Tensor, beta angles of the best candidates
        - gammas: torch.Tensor, gamma angles of the best candidates
        """
        
        # Evaluate response tensor
        f = self.SOFFT.eval(
            sigma
        )
        
        # Initialize parameters
        max_candidates = self.candidates
        ids, _ = sparse_topk_per_batch_jit(
            z_scores=f, 
            max_k=max_candidates,
        )

        # determine Euler angles from ids
        alphas, betas, gammas = self.SOFFT.ids_to_angles(
            ids=ids,
            shape = f[0].shape)
        
        return alphas, betas, gammas
    

    @torch.no_grad()
    def optimize(
        self, 
        alphas: torch.Tensor,
        betas: torch.Tensor,
        gammas: torch.Tensor,
        sigma: torch.Tensor,
        truncate_ell: int,
        truncate_Dsize: int,
    ):
        """
        Perform a Newton step search over Euler angles to refine the best candidates.
        Parameters:
        - alphas: torch.Tensor, alpha angles of the best candidates
        - betas: torch.Tensor, beta angles of the best candidates
        - gammas: torch.Tensor, gamma angles of the best candidates
        - coeffs: Tuple of torch.Tensor, coefficients for the SO3 function
        - wigner: Wigner, Wigner object to compute D-matrices
        - cut_off_id: int, index to cut off coefficients
        - Gradients: torch.Tensor, gradients object for computing Newton steps
        - max_candidates: int, maximum number of candidates per subtomogram
        - reinits: int, number of reinitializations for the Newton step
        - step_size_hessian: float, step size for the Hessian
        - num_iter: int, number of iterations for the Newton step search
        - final: bool, whether to perform the final step
        Returns:
        - curr_max_alpha: torch.Tensor, refined alpha angles of the best candidates
        - curr_max_beta: torch.Tensor, refined beta angles of the best candidates
        - curr_max_gamma: torch.Tensor, refined gamma angles of the best candidates
        - curr_max_val: torch.Tensor, scores of the refined candidates
        """

        # Compute Wigner D-matrices for the current angles
        D_tensor = self.WignerDMatrices.D(
            alpha = alphas,
            beta = betas, 
            gamma = gammas, 
            truncate = truncate_ell
        ).to(dtype=sigma.dtype)
        
        D_tensor = F.pad(D_tensor, (0,1), value=0)  # Pad D-m atrices with zeros
        
        # Prepare coefficients
        sigma = F.pad(
            sigma[:, :truncate_Dsize], 
            (0,1), 
            value=0)  # Pad coefficients with zeros
        
        # Evaluate initial values
        evaled = torch.einsum('nbc,nc->nb', D_tensor, sigma).real
        
        # Initialize iterations with previous values
        curr_max_val = evaled.real
        curr_max_alpha = alphas
        curr_max_beta = betas
        curr_max_gamma = gammas
        
        # Set coefficients for gradient computation
        self.WignerDGradients.set_so3_coeffs(sigma)
        B, C = alphas.shape

        lr = self.base_lr * torch.ones(
            [B, C, 3],
            device=alphas.device, 
            dtype=alphas.dtype
            )
    
        for _ in range(self.num_steps):

            # Compute gradient and Hessian
            gradient, hessian = self.WignerDGradients.get_derivatives(
                D_tensor,
                alphas
            )
            
            # Compute step update
            if self.step_type == 'newton':
                try:
                    step = _get_newton_step(gradient, hessian, lr=lr)
                except RuntimeError as exc:
                    if "singular" not in str(exc).lower():
                        raise

                    flat_hessian = hessian.reshape(-1, hessian.shape[-2], hessian.shape[-1])
                    flat_gradient = gradient.reshape(-1, gradient.shape[-1])
                    flat_step = torch.zeros_like(flat_gradient)
                    skipped = 0

                    for idx in range(flat_hessian.shape[0]):
                        try:
                            flat_step[idx] = torch.linalg.solve(flat_hessian[idx], flat_gradient[idx])
                        except RuntimeError as inner_exc:
                            if "singular" in str(inner_exc).lower():
                                skipped += 1
                                continue
                            raise

                    if skipped > 0 and not self._printed_newton_singular_warn:
                        print(
                            "[WARN] Singular Hessian in Newton step; "
                            "skipping affected candidates and continuing."
                        )
                        self._printed_newton_singular_warn = True

                    step = lr * flat_step.reshape_as(gradient)
            else:   
                step = _get_gradient_step(gradient, lr=lr)
           
            # Take a step and evaluate
            D_tensor, alphas,betas, gammas, stop, lr = self._take_step(
                D_tensor=D_tensor,
                alphas=alphas,
                betas=betas,
                gammas=gammas,
                step=step,
                coeffs=sigma,
                curr_max_val=curr_max_val,
                curr_max_alpha=curr_max_alpha,
                curr_max_beta=curr_max_beta,
                curr_max_gamma=curr_max_gamma,
                truncate_ell=truncate_ell,
                stop_early=self.stop_early,
                lr=lr
            )

            if stop:
                break
        
        return curr_max_alpha, curr_max_beta, curr_max_gamma, curr_max_val

    @torch.no_grad()
    def search_orientations(
            self,
            sigma):

       
        # grid searhch via SO3FFT at L0 
        alphas, betas, gammas = self._grid_search_so3fft(
            sigma=sigma
        )

        # refine candidates at increasing L values
        for i, Li in enumerate(self.Li):
            is_last = i == len(self.Li) - 1

            if Li >= self.Lmax and not is_last:
                continue
            
            # compute size of truncated D-matrices 
            truncate_Dsize = self._truncate_dsize_cache[int(Li)]

            # refine via continuous optimization
            alphas, betas, gammas, scores = self.optimize(
                alphas=alphas,
                betas=betas,
                gammas=gammas,
                sigma=sigma,
                truncate_ell=Li,
                truncate_Dsize=truncate_Dsize,
            )
           
        # reduce to one orientation per batch 
        alphas, betas, gammas,scores = reduce_over_candidates(alphas, betas, gammas, scores)
            
        return alphas, betas, gammas, scores
            
