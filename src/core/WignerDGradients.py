import torch
import numpy as np
import os 
from pathlib import Path

def _precompute_Ms(wigner):
    Ms = np.zeros((2,wigner.Dsize+1)) # the +1 is to account for the fact that we append a 0 to the D component (for invalid indices)
    for j in range(wigner.ell_max+1):
        for m in range(-j, j+1):
            for m_prime in range(-j, j+1):
                idx = wigner.Dindex(j, m, m_prime)
                Ms[0,idx] = m
                Ms[1,idx] = m_prime
    
    return Ms

def _precompute_remapping(wigner):
    
    # Initialize maps
    map_p1 = torch.full((wigner.Dsize,), -1, dtype=torch.int64)
    map_n1 = torch.full((wigner.Dsize,), -1, dtype=torch.int64)
    map_p2 = torch.full((wigner.Dsize,), -1, dtype=torch.int64)
    map_n2 = torch.full((wigner.Dsize,), -1, dtype=torch.int64)

    for l in range(wigner.ell_max+1):
        for m in range(-l, l+1):
            for mp in range(-l, l+1):
                # Get current index
                id      = wigner.Dindex(l, m, mp)

                # Get shifted indices
                id_p1   = wigner.Dindex(l, m+1, mp) if m+1 <= l and m+1 >= -l else -1
                id_n1   = wigner.Dindex(l, m-1, mp) if m-1 <= l and m-1 >= -l else -1
                id_p2   = wigner.Dindex(l, m+2, mp) if m+2 <= l and m+2 >= -l else -1
                id_n2   = wigner.Dindex(l, m-2, mp) if m-2 <= l and m-2 >= -l else -1
                
                # Append shift to maps
                map_p1[id] = id_p1
                map_n1[id] = id_n1
                map_p2[id] = id_p2
                map_n2[id] = id_n2

    return map_p1, map_n1, map_p2, map_n2

def _precompute_t_factors(wigner):
    factors = np.zeros((3, wigner.Dsize+1)) # the +1 is to account for the fact that we append a 0 to the D component (for invalid indices)
    for l in range(wigner.ell_max+1):
        for m in range(-l, l+1):
            for m_prime in range(-l, l+1):
                idx = wigner.Dindex(l, m, m_prime)

                factors[0,idx] = np.sqrt((l+m)*(l-m+1)*(l+m-1)*(l-m+2)) / 4
                factors[1,idx] = - ((l+m)*(l-m+1) + (l+m+1)*(l-m)) / 4 
                factors[2,idx] = np.sqrt((l+m+1)*(l-m)*(l+m+2)*(l-m-1)) / 4

    return factors

def _get_factor_c1(j,m):
    sub_root = (j + m) * (j - m + 1)
    if sub_root < 0:
        return 0
    return -1/2 * np.sqrt(sub_root)

def _get_factor_c2(j,m):
    sub_root = (j - m) * (j + m + 1)
    if sub_root < 0:
        return 0
    return 1/2 * np.sqrt(sub_root)

def _precompute_factors(wigner):
    factors = np.zeros((2,wigner.Dsize+1)) # the +1 is to account for the fact that we append a 0 to the D component (for invalid indices)
    for j in range(wigner.ell_max+1):
        for m in range(-j, j+1):
            for m_prime in range(-j, j+1):
                idx = wigner.Dindex(j, m, m_prime)
                factors[0,idx] = _get_factor_c1(j,m)
                factors[1,idx] = _get_factor_c2(j,m)

    return factors

def _load_indices(wigner, path):
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    out_path = out_dir / f"wigner_deriv_data_ellmax{wigner.ell_max}.npz"
    if os.path.exists(out_path):
        arrs = np.load(out_path, allow_pickle=False)
        ms       = arrs["ms"]
        factors  = arrs["factors"]
        t_factors= arrs["t_factors"]
        map_p1   = arrs["map_p1"]
        map_n1   = arrs["map_n1"]
        map_p2   = arrs["map_p2"]
        map_n2   = arrs["map_n2"]

    else:
        print(f"Pre-computing Wigner derivative indices (ell_max={wigner.ell_max}) — this only happens once and is cached to disk.")
        ms = _precompute_Ms(wigner)
        factors = _precompute_factors(wigner)
        t_factors = _precompute_t_factors(wigner)
        map_p1, map_n1,map_p2, map_n2 = _precompute_remapping(wigner)
        np.savez_compressed(
            out_path,
            ms=ms,
            factors=factors,
            t_factors=t_factors,
            map_p1=map_p1,
            map_n1=map_n1,
            map_p2=map_p2,
            map_n2=map_n2,
        )
    return ms, factors, t_factors, map_p1, map_n1, map_p2, map_n2


COL = {
  'val':0, 'a':1, 'g':2, 'aa':3, 'gg':4, 'ag':5,
  'b_minus':6, 'b_plus':7,
  'ab_minus':8, 'ab_plus':9,
  'gb_minus':10, 'gb_plus':11,
  'bb_minus2':12, 'bb_0':13, 'bb_plus2':14
}
class WignerDGradients:
    """
    Class to compute gradients and Hessians of functions on SO(3) represented in Wigner D basis.
    """

    def __init__(self, wigner, device):
        # Constants
        self.device = device
        self.dtype_r = torch.float32
        self.dtype_c = torch.complex64

        # Load precomputed indices and factors
        ms, factors, t_factors, map_p1, map_n1, map_p2, map_n2 = _load_indices(wigner, device)

        # m, m' (real)
        self.m  = torch.as_tensor(ms[0], dtype=self.dtype_r, device=self.device)                # [C]
        self.mp = torch.as_tensor(ms[1], dtype=self.dtype_r, device=self.device)                # [C]

        # beta 1-step factors (your factors[0], factors[1])
        self.f_minus = torch.as_tensor(factors[0], dtype=self.dtype_c, device=self.device)      # [C]
        self.f_plus  = torch.as_tensor(factors[1], dtype=self.dtype_c, device=self.device)      # [C]

        # beta-beta 2-step factors (your t_factors[0]=t_n2, [1]=t_0, [2]=t_p2)
        self.tn2 = torch.as_tensor(t_factors[0], dtype=self.dtype_c, device=self.device)        # [C]
        self.t0  = torch.as_tensor(t_factors[1], dtype=self.dtype_c, device=self.device)        # [C]
        self.tp2 = torch.as_tensor(t_factors[2], dtype=self.dtype_c, device=self.device)        # [C]

        # Forward maps: source id -> destination id, invalid = -1
        self.idx_m_minus1 = torch.as_tensor(np.append(map_n1, 0), dtype=torch.long, device=self.device)  # [C]
        self.idx_m_plus1  = torch.as_tensor(np.append(map_p1, 0), dtype=torch.long, device=self.device)  # [C]
        self.idx_m_minus2 = torch.as_tensor(np.append(map_n2, 0), dtype=torch.long, device=self.device)  # [C]
        self.idx_m_plus2  = torch.as_tensor(np.append(map_p2,0), dtype=torch.long, device=self.device)  # [C]
            

    def set_so3_coeffs(self, so3_coeffs: torch.Tensor):
        """
        Prepares the coefficient bank for gradient and Hessian computations.
        Parameters:
        - so3_coeffs: torch.Tensor, tensor of shape [B, C] containing SO(3) coefficients.
        """

        # Effective channel count: limited by both coeff_t and precomputed buffers
        C_pre = self.m.shape[0]  # length of precomputed per-m arrays
        C_eff = min(so3_coeffs.shape[1], C_pre)

        # Work only on the active slice
        coeff = so3_coeffs[:, :C_eff]           # [B, C_eff]
        B = coeff.shape[0]
        j = 1j

        # Slice all 1D coefficient-side vectors to C_eff
        m     = self.m[:C_eff]
        mp    = self.mp[:C_eff]
        t0    = self.t0[:C_eff]
        f_minus = self.f_minus[:C_eff]
        f_plus  = self.f_plus[:C_eff]
        tn2     = self.tn2[:C_eff]
        tp2     = self.tp2[:C_eff]

        # For index maps, we need both the source slice (first C_eff entries) and to ensure dest < C_eff.
        idx_m_minus1 = self.idx_m_minus1[:C_eff]
        idx_m_plus1  = self.idx_m_plus1 [:C_eff]
        idx_m_minus2 = self.idx_m_minus2[:C_eff]
        idx_m_plus2  = self.idx_m_plus2 [:C_eff]

        # ----- m-based columns (rotation-independent) -----
        c_val = coeff
        c_a   = (j * m)  * coeff
        c_g   = (j * mp) * coeff
        c_aa  = (j * m)  * (j * m)  * coeff
        c_gg  = (j * mp) * (j * mp) * coeff
        c_ag  = (j * m)  * (j * mp) * coeff

        # ----- β columns via scatter-add (push to destinations) -----
        c_b_minus1 = torch.zeros(B, C_eff, dtype=coeff.dtype, device=coeff.device)
        c_b_plus1  = torch.zeros(B, C_eff, dtype=coeff.dtype, device=coeff.device)

        # valid sources are those with a valid destination inside [0, C_eff)
        mask_bm = (idx_m_minus1 >= 0) & (idx_m_minus1 < C_eff)
        if mask_bm.any():
            dest_bm = idx_m_minus1[mask_bm]
            src_bm  = mask_bm.nonzero(as_tuple=False).squeeze(1)
            c_b_minus1.index_add_(1, dest_bm, coeff[:, src_bm] * f_minus[src_bm])

        mask_bp = (idx_m_plus1 >= 0) & (idx_m_plus1 < C_eff)
        if mask_bp.any():
            dest_bp = idx_m_plus1[mask_bp]
            src_bp  = mask_bp.nonzero(as_tuple=False).squeeze(1)
            c_b_plus1.index_add_(1, dest_bp, coeff[:, src_bp] * f_plus[src_bp])

        # ----- ββ columns (include the 1/4 in the coeffs so runtime is just phases) -----
        c_bb_minus2 = torch.zeros(B, C_eff, dtype=coeff.dtype, device=coeff.device)
        c_bb_plus2  = torch.zeros(B, C_eff, dtype=coeff.dtype, device=coeff.device)
        c_bb_0      = t0 * coeff   # center term has no index shift

        mask_bbm = (idx_m_minus2 >= 0) & (idx_m_minus2 < C_eff)
        if mask_bbm.any():
            dest_bbm = idx_m_minus2[mask_bbm]
            src_bbm  = mask_bbm.nonzero(as_tuple=False).squeeze(1)
            c_bb_minus2.index_add_(1, dest_bbm, coeff[:, src_bbm] * tn2[src_bbm])

        mask_bbp = (idx_m_plus2 >= 0) & (idx_m_plus2 < C_eff)
        if mask_bbp.any():
            dest_bbp = idx_m_plus2[mask_bbp]
            src_bbp  = mask_bbp.nonzero(as_tuple=False).squeeze(1)
            c_bb_plus2.index_add_(1, dest_bbp, coeff[:, src_bbp] * tp2[src_bbp])

        # ----- Mixed columns -----
        c_ab_minus = (j * m)  * c_b_minus1
        c_ab_plus  = (j * m)  * c_b_plus1
        c_gb_minus = (j * mp) * c_b_minus1
        c_gb_plus  = (j * mp) * c_b_plus1

        # Stack into a bank: [B, K, C_eff] where K=15 is the number of coeff types
        self.B = B
        self.C = C_eff
        self.CoeffBank = torch.stack([
            c_val, c_a, c_g, c_aa, c_gg, c_ag,      # 0..5
            c_b_minus1, c_b_plus1,                  # 6..7
            c_ab_minus, c_ab_plus,                  # 8..9
            c_gb_minus, c_gb_plus,                  # 10..11
            c_bb_minus2, c_bb_0, c_bb_plus2         # 12..14
        ], dim=1).contiguous()

        

    def get_derivatives(self, D_mat: torch.Tensor, alpha: torch.Tensor):
        """
        Compute gradients and Hessians of the function represented by the given Wigner D coefficients.
        Parameters:
        - D_mat: torch.Tensor, shape (B, C)
            The Wigner D matrix coefficients for each item in the batch.
        - alpha: torch.Tensor, shape (B,)
            The alpha Euler angles for each item in the batch.
        Returns:
        - grad: torch.Tensor, shape (B, 3)
            The gradient vectors for each item in the batch.
        - hessian: torch.Tensor, shape (B, 3, 3)
            The Hessian matrices for each item in the batch.
        """
        
        # This is the main workhorse call of this function 
        # Evaluate all columns at once
        Y = torch.einsum('BnC,BcC->Bnc', D_mat, self.CoeffBank)   # (B, n, c) = (num_subtomograms_per_batch, reinits*candidates, num_coeffs)

        # Mix β/ββ with α-phases (match your sign convention!)
        alpha_t = torch.as_tensor(alpha, dtype=self.dtype_r, device=self.device)
        
        phase_p  = torch.exp(+1j*alpha_t)   # [R,1]
        phase_n  = torch.exp(-1j*alpha_t)
        phase_2p = torch.exp(+2j*alpha_t)
        phase_2n = torch.exp(-2j*alpha_t)
        
        # Read out evaluated columns
        # Somehow, I need to use phase_p with negativ evaluations...
        Y_bm  = Y[..., COL["b_minus"]]
        Y_bp  = Y[..., COL["b_plus"]]
        Y_abm = Y[..., COL["ab_minus"]]
        Y_abp = Y[..., COL["ab_plus"] ]
        
        beta_out = phase_p  * Y_bm + phase_n * Y_bp     # [R,1]
      
        gb_out   = phase_p  * Y[..., COL['gb_minus']] + phase_n * Y[..., COL['gb_plus']]

        alpha_beta = phase_p * ( 1j * Y_bm + Y_abm ) + \
                    phase_n * ( -1j * Y_bp + Y_abp )   

        bb_out   = Y[..., COL['bb_0']] + phase_2p * Y[..., COL['bb_minus2']] +  phase_2n * Y[..., COL['bb_plus2']]
        
        grad = torch.stack([
                Y[..., COL['a']],
                beta_out,
                Y[..., COL['g']]
                ], dim=-1).real
        
        # Hessian assembly
        hessian = torch.stack([
            torch.stack([Y[..., COL['aa']], alpha_beta, Y[..., COL['ag']]], dim=-1),
            torch.stack([alpha_beta, bb_out, gb_out], dim=-1),
            torch.stack([Y[..., COL['ag']], gb_out, Y[..., COL['gg']]], dim=-1)
        ], dim=-2).to(grad.device).real

        return grad, hessian