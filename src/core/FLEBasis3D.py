"""
Fourier–Laguerre Expansion (FLE) basis for 3D volumes on the ball.

Adapted from the reference implementation by Oscar Mickelin:
  https://github.com/oscarmickelin/fle_3d

Based on:
  Kileel, J., Marshall, N. F., Mickelin, O., & Singer, A. (2025).
  Fast Expansion Into Harmonics on the Ball.
  SIAM Journal on Scientific Computing.
  https://epubs.siam.org/doi/10.1137/24M1668159
"""

import numpy as np
import scipy.special as spl
import scipy.sparse as spr
import cufinufft
import torch
from scipy.io import loadmat
import torch.nn.functional as F

from core.DCT import dct, idct

class FLEBasis3D:
    #
    #   N               basis for N x N x N volumes
    #   bandlimit       bandlimit parameter (scaled so that N is max suggested)
    #   eps             requested relative precision
    #   maxitr          maximum number of iterations for the expand method (if not specified, pre-tuned values are used)
    #   maxfun          maximum number of basis functions to use (if not specified, which is the default, the number implied by the choice of bandlimit is used)
    #   max_l           use only indices l <= max_l, if not None (default).
    #   mode            choose either "real" or "complex" (default) output
    #   sph_harm_solver solver to use for spherical harmonics expansions.
    #                   Choose either "nvidia_torch" (default) or "FastTransforms.jl".
    #   reduce_memory   If True, reduces the number of radial points in defining
    #                   NUFFT grids, and does an alternative interpolation to
    #                   compensate. To reproduce the tables and figures of the
    #                   paper, set this to False. 
    def __init__(
        self,
        N,
        bandlimit,
        eps,
        maxitr=None,
        maxfun=None,
        max_l=None,
        mode="complex",
        sph_harm_solver="nvidia_torch",
        reduce_memory=True,
        batchsize = 1, 
        radius = 1, 
        device = 'cuda', 
        dtype = torch.complex128,
        jl_zeros_path="jl_zeros_l=3000_k=2500.mat",
        cs_path="cs_l=3000_k=2500.mat",
        precision_mode="accurate",
    ):
        self.batchsize = batchsize
        realmode = mode == "real"
        complexmode = mode == "complex"
        assert realmode or complexmode

        self.complexmode = complexmode
        self.sph_harm_solver = sph_harm_solver
        self.reduce_memory = reduce_memory
        self.radius = radius
        self.device = device
        self.jl_zeros_path = jl_zeros_path
        self.cs_path = cs_path
        self.precision_mode = precision_mode


        if not maxitr:
            maxitr = 1 + int(6 * np.log2(N))

        numsparse = 32
        if eps >= 1e-10:
            numsparse = 32 
            if not maxitr:
                maxitr = 1 + int(3 * np.log2(N))

        if eps >= 1e-7:
            numsparse = 16
            if not maxitr:
                maxitr = 1 + int(2 * np.log2(N))
 

        if eps >= 1e-4:
            numsparse = 8
            if not maxitr:
                maxitr = 1 + int(np.log2(N))


        self.maxitr = maxitr
        assert dtype in [torch.complex64, torch.complex128]
        self.dtype = dtype
        if self.dtype == torch.complex64:
            self.dtype_secondary = torch.float32
        else:
            self.dtype_secondary = torch.float64
        self.W = self.precomp(
            N,
            bandlimit,
            eps,
            maxitr,
            numsparse,
            maxfun=maxfun,
            max_l=max_l
        )

    def precomp(
        self,
        N,
        bandlimit,
        eps,
        maxitr,
        numsparse,
        maxfun=None,
        max_l=None
    ):

        # Original dimension
        self.N1 = N
        # If dimensions are odd, add one (will be zero-padding)
        N = N + (N % 2)

        # Either use maxfun or estimate an upper bound on maxfun based on N
        if maxfun:
            ne = maxfun
        else:
            # approximate number of pixels in the ball of radius N/2
            ne = int(N**3 * np.pi / 6)

        ls, ks, ms, mds, lmds, cs, ne = self.lap_eig_ball(
            ne, bandlimit, max_l=max_l
        )
        self.lmds = lmds
        self.ne = ne
        self.cs = cs
        
        self.cs_torch = torch.tensor(cs, device=self.device, dtype=self.dtype_secondary)
        self.ks = ks
        self.ls = ls
        self.ms = ms

        self.c2r = self.precomp_transform_complex_to_real(ms)
        self.r2c = spr.csr_matrix(self.c2r.transpose().conj())

        # max k,l,m
        kmax = np.max(np.abs(ks))
        lmax = np.max(np.abs(ls))
        mmax = np.max(np.abs(ms))

        self.kmax = kmax
        self.lmax = lmax
        self.mmax = mmax

        
        epsdis = (
            eps
            / 4
            / (
                np.pi**2
                * (3 / 2) ** (1 / 4)
                * (3 + np.pi / 2 * np.log(5.3 * N))
            )
        )

        Q = int(np.ceil(max(5.3 * N, np.log2(1 / epsdis))))

        tmp = 1 / (np.sqrt(4 * np.pi))

        for Q2 in range(1, Q):
            tmp = tmp / Q2 * ((np.sqrt(3) * np.pi / 16) ** (2 / 3) * (N + 1))
            if tmp < epsdis:
                break

        n_radial = int(Q2)

        if self.reduce_memory:
            n_radial = int(np.ceil(int(Q2) * 0.65))

        S = int(
            max(
                np.ceil(
                    2 * np.exp(1) * 6 ** (1 / 3) * np.pi ** (2 / 3) * (N // 2)
                ),
                4 * np.log2(27.6 / epsdis),
            )
        )

        for S2 in range(1, S):
            tmp = np.exp(1) * lmds[-1] / (2 * (S2 // 2 + 1) + 3)
            tmp2 = (
                28
                / 27
                * np.sqrt(2 * self.lmax + 1)
                * (np.exp(1) * lmds[-1]) ** (3 / 2)
                * (tmp ** (S2 // 2 - 1 / 2))
                * 1
                / (1 - tmp)
            )

            if tmp2 < epsdis:
                if tmp < 1:  
                    break

        S = max(S2, 2 * self.lmax, 18)

        if self.precision_mode == "fast":
            epsnufH = 1e-4
        else:
            epsnufH = max(
                eps
                / (2 * np.pi ** (3 / 2) * (3 / 2) ** (1 / 4))
                / (2 + np.pi / 2 * np.log(Q)),
                1.1e-15,
            )
        epsnuf = max(
            1
            / 4
            * np.sqrt(np.pi)
            * eps
            / (np.pi ** (2) * (3 / 2) ** (1 / 4))
            / (3 + np.pi / 2 * np.log(Q)),
            1.1e-15,
        )



        if self.sph_harm_solver == "nvidia_torch":
            
            import torch_harmonics as th

            self.step2 = self.step2_torch
            self.step2_H = self.step2_H_torch
            self.torch = torch
            self.th = th

            n_phi = S + 1
             
            n_theta = S  

            ##### Added this to make the symmetry trick in step 1 work
            if n_phi % 2 == 1:
                n_phi += 1

            phi = 2 * np.pi * np.arange(n_phi // 2) / n_phi

            # grid = "legendre-gauss"
            grid = "equiangular"
            if grid == "equiangular":
                cost, weights = self.th.quadrature.clenshaw_curtiss_weights(
                    n_theta, -1, 1
                )
            elif grid == "legendre-gauss":
                cost, weights = self.th.quadrature.legendre_gauss_weights(
                    n_theta, -1, 1
                )
            if torch.is_tensor(cost):
                cost = cost.cpu().numpy()
            theta = np.flip(np.arccos(cost))

            #device = "cpu"
            device = self.device
            
            sht = self.th.RealSHT(
                n_theta, n_phi, lmax=self.lmax + 1, grid=grid, csphase=True
            ).to(device)
            isht = self.th.InverseRealSHT(
                n_theta,
                n_phi,
                lmax=self.lmax + 1,
                mmax=self.lmax + 1,
                grid=grid,
                csphase=True,
            ).to(device)
            
            self.sht = sht
            self.isht = isht
            #self.device = device
            self.weights = weights
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, device=device, dtype=self.dtype_secondary)
            
            self.weights_torch = weights.view(1, -1, 1).conj()
            

        elif self.sph_harm_solver == "FastTransforms.jl":
            from juliacall import Main as jl

            jl.seval("using FastSphericalHarmonics")
            jl.seval("using FastTransforms")
            self.jl = jl
            self.step2 = self.step2_fastTransforms
            self.step2_H = self.step2_H_fastTransforms


            n_phi = 2 * S - 1
            n_theta = S


            phi = 2 * np.pi * np.arange(n_phi) / n_phi
            theta = np.pi * (np.arange(n_theta) + 0.5) / n_theta  # uniform
            mu = jl.FastTransforms.chebyshevmoments1(jl.Float64, n_theta)
            self.weights = jl.FastTransforms.fejerweights1(mu)

        self.phi = phi
        self.theta = theta
        n_interp = n_radial
        if self.reduce_memory:
            n_interp = 2 * n_radial

        self.n_radial = n_radial
        self.n_phi = n_phi
        self.n_theta = n_theta
        self.n_interp = n_interp

        self.N = N

        mdmax = np.max(mds)
        self.mdmax = mdmax
        self.mds = mds

        # Make a list of lists: idx_list[l] is the index of all
        # sequential index values i with \ell value l, used for the interpolation
        idx_list = [None] * (lmax + 1)

        for i in range(lmax + 1):
            idx_list[i] = []
        for i in range(ne):
            l = ls[i]
            if ms[i] == 0:
                idx_list[l].append(i)

        self.idx_list = idx_list

        # self.idlm_list[l][m] contains all the sequential
        # indices i with physical indices l,m
        idlm_list = [
            [None for _ in range(2 * self.lmax + 1)]
            for _ in range(self.lmax + 1)
        ]
        for l in range(self.lmax + 1):
            for md in range(2 * self.lmax + 1):
                idlm_list[l][md] = []

        idlm_list_torch =[
            [None for _ in range(2 * self.lmax + 1)]
            for _ in range(self.lmax + 1)
        ]
        for l in range(self.lmax + 1):
            for md in range(2 * self.lmax + 1):
                idlm_list_torch[l][md] = []

        for i in range(ne):
            l = ls[i]
            md = mds[i]
            idlm_list[l][md].append(i)
            idlm_list_torch[l][md].append(i)
        
        for l in range(self.lmax + 1):
            for md in range(2 * self.lmax + 1):
                idlm_list_torch[l][md] = torch.tensor(idlm_list_torch[l][md], device=self.device, dtype=torch.int64)

        self.idlm_list = idlm_list
        self.idlm_list_torch = idlm_list_torch

        ######## Create NUFFT plans
        lmd0 = np.min(lmds)
        lmd1 = np.max(lmds)

        if ne == 1:
            lmd1 = lmd1 * (1 + 2e-16)
        tmp_pts = 1 - (2 * np.arange(n_radial) + 1) / (2 * n_radial)
        tmp_pts = np.cos(np.pi * tmp_pts)
        pts = (tmp_pts + 1) / 2
        pts = (lmd1 - lmd0) * pts + lmd0

        pts = pts.reshape(-1, 1)
        R = N // 2

        h = 1/ self.radius ** (1.5)

        self.R = R
        self.N = N
        self.h = h
        self.h_torch = torch.tensor(h, device=self.device, dtype=self.dtype_secondary)

        phi = phi.reshape(1, -1)
        theta = theta.reshape(-1, 1)
        x = np.cos(phi) * np.sin(theta)
        x = x.reshape(1, -1)
        y = np.sin(phi) * np.sin(theta)
        y = y.reshape(1, -1)
        z = np.ones(phi.shape) * np.cos(theta)
        z = z.reshape(1, -1)

        x = x * pts / self.radius
        y = y * pts /  self.radius
        z = z * pts /  self.radius

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

        nufft_type = 2
        self.plan2_batched = cufinufft.Plan(
            nufft_type,
            (N, N, N),
            n_trans=self.batchsize,
            eps=epsnufH,
            isign=-1,
            dtype = str(self.dtype).split(".")[-1],
            gpu_device_id=self.device.index,
            gpu_method = 2
        )

        x_gpu = torch.tensor(x, device=self.device).to(self.dtype_secondary)
        y_gpu = torch.tensor(y, device=self.device).to(self.dtype_secondary)
        z_gpu = torch.tensor(z, device=self.device).to(self.dtype_secondary)
        
        self.plan2_batched.setpts(x_gpu, y_gpu, z_gpu)
        nufft_type = 1
        epsnuf = 0.001
        

        self.plan1_batched = cufinufft.Plan(
            nufft_type,
            (N, N, N),
            n_trans=self.batchsize,
            eps=epsnuf,
            isign=1,
            dtype=str(self.dtype).split(".")[-1],
            gpu_device_id=self.device.index,
            gpu_method = 2
        )

        self.plan1_batched.setpts(x_gpu, y_gpu, z_gpu)
     
        # Source points for interpolation, i.e., Chebyshev nodes in the radial direction
        # The way we set up the interpolation below is with source and target radii
        # between 0 and 1, so we use xs and not the variable pts defined above.
        xs = 1 - (2 * np.arange(n_interp) + 1) / (2 * n_interp)
        xs = np.cos(np.pi * xs)

        if numsparse <= 0:
            ws = self.get_weights(xs)

        A3 = [None] * (lmax + 1)
        A3_torch = [None] * (lmax + 1)
        A3_T = [None] * (lmax + 1)
        A3_T_torch = [None] * (lmax + 1)

        b_sz = (n_interp, 2 * lmax + 1)
        b = np.zeros(b_sz)
        for i in range(lmax + 1):
            # Source function values
            ys = b[:, i]
            ys = ys.flatten()

            # Target points for interpolation, i.e., \lambda_{\ell k}
            # for all k that are included after truncation by lambda
            x = 2 * (lmds[idx_list[i]] - lmd0) / (lmd1 - lmd0) - 1

            _, x_ind, _ = np.intersect1d(x, xs, return_indices=True)
            x[x_ind] = x[x_ind] + 2e-16

            n = len(x)
            mm = len(xs)

            # if s is less than or equal to 0 we just do dense
            if numsparse > 0:
                
                A3[i], A3_T[i], A3_torch[i] , A3_T_torch[i] = self.barycentric_interp_sparse(
                    x, xs, ys, numsparse
                ) 

            else:
                A3[i] = np.zeros((n, mm))
                denom = np.zeros(n)
                for j in range(mm):
                    xdiff = x - xs[j]
                    temp = ws[j] / xdiff
                    A3[i][:, j] = temp.flatten()
                    denom = denom + temp
                denom = denom.reshape(-1, 1)
                A3[i] = A3[i] / denom
                A3_T[i] = A3[i].T
                A3_torch[i] = torch.tensor(A3[i], device=self.device, dtype=self.dtype)
                A3_T_torch[i] = torch.tensor(A3_T[i], device=self.device, dtype=self.dtype)
            
        self.A3 = A3
        self.A3_torch = A3_torch
        self.A3_T = A3_T
        self.A3_T_torch = A3_T_torch

        # Set up indices indicating the complement of the unit ball
        xtmp = np.arange(-R, R + N % 2)
        ytmp = np.arange(-R, R + N % 2)
        ztmp = np.arange(-R, R + N % 2)
        xstmp, ystmp, zstmp = np.meshgrid(xtmp, ytmp, ztmp)
        xstmp = xstmp /  self.radius
        ystmp = ystmp /  self.radius
        zstmp = zstmp /  self.radius
        
        rstmp = np.sqrt(xstmp**2 + ystmp**2 + zstmp**2)
        idx = rstmp > 1 + 1e-13

        self.idx = idx
        self.idx_torch = torch.tensor(idx, device=self.device)
        self.precomp_factors_step2 = (1j ** torch.arange(int(self.lmax) + 1, device=device).view(1, -1, 1)) / (4 * torch.pi)
        #  #      for l in range(self.lmax + 1):
           # b[:, l, :] = (
           #     (-1j) ** l / (4 * np.pi) * b[:, l, :] * 2 * np.pi / self.n_phi
           # )
        self.precomp_factors_step2_H = ((-1j)**torch.arange(int(self.lmax) + 1, device=device).view(1, -1, 1)) / (4 * torch.pi) * 2 * torch.pi / self.n_phi
        # Efficiently gather rhs values
        self.precomputed_indices = [
            {
                "row_indices": torch.cat([
                    torch.arange(len(self.idlm_list[l][md])).to(self.device) for md in range(2 * l + 1)
                ]),
                "col_indices": torch.cat([
                    torch.full((len(self.idlm_list[l][md]),), md, dtype=torch.long).to(self.device) for md in range(2 * l + 1)
                ])
            }
            for l in range(self.lmax + 1)
        ]

        # Find the maximum number of rows and columns
        max_rows = max(A.shape[0] for A in self.A3_torch)
        max_cols = max(A.shape[1] for A in self.A3_torch)

        self.signs = torch.tensor([(-1) ** (k + 1) for k in range(self.lmax)], dtype=torch.int32, device=self.device).view(1, 1, -1)
        
        
    
    def _precompute_indices(self):
        max_elements = max(
            len(self.idlm_list[l][md])
            for l in range(self.lmax + 1)
            for md in range(2 * l + 1)
        )
        precomputed_indices = []

        for l in range(self.lmax + 1):
            row_indices = []
            col_indices = []
            for md in range(2 * l + 1):
                size = len(self.idlm_list[l][md])
                row_indices.append(torch.arange(size, dtype=torch.long))
                col_indices.append(torch.full((size,), md, dtype=torch.long))

            # Pad to the maximum size with the dummy entry index (last index in tmp)
            padded_row = torch.cat(
                row_indices + [torch.full((max_elements - len(row_indices[0]),), max_elements, dtype=torch.long)]
            )
            padded_col = torch.cat(
                col_indices + [torch.full((max_elements - len(col_indices[0]),), 0, dtype=torch.long)]
            )
            precomputed_indices.append({"row_indices": padded_row, "col_indices": padded_col})

        return precomputed_indices, max_elements
    
    def create_denseB(self, numthread=1):
        #####
        # NOTE THE FOLLOWING ISSUE WITH SPL.SPH_HARM:
        # https://github.com/scipy/scipy/issues/7778
        # We therefore use pyshtools for large m,
        # although pyshtools is slower. The cutoff m = 75
        # works for N <= 64. For larger N, make the
        # cutoff 75 smaller if you want to compute the dense matrix
        #####
        if self.N > 32:
            from pyshtools.expand import spharm_lm

        psi = [None] * self.ne
        for i in range(self.ne):
            l = self.ls[i]
            m = self.ms[i]
            lmd = self.lmds[i]
            c = self.cs[i]


            if (np.abs(m) <= 75) or (self.N <= 32):
                if m >= 0:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * spl.sph_harm(m, l, p, t)
                        * (r <= 1)
                    )
                else:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * (-1) ** int(m)
                        * np.conj(spl.sph_harm(np.abs(m), l, p, t))
                        * (r <= 1)
                    )

            else:
                if m >= 0:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * spharm_lm(
                            l,
                            m,
                            t,
                            p,
                            kind="complex",
                            degrees=False,
                            csphase=-1,
                            normalization="ortho",
                        )
                        * (r <= 1)
                    )
                else:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * (-1) ** int(m)
                        * np.conj(
                            spharm_lm(
                                l,
                                np.abs(m),
                                t,
                                p,
                                kind="complex",
                                degrees=False,
                                csphase=-1,
                                normalization="ortho",
                            )
                        )
                        * (r <= 1)
                    )
        self.psi = psi

        # Evaluate eigenfunctions
        R = self.N // 2
        h = 1 / R ** (1.5)
        x = np.arange(-R, R + self.N % 2)
        y = np.arange(-R, R + self.N % 2)
        z = np.arange(-R, R + self.N % 2)
        ys, xs, zs = np.meshgrid(
            x, y, z
        )  # this gives the same ordering as NUFFT
        xs = xs / R
        ys = ys / R
        zs = zs / R
        rs = np.sqrt(xs**2 + ys**2 + zs**2)
        ps = np.arctan2(ys, xs)
        ps = ps + 2 * np.pi * (
            ps < 0
        )  # changes the phi definition interval from (-pi, pi) to (0,2*pi)
        ts = np.arctan2(np.sqrt(xs**2 + ys**2), zs)

        # Compute in parallel if numthread > 1
        
        from joblib import Parallel, delayed

        if numthread <= 1:
            B = np.zeros(
                (self.N, self.N, self.N, self.ne),
                dtype=np.complex128,
                order="F",
            )
            for i in range(self.ne):
                B[:, :, :, i] = self.psi[i](rs, ts, ps)
            B = h * B
        else:
            func = lambda i, rs=rs, ts=ts: self.psi[i](rs, ts, ps)
            B_list = Parallel(n_jobs=numthread, prefer="threads")(
                delayed(func)(i) for i in range(self.ne)
            )
            B_par = np.zeros(
                (self.N, self.N, self.N, self.ne),
                dtype=np.complex128,
                order="F",
            )
            for i in range(self.ne):
                B_par[:, :, :, i] = B_list[i]
            B = h * B_par

        if self.N > self.N1:
            B = B[: self.N1, : self.N1, : self.N1, :]
        B = B.reshape(self.N1**3, self.ne)

        if not self.complexmode:
            B = self.transform_complex_to_real(B, self.ms)

        return B.reshape(self.N1**3, self.ne)

    def lap_eig_ball(self, ne, bandlimit, max_l=None):
        # Computes dense matrix representation of the basis transform,
        # using the complex representation of the basis.

        # number of roots to check

        if not max_l:
            max_l = int(2.5 * ne ** (1 / 3))
        max_k = int(2.5 * ne ** (1 / 3))

        # preallocate
        ls = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        ks = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        ms = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        mds = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        cs = np.zeros(
            (max_l * (2 * max_l + 1) * max_k), dtype=np.float64, order="F"
        )
        lmds = (
            np.ones((max_l * (2 * max_l + 1) * max_k), dtype=np.float64)
            * np.inf
        )

        # load table of roots of jn (the scipy code has an issue where it gets
        # stuck in an infinite loop in Newton's method as of June 2022)

        data = loadmat(self.jl_zeros_path)
        roots_table = data["roots_table"]
        
        data = loadmat(self.cs_path)
        cs_table = data["cs"]
        
        # Sweep over lkm in the following order: k in {1,kmax}, l in {0,lmax}, m in {0,-1,1,-2, ...,-l,l}

        # If we notice that for a given k and l, the current root
        # is larger than the largest one in the list (and the list has length at least ne), then all other l for same k
        # and all other k for same l will be superfluous, so l will only have to up to this particular l - 1.

        ind = 0
        stop_l = max_l
        largest_lmd = 0
        for k in range(1, max_k):
            for l in range(stop_l):
                m_range = 2 * l + 1
                ks[ind : ind + m_range] = k
                ls[ind : ind + m_range] = l
                m_indices = np.arange(m_range)
                ms[ind : ind + m_range] = (-1) ** m_indices * (
                    (m_indices + 1) // 2
                )
                mds[ind : ind + m_range] = 2 * np.abs(
                    ms[ind : ind + m_range]
                ) - (ms[ind : ind + m_range] < 0)
                new_lmd = roots_table[l, k - 1]
                lmds[ind : ind + m_range] = new_lmd
                cs[ind : ind + m_range] = cs_table[l, k - 1]
                ind += m_range
                if (ind >= ne) and (new_lmd > largest_lmd):
                    stop_l = l
                    break
                largest_lmd = max(largest_lmd, new_lmd)




        idx = np.argsort(lmds[:ind], kind="stable")


        ls = ls[idx[:ne]]
        ks = ks[idx[:ne]]
        ms = ms[idx[:ne]]
        mds = mds[idx[:ne]]
        lmds = lmds[idx[:ne]]
        cs = cs[idx[:ne]]

        if bandlimit:
            threshold = (
                bandlimit * np.pi / 2
            )

            ne = np.searchsorted(lmds, threshold, side="left") - 1

        # potentially subtract 1 from ne to keep -m, +m pairs
        if ms[ne - 1] < 0:
            ne = ne - 1

        # make sure that ne is always at least 1
        if ne <= 1:
            ne = 1

        # # take top ne values (with the new ne)
        ls = ls[:ne]
        ks = ks[:ne]
        ms = ms[:ne]
        lmds = lmds[:ne]
        mds = mds[:ne]
        cs = cs[:ne]


        return ls, ks, ms, mds, lmds, cs, ne

    def precomp_transform_complex_to_real(self, ms):
        ne = len(ms)
        nnz = np.sum(ms == 0) + 2 * np.sum(ms != 0)
        idx = np.zeros(nnz, dtype=int)
        jdx = np.zeros(nnz, dtype=int)
        vals = np.zeros(nnz, dtype=np.complex128)

        k = 0
        for i in range(ne):
            m = ms[i]
            if m == 0:
                vals[k] = 1
                idx[k] = i
                jdx[k] = i
                k = k + 1
            if m < 0:
                s = (-1) ** np.abs(m)

                vals[k] = -1j / np.sqrt(2)
                idx[k] = i
                jdx[k] = i
                k = k + 1

                vals[k] = s * 1j / np.sqrt(2)
                idx[k] = i
                jdx[k] = i + 1
                k = k + 1

                vals[k] = 1 / np.sqrt(2)
                idx[k] = i + 1
                jdx[k] = i
                k = k + 1

                vals[k] = s / (np.sqrt(2))
                idx[k] = i + 1
                jdx[k] = i + 1
                k = k + 1

        A = spr.csr_matrix(
            (vals, (idx, jdx)), shape=(ne, ne), dtype=np.complex128
        )
        return A

    def transform_complex_to_real(self, Z, ms):
        ne = Z.shape[1]
        X = np.zeros(Z.shape, dtype=np.float64)

        for i in range(ne):
            m = ms[i]
            if m == 0:
                X[:, i] = np.real(Z[:, i])
            if m < 0:
                s = (-1) ** np.abs(m)
                x0 = (Z[:, i] - s * Z[:, i + 1]) * 1j / np.sqrt(2)
                x1 = (Z[:, i] + s * Z[:, i + 1]) / (np.sqrt(2))
                X[:, i] = np.real(x0)
                X[:, i + 1] = np.real(x1)

        return X

    def transform_real_to_complex(self, X, ms):
        ne = X.shape[1]
        Z = np.zeros(X.shape, dtype=np.complex128)

        for i in range(ne):
            m = ms[i]
            if m == 0:
                Z[:, i] = X[:, i]
            if m < 0:
                s = (-1) ** np.abs(m)
                z0 = (-1j * X[:, i] + X[:, i + 1]) / np.sqrt(2)
                z1 = s * (X[:, i] + 1j * X[:, i + 1]) / np.sqrt(2)
                Z[:, i] = z0
                Z[:, i + 1] = z1

        return Z

    def get_weights(self, xs):
        m = len(xs)
        I = np.ones(m, dtype=bool)
        I[0] = False
        e = np.sum(-np.log(np.abs(xs[0] - xs[I])))
        const = np.exp(e / m)
        ws = np.zeros(m)
        I = np.ones(m, dtype=bool)
        for j in range(m):
            I[j] = False
            xt = const * (xs[j] - xs[I])
            ws[j] = 1 / np.prod(xt)
            I[j] = True

        return ws

    def evaluate(self, a):

        if not self.complexmode:
            a = self.r2c @ a.flatten()

        f = self.step1_H(self.step2_H(self.step3_H(a)))
        f = f.reshape(self.N, self.N, self.N)

        if self.N > self.N1:
            f = f[: self.N1, : self.N1, : self.N1]
        return f

    def evaluate_torch(self, a:torch.Tensor, return_step=1, starting_from=3):

        if not self.complexmode:
            a = self.r2c @ a.flatten()
        if starting_from >= 3:
            a = self.step3_H_torch(a)
        if return_step == 3:
            return a
        if starting_from >= 2:
            a = self.step2_H_alltorch(a)
        if return_step == 2:
            return a
        if starting_from >= 1:
            a = self.step1_H_torch(a)
        

        a = a.view((self.batchsize, self.N, self.N, -1))
        if self.N > self.N1:
            a = a[:,: self.N1, : self.N1, : self.N1]
        return a    

    def evaluate_t(self, f):
        f = np.copy(f).reshape(self.N1, self.N1, self.N1)


        if self.N > self.N1:
            f = np.pad(f, ((0, 1), (0, 1), (0, 1))) 

        # Remove pixels outside disk
        f[self.idx] = 0
        f = f.flatten()

        a = self.step3(self.step2(self.step1(f))) * self.h
        if not self.complexmode:
            a = self.c2r @ a.flatten()

        return a
    

    def evaluate_t(self, f:torch.tensor, return_step=3):
        if f.shape[0] == self.batchsize:
            #f = torch.clone(f).view(self.batchsize, self.N1, self.N1, self.N1)
            f = f.view(self.batchsize, self.N1, self.N1, self.N1)
        else:
            f = torch.clone(f)
            diff = self.batchsize - f.shape[0]
            f = F.pad(f, (0, 0, 0, 0, 0, 0, 0, diff))

       
        if self.N > self.N1:
            
            f = F.pad(f, ((0, 1), (0, 1), (0, 1), (0, 0)))
       
        # Remove pixels outside disk
        f[:,self.idx_torch] = 0 
        
        a = self.step1_torch(f)
        if return_step == 1:
            return a
        
        a = self.step2_alltorch(a)
        
        if return_step == 2:
            
            return a
        a = self.step3_torch(a) * self.h_torch
        if not self.complexmode:
            a = self.c2r @ a.flatten()
        #a = self.step3_torch(self.step2_alltorch(self.step1_torch(f))) * self.h_torch

        return a
    
    def get_betas(self, f, abs_square = False):
        f = np.copy(f).reshape(self.N1, self.N1, self.N1)

        if self.N > self.N1:
            f = np.pad(f, ((0, 1), (0, 1), (0, 1)))
        
        # Remove pixels outside disk
        f[self.idx] = 0
        f = f.flatten()

        a = self.step1(f)
        if abs_square:
            a = np.abs(a)**2
        b = self.step2(a)

        return b
    
    def get_alphas(self, betas, comp_type = None):
        if comp_type =="torch":
            a = self.step3_torch(betas) * self.h
        else:
            a = self.step3(betas) * self.h
        if not self.complexmode:
            a = self.c2r @ a.flatten()
        return a

    def step1(self, f,type=None):
        
        f = f.reshape(self.N, self.N, self.N)
        f = np.array(f, dtype=np.complex128)
        f = torch.tensor(f, device=self.device)

        z = np.zeros(
            (self.n_radial, self.n_theta, self.n_phi), dtype=np.complex128
        )
        z0 = self.plan2.execute(f)
        z0 =z0.cpu().numpy()
        if self.sph_harm_solver == "FastTransforms.jl":
            z = z0.reshape(self.n_radial, self.n_theta, self.n_phi)
        else:
            z0 = z0.reshape(self.n_radial, self.n_theta, self.n_phi // 2)
            z[:, :, : self.n_phi // 2] = z0 
            z[:, ::-1, self.n_phi // 2 :] = np.conj(z0)
            
        z = z.flatten()
        return z
    
    # input is torch tensor
    def step1_torch(self, f: torch.Tensor,type=None):

        # Reshape f into a 3D tensor
        f = f.to(self.dtype).contiguous().view(self.batchsize, self.N, self.N, self.N)

        z = torch.zeros((self.batchsize, self.n_radial, self.n_theta, self.n_phi), dtype=self.dtype, device=self.device)
        z0 = self.plan2_batched.execute(f)
        
        z0 = z0.view(f.shape[0], self.n_radial, self.n_theta, self.n_phi // 2) 
        z[:, :, :, :self.n_phi // 2] = z0
        z[:,:, :, self.n_phi // 2:] = torch.flip(z0, dims=[2]).conj()

        return z

    def step2_alltorch(self, z:torch.Tensor):
        # https://github.com/NVIDIA/torch-harmonics
        
        z = z.view(self.batchsize, self.n_radial, self.n_theta, self.n_phi)
        
        # # Separate real and imaginary parts of z (still on GPU)
        sht_real = self.sht(torch.real(z))
        breal = self.torch_reshape_order_t_torch(sht_real)
        
        # Perform the spherical harmonics transform (assuming `sht` works on PyTorch tensors)
        if z.dtype.is_complex:
            sht_imag = self.sht(torch.imag(z)) 
            bimag = self.torch_reshape_order_t_torch(sht_imag)
            
            real_part = breal.real - bimag.imag      # views -> real tensors (1 alloc each)
            imag_part = breal.imag + bimag.real
            out = torch.complex(real_part, imag_part)  # single complex allocation
            # scale after combining to make only one full pass
            return out.mul_(self.precomp_factors_step2)
        else:
            return breal.mul_(self.precomp_factors_step2)


    
    def step2_torch(self, z):
        # https://github.com/NVIDIA/torch-harmonics
        
        z = z.reshape(self.n_radial, self.n_theta, self.n_phi)
        # From https://arxiv.org/pdf/1202.6522.pdf, bottom of page 3, torch only
        # computes the coefficients for real-valued data and then only for m >= 0.

        breal = self.torch_reshape_order_t(
            self.sht(self.torch.DoubleTensor(np.real(z)).to(self.device))
            .cpu()
            .numpy()
        )
        bimag = self.torch_reshape_order_t(
            self.sht(self.torch.DoubleTensor(np.imag(z)).to(self.device))
            .cpu()
            .numpy()
        )

        b = breal + 1j * bimag

        for l in range(self.lmax + 1):
            b[:, l, :] = b[:, l, :] * (1j) ** l / (4 * np.pi)
      
        return b

    def step2_fastTransforms(self, z):

        z = z.reshape(self.n_radial, self.n_theta, self.n_phi)
        b = np.zeros(
            (self.n_radial, self.lmax + 1, 2 * self.lmax + 1),
            dtype=np.complex128,
        )
        for q in range(self.n_radial):
            b[q, :, :] = self.fastTransforms_reshape_order_t(
                np.complex128(self.jl.sph_transform(np.real(z[q, :, :])))
            ) + 1j * self.fastTransforms_reshape_order_t(
                np.complex128(self.jl.sph_transform(np.imag(z[q, :, :])))
            )

        for l in range(self.lmax + 1):
            b[:, l, :] = b[:, l, :] * (1j) ** l / (4 * np.pi)

        return b


    def torch_reshape_order_t(self, b):
        # converts the order of m returned by torch to 0,-1,1,-2,2,-3,3,...
        # torch only computes the coefficients for m >= 0, but can use
        # alpha_{l,-m} = (-1)**m*\overline{alpha_{l,m}} for real-valued structures.
        # We therefore separate the input to torch_step2 into real and imaginary parts.


        s = b.shape
        bn = np.zeros((s[0], s[1], 2 * s[2] - 1), dtype=np.complex128)

        bn[:, :, 0] = b[:, :, 0]

        bn[:, :, 1 : (2 * s[2] - 1) : 4] = np.conj(b[:, :, 1 : s[2] : 2]) * (-1)
        bn[:, :, 3 : (2 * s[2] - 1) : 4] = np.conj(b[:, :, 2 : s[2] : 2])
        bn[:, :, 2 : (2 * s[2]) : 2] = b[:, :, 1 : s[2]]
        return bn
    
    def torch_reshape_order_t_torch(self, b:torch.Tensor):
        # Converts the order of m returned by torch to 0, -1, 1, -2, 2, -3, 3, ...
        # Assumes b is a PyTorch complex tensor.

        s = b.shape
        bn = torch.zeros((s[0], s[1], s[2], 2 * s[3] - 1), dtype=self.dtype, device=b.device)
        bn[:, :, :, 0] = b[:, :, :, 0]
        bn[:, :, :, 1:(2 * s[3] - 1):4] = torch.conj(b[:, :, :, 1:s[3]:2]) * (-1)
        bn[:, :, :, 3:(2 * s[3] - 1):4] = torch.conj(b[:, :, :, 2:s[3]:2])
        bn[:, :, :, 2:(2 * s[3]):2] = b[:, :, :, 1:s[3]]
        return bn

    def fastTransforms_reshape_order_t(self, b):
        # converts the order of m returned by FastTransforms.jl
        # to have columns sweeping the m-index as 0,-1,1,-2,2,-3,3,...
        bn = np.zeros((self.lmax + 1, 2 * self.lmax + 1), dtype=np.complex128)
        for l in range(self.lmax + 1):
            for m in range(l + 1):
                indpos = self.jl.sph_mode(l, m)
                indneg = self.jl.sph_mode(l, -m)
                # julia has 1-indexing and the package does real-valued harmonics
                # The convention below is different from ordinary conventions for
                # real-valued harmonics, but seems to be what they are using.
                if m > 0:
                    bn[l, 2 * m] = (
                        (-1) ** m
                        / np.sqrt(2)
                        * (
                            b[indpos[1] - 1, indpos[2] - 1]
                            - 1j * b[indneg[1] - 1, indneg[2] - 1]
                        )
                    )
                    bn[l, 2 * m - 1] = (
                        1
                        / np.sqrt(2)
                        * (
                            b[indpos[1] - 1, indpos[2] - 1]
                            + 1j * b[indneg[1] - 1, indneg[2] - 1]
                        )
                    )

                else:
                    bn[l, m] = b[indpos[1] - 1, indpos[2] - 1]
        return bn
    
    def step3(self, b):
       
        
        if self.n_interp > self.n_radial:
            b = dct(b, axis=0, type=2) / (2 * self.n_radial)
           
            bz = np.zeros(b.shape)
            b = np.concatenate((b, bz), axis=0)
            b = idct(b, axis=0, type=2) * 2 * b.shape[0]
       
        a = np.zeros(self.ne, dtype=np.complex128)
      
        for l in range(self.lmax + 1):
           
            tmp = self.A3[l] @ b[:, l, :]
          
            m_range = 2 * l + 1

            inds = np.concatenate(
                [
                    self.idlm_list[l][md]
                    for md in range(m_range)
                    if self.idlm_list[l][md]
                ]
            )
            rhs = np.concatenate(
                [tmp[: len(self.idlm_list[l][md]), md] for md in range(m_range)]
            )
            a[inds] = rhs

        a = a * self.cs
        a = a.flatten()
        return a
    
    def torch_dct_wrapper(self, input):
        if len(input.shape) == 3:
            input = input.permute(1,2,0)
        else:
            input = input.permute(0,2,3,1)
        x = dct(input, dtype = self.dtype_secondary)
        if len(input.shape) == 3:
            x = x.permute(2,0,1)
        else: 
            x = x.permute(0,3,1,2)

        return x
    
    def torch_idct_wrapper(self, input):
        if len(input.shape) == 3:
            input = input.permute(1,2,0)
        else:
            input = input.permute(0,2,3,1)
        x = idct(input)
        if len(input.shape) == 3:
            x = x.permute(2,0,1)
        else: 
            x = x.permute(0,3,1,2)
        return x
    
    def step3_torch(self, b):
        if self.n_interp > self.n_radial:
            # Perform DCT (assumes a PyTorch DCT implementation or approximation is available)
          
            b = (self.torch_dct_wrapper(b.real) + 1j*self.torch_dct_wrapper(b.imag)) / (2 * self.n_radial)
           
            # Pad with zeros
            b =  torch.cat((b, torch.zeros_like(b, device=b.device)), dim=1)
            # Perform IDCT (using PyTorch implementation or approximation)
            b = (self.torch_idct_wrapper(b.real) + 1j*self.torch_idct_wrapper(b.imag)) * 2 * b.shape[1]

        # Initialize `a` once
        a = torch.zeros(self.batchsize, self.ne, dtype=self.dtype, device=b.device)

        # Precompute indices once if not already done
        if not hasattr(self, 'precomputed_inds'):
            self.precomputed_inds = [
                torch.cat([self.idlm_list_torch[l][md] for md in range(2 * l + 1) if self.idlm_list_torch[l][md] is not None]).to(self.device)
                for l in range(self.lmax + 1)
            ]
        a = self.step3b_torch(a, b, self.precomputed_inds, self.precomputed_indices)
        # Final scaling
        a = a * self.cs_torch
        a = a.flatten(start_dim=1)
        return a

    def step3b_torch(self, a,b, precomputed_inds, precomputed_indices):        
        
        for l in range(self.lmax + 1):
                tmp = torch.matmul(self.A3_torch[l], b[:,:, l, :])
               
                indices = self.precomputed_indices[l] 

                #rhs = tmp.index_select(1, col_indices).index_select(2, row_indices).transpose(0,1)

                # In-place update
                a.index_add_(1, self.precomputed_inds[l], tmp[:,indices["row_indices"], indices["col_indices"]])
        return a

    def step1_H(self, z):

        if self.sph_harm_solver == "FastTransforms.jl":
            # Whole z
            f = self.plan1.execute(z.flatten())
            f = f.reshape(self.N, self.N, self.N)
            f[self.idx] = 0
            f = f.flatten()
        else:
            # Half z
            z = z[:, :, : self.n_phi // 2]
            z = torch.tensor(z, dtype=self.dtype).to(self.device)
            f = self.plan1.execute(z.flatten()).cpu().numpy()
            f = 2 * np.real(f)
            f = f.reshape(self.N, self.N, self.N)
            f[self.idx] = 0
            f = f.flatten()

        return f

    def step1_H_torch(self, z):
        # Half z
        z = z[:, :, :,: self.n_phi // 2].flatten(start_dim=1)
        f = self.plan1_batched.execute(z)
        f = 2 * torch.real(f)
        f = f.reshape(self.batchsize,self.N, self.N, self.N)
        f[:,self.idx_torch] = 0
        return f

    def step2_H_torch(self, b):
        for l in range(self.lmax + 1):
            b[:, l, :] = (
                (-1j) ** l / (4 * np.pi) * b[:, l, :] * 2 * np.pi / self.n_phi
            )
        
        b1, b2 = self.torch_reshape_order(b)
        b1 = self.torch.tensor(b1, dtype=self.torch.complex128)
        b2 = self.torch.tensor(b2, dtype=self.torch.complex128)
        z1 = np.complex128(self.isht(b1.to(self.device)).cpu().numpy())
        z2 = np.complex128(self.isht(b2.to(self.device)).cpu().numpy())
        z = z1 + 1j * z2

        for i in range(len(self.weights)):
            z[:, i, :] = z[:, i, :] * np.conj(self.weights[i])

        return z

    def step2_H_alltorch(self, b:torch.Tensor):
        b1, b2 = self.torch_reshape_order_torch(b* self.precomp_factors_step2_H)
        return (self.isht(b1) + 1j * self.isht(b2))*self.weights_torch


    def step2_H_fastTransforms(self, b):

        for l in range(self.lmax + 1):
            b[:, l, :] = (
                (-1j) ** l / (4 * np.pi) * b[:, l, :] * 2 * np.pi / self.n_phi
            )

        z = np.zeros(
            (self.n_radial, self.n_theta, self.n_phi), dtype=np.complex128
        )
        bq = self.fastTransforms_reshape_order(b)

        for q in range(self.n_radial):
            b1 = bq[q, :, :]
            z[q, :, :] = np.complex128(self.jl.sph_evaluate(self.jl.Matrix(b1)))

        for i in range(len(self.weights)):
            z[:, i, :] = z[:, i, :] * np.conj(self.weights[i])

        return z

    def step3_H(self, a):
        a = a * self.h
        a = a.flatten()
        a = a * self.cs

 
        b = np.zeros(
            (self.n_interp, self.lmax + 1, 2 * self.lmax + 1),
            dtype=np.complex128,
            order="F",
        )

        for l in range(self.lmax + 1):
            m_range = 2 * l + 1
            for md in range(m_range):
                b[:, l, md] = (
                    np.conj(self.A3_T[l][:, : len(self.idlm_list[l][md])])
                    @ a[self.idlm_list[l][md]]
                )

        if self.n_interp > self.n_radial:
            b = dct(b, axis=0, type=2)
            b = b[: self.n_radial, :]
            b = idct(b, axis=0, type=2)

        return b


    def step3_H_torch(self, a):

        a = a * self.h_torch
        a = a.flatten(start_dim=1)
        a = a * self.cs_torch
 

        if not hasattr(self, 'b_buffer'):
            b_tmp = np.zeros(
            (self.batchsize,self.n_interp, self.lmax + 1, 2 * self.lmax + 1),
            dtype=str(self.dtype).split(".")[-1],
            order="F",
            )
            self.b_buffer = torch.from_numpy(b_tmp).to(self.device)
        b = self.b_buffer        
        
        for l in range(self.lmax + 1):
            m_range = 2 * l + 1
            for md in range(m_range):
                b[:,:, l, md] = (
                    a[:,self.idlm_list_torch[l][md]] @
                    torch.conj(self.A3_T_torch[l][:, : len(self.idlm_list_torch[l][md])]).T
                     
                )

        if self.n_interp > self.n_radial:
            b = self.torch_dct_wrapper(b.real) + 1j*self.torch_dct_wrapper(b.imag)
        
            b = b[:,:self.n_radial]
            #Perform IDCT (using PyTorch implementation)
            b = self.torch_idct_wrapper(b.real) + 1j*self.torch_idct_wrapper(b.imag)

        return b

    def torch_reshape_order(self, b):
        # converts the column order of b from 0,-1,1,-2,2,-3,3,...
        # to the order required by torch.
        # torch only computes the coefficients for m >= 0, but can use
        # alpha_{l,-m} = (-1)**m*\overline{alpha_{l,m}} for real-valued structures.
        # We therefore separated the input to torch_step2 into real and imaginary parts,
        # and now need to invert this separation

        s = b.shape
        tmp1 = b[:, :, 0 : s[2] : 2]
        # Every other one here has to have their sign flipped
        signs = [(-1) ** (k + 1) for k in range(self.lmax)]
    
        tmp2 = np.concatenate(
            (b[:, :, 0].reshape(s[0], s[1], 1), b[:, :, 1 : s[2] : 2] * signs),
            axis=2,
        )
        b1 = 0.5 * np.real(tmp1 + tmp2) + 1j * 0.5 * np.imag(tmp1 - tmp2)
        b2 = 0.5 * np.imag(tmp1 + tmp2) - 1j * 0.5 * np.real(tmp1 - tmp2)

        return b1, b2

    
    def torch_reshape_order_torch(self, b):
        """
        Converts the column order of b from 0, -1, 1, -2, 2, -3, 3, ...
        to the order required by torch.

        Args:
            b (torch.Tensor): Input tensor with shape (batch, channels, columns).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two tensors `b1` and `b2` with the reshaped order.
        """
        # Ensure the input is a torch tensor
        if not isinstance(b, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        s = b.shape

        # Extract every other column starting from the first
        tmp1 = b[:,:, :, 0:s[3]:2]
       

        # Extract every other column starting from the second and apply signs
        tmp2 = torch.cat(
            (b[:, :,:, 0].unsqueeze(3), b[:,:, :, 1:s[3]:2] * self.signs),
            dim=3,
        )

        # Compute the reshaped components
        b1 = 0.5 * (torch.real(tmp1) + torch.real(tmp2)) + 1j * 0.5 * (torch.imag(tmp1) - torch.imag(tmp2))
        b2 = 0.5 * (torch.imag(tmp1) + torch.imag(tmp2)) - 1j * 0.5 * (torch.real(tmp1) - torch.real(tmp2))

        return b1, b2

    def fastTransforms_reshape_order(self, b):
        # converts the order with columns sweeping the m-index as 0,-1,1,-2,2,-3,3,...
        # to that required by FastTransforms.jl
        s = b.shape
        bn = np.zeros((s[0], self.n_theta, self.n_phi), dtype=np.complex128)
        for l in range(self.lmax + 1):
            for m in range(l + 1):
                indpos = self.jl.sph_mode(l, m)
                indneg = self.jl.sph_mode(l, -m)
                # julia has 1-indexing and the package does real-valued harmonics
                # The convention below is different from ordinary conventions for
                # real-valued harmonics, but seems to be what they are using.
                if m > 0:
                    bn[:, indpos[1] - 1, indpos[2] - 1] = (
                        1
                        / np.sqrt(2)
                        * ((-1) ** m * b[:, l, 2 * m] + b[:, l, 2 * m - 1])
                    )
                    bn[:, indneg[1] - 1, indneg[2] - 1] = (
                        1
                        / np.sqrt(2)
                        * (
                            1j * (-1) ** m * b[:, l, 2 * m]
                            - 1j * b[:, l, 2 * m - 1]
                        )
                    )

                else:
                    bn[:, l, m] = b[:, l, m]
        return bn

    def expand(self, f):
        b = self.evaluate_t(f)
        a0 = b
        for iter in range(self.maxitr):
            a0 = a0 - self.evaluate_t(self.evaluate(a0)) + b

        return a0

    def lowpass(self, a, bandlimit):
        threshold = (
            bandlimit * np.pi / 2
        )  
        ne = np.searchsorted(self.lmds, threshold, side="left") - 1
        a[ne::] = 0
        return a

    def barycentric_interp_sparse(self, x, xs, ys, s):
        # https://people.maths.ox.ac.uk/trefethen/barycentric.pdf

        n = len(x)
        m = len(xs)

        # Modify points by 2e-16 to avoid division by zero
        vals, x_ind, xs_ind = np.intersect1d(
            x, xs, return_indices=True, assume_unique=True
        )
        x[x_ind] = x[x_ind] + 2e-16

        idx = np.zeros((n, s))
        jdx = np.zeros((n, s))
        vals = np.zeros((n, s))
        xss = np.zeros((n, s))
        idps = np.zeros((n, s))
        numer = np.zeros((n, 1))
        denom = np.zeros((n, 1))
        temp = np.zeros((n, 1))
        ws = np.zeros((n, s))
        xdiff = np.zeros(n)
        for i in range(n):
            # get a kind of balanced interval around our point
            k = np.searchsorted(x[i] < xs, True)

            idp = np.arange(k - s // 2, k + (s + 1) // 2)
            if idp[0] < 0:
                idp = np.arange(s)
            if idp[-1] >= m:
                idp = np.arange(m - s, m)
            xss[i, :] = xs[idp]
            jdx[i, :] = idp
            idx[i, :] = i

        x = x.reshape(-1, 1)
        Iw = np.ones(s, dtype=bool)
        ew = np.zeros((n, 1))
        xtw = np.zeros((n, s - 1))

        Iw[0] = False
        const = np.zeros((n, 1))
        for j in range(s):
            ew = np.sum(
                -np.log(np.abs(xss[:, 0].reshape(-1, 1) - xss[:, Iw])), axis=1
            )
            constw = np.exp(ew / s)
            constw = constw.reshape(-1, 1)
            const += constw
        const = const / s

        for j in range(s):
            Iw[j] = False
            xtw = const * (xss[:, j].reshape(-1, 1) - xss[:, Iw])
            ws[:, j] = 1 / np.prod(xtw, axis=1)
            Iw[j] = True

        xdiff = xdiff.flatten()
        x = x.flatten()
        temp = temp.flatten()
        denom = denom.flatten()
        for j in range(s):
            xdiff = x - xss[:, j]
            temp = ws[:, j] / xdiff
            vals[:, j] = vals[:, j] + temp
            denom = denom + temp
        vals = vals / denom.reshape(-1, 1)
        
        vals = vals.flatten()
        idx = idx.flatten()
        jdx = jdx.flatten()
        A = spr.csr_matrix((vals, (idx, jdx)), shape=(n, m), dtype=np.float64)
        A_T = spr.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=np.float64)

        indices = torch.tensor(np.array([idx, jdx]), dtype=torch.int64)  # Shape: (2, nnz)
        values = torch.tensor(vals, dtype=torch.float64)       # Shape: (nnz,)
        A_torch = torch.sparse_coo_tensor(indices, values, size=(n, m), dtype=self.dtype).to(self.device)
        A_T_torch = A_torch.transpose(0, 1)

        return A, A_T, A_torch.to_dense(), A_T_torch.to_dense()
