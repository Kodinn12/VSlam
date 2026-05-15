"""Levenberg-Marquardt pose refinement."""

import numpy as np
import math
from ..utils.logger import get_logger
from ..utils.se3_ops import se3_inv_gpu, batch_exp_se3_gpu, PoseTransform, USE_CUPY

logger = get_logger(__name__)

try:
    import cupy as cp
    if not USE_CUPY:
        raise ImportError
except ImportError:
    cp = np
    USE_CUPY = False

class LMPoseRefiner:
    def __init__(self, K, max_iter=20, lam0=1e-3, conv_delta=1e-7, huber_thresh=2.0, use_gpu=True):
        self.fx, self.fy = float(K[0,0]), float(K[1,1])
        self.cx, self.cy = float(K[0,2]), float(K[1,2])
        self.max_iter = int(max_iter)
        self.lam0 = float(lam0)
        self.conv_delta = float(conv_delta)
        self.huber_c = float(huber_thresh)
        self.use_gpu = use_gpu and USE_CUPY
        if self.use_gpu:
            self._setup_gpu_buffers()

    def _setup_gpu_buffers(self):
        M = 2500
        self.max_points = M
        self._J_proj_gpu = cp.zeros((M, 2, 3), dtype=cp.float64)
        self._J_cam_gpu = cp.zeros((M, 3, 6), dtype=cp.float64)
        self._J_full_gpu = cp.zeros((M, 2, 6), dtype=cp.float64)
        self._J_gpu = cp.zeros((M*2, 6), dtype=cp.float64)
        self._r_gpu = cp.zeros(M*2, dtype=cp.float64)
        self._W_gpu = cp.zeros(M*2, dtype=cp.float64)
        self._JtWJ_gpu = cp.zeros((6, 6), dtype=cp.float64)
        self._JtWr_gpu = cp.zeros(6, dtype=cp.float64)
        self._JtWJ_diag_gpu = cp.zeros((6, 6), dtype=cp.float64)
        self._pts3d_gpu = cp.zeros((M, 3), dtype=cp.float64)
        self._pts2d_gpu = cp.zeros((M, 2), dtype=cp.float64)

    def refine(self, T_init, pts3d_w, pts2d):
        if not np.all(np.isfinite(T_init)):
            return T_init
        if len(pts3d_w) < 4:
            return T_init
            
        # Filter non-finite points (V45)
        finite_mask = np.all(np.isfinite(pts3d_w), axis=1) & np.all(np.isfinite(pts2d), axis=1)
        if not np.any(finite_mask):
            return T_init
        if not np.all(finite_mask):
            pts3d_w = pts3d_w[finite_mask]
            pts2d = pts2d[finite_mask]
            if len(pts3d_w) < 4:
                return T_init

        if self.use_gpu and len(pts3d_w) > 10:
            return self._refine_gpu(T_init, pts3d_w, pts2d)
        return self._refine_cpu(T_init, pts3d_w, pts2d)

    def _project_gpu(self, pts3d_gpu, T_gpu):
        Ti = se3_inv_gpu(T_gpu)
        Ri = Ti[:3,:3]; ti = Ti[:3,3]
        pc = cp.dot(pts3d_gpu, Ri.T) + ti
        
        # Numerical stability: clip z to avoid division by zero or negative (behind cam)
        z = cp.maximum(pc[:, 2], 1e-4)
        
        u = self.fx * pc[:, 0] / z + self.cx
        v = self.fy * pc[:, 1] / z + self.cy
        return u, v, pc, z

    def _residuals_and_jacobian_gpu(self, pts3d_gpu, pts2d_gpu, T_gpu):
        N = len(pts3d_gpu)
        u, v, pc, z = self._project_gpu(pts3d_gpu, T_gpu)
        
        # Residuals
        self._r_gpu[:N*2:2] = u - pts2d_gpu[:,0]
        self._r_gpu[1:N*2:2] = v - pts2d_gpu[:,1]
        
        # Safety: clip residuals
        self._r_gpu[:N*2] = cp.clip(self._r_gpu[:N*2], -1e6, 1e6)
        
        # Jacobians
        iz = 1.0 / z
        iz2 = iz*iz
        
        self._J_proj_gpu[:N,0,0] = self.fx * iz
        self._J_proj_gpu[:N,0,2] = -self.fx * pc[:,0] * iz2
        self._J_proj_gpu[:N,1,1] = self.fy * iz
        self._J_proj_gpu[:N,1,2] = -self.fy * pc[:,1] * iz2
        
        self._J_cam_gpu[:N,0,0] = 1
        self._J_cam_gpu[:N,1,1] = 1
        self._J_cam_gpu[:N,2,2] = 1
        self._J_cam_gpu[:N,0,4] = pc[:,2]
        self._J_cam_gpu[:N,0,5] = -pc[:,1]
        self._J_cam_gpu[:N,1,3] = -pc[:,2]
        self._J_cam_gpu[:N,1,5] = pc[:,0]
        self._J_cam_gpu[:N,2,3] = pc[:,1]
        self._J_cam_gpu[:N,2,4] = -pc[:,0]
        
        # Batch multiply J_proj and J_cam
        self._J_full_gpu[:N] = cp.matmul(self._J_proj_gpu[:N], self._J_cam_gpu[:N])
        
        # Flatten into the 2N x 6 matrix
        self._J_gpu[:N*2:2] = self._J_full_gpu[:N,0,:]
        self._J_gpu[1:N*2:2] = self._J_full_gpu[:N,1,:]
        
        # Safety: zero out any non-finite entries
        non_finite = ~cp.isfinite(self._J_gpu[:N*2])
        if cp.any(non_finite):
            self._J_gpu[:N*2][non_finite] = 0.0
            
        return self._r_gpu[:N*2], self._J_gpu[:N*2]

    def _huber_weights_gpu(self, r):
        N = len(r)//2
        err = cp.sqrt(r[::2]**2 + r[1::2]**2)
        w = cp.ones(N, dtype=cp.float64)
        m = err > self.huber_c
        w[m] = self.huber_c / cp.maximum(err[m], 1e-10)
        W = cp.empty(N*2, dtype=cp.float64)
        W[::2] = w
        W[1::2] = w
        return W

    def _refine_gpu(self, T_init, pts3d_w, pts2d):
        N = len(pts3d_w)
        if N > self.max_points:
            N = self.max_points
            pts3d_w = pts3d_w[:N]
            pts2d = pts2d[:N]
        self._pts3d_gpu[:N] = cp.asarray(pts3d_w, dtype=cp.float64)
        self._pts2d_gpu[:N] = cp.asarray(pts2d, dtype=cp.float64)
        pts3d_gpu = self._pts3d_gpu[:N]
        pts2d_gpu = self._pts2d_gpu[:N]
        T_gpu = cp.asarray(T_init, dtype=cp.float64)
        lam = self.lam0
        prev_cost = np.inf
        
        for _ in range(self.max_iter):
            r, J = self._residuals_and_jacobian_gpu(pts3d_gpu, pts2d_gpu, T_gpu)
            W = self._huber_weights_gpu(r)
            
            # Weighted residuals and Jacobian
            W_sqrt = cp.sqrt(W)
            r_weighted = r * W_sqrt
            J_weighted = J * W_sqrt[:, None]
            
            cost = float(0.5 * cp.dot(r_weighted, r_weighted))
            if not np.isfinite(cost):
                break
                
            JtWJ_g = cp.dot(J_weighted.T, J_weighted)
            JtWr_g = cp.dot(J_weighted.T, r)
            
            JtWJd_g = JtWJ_g.copy()
            # Levenberg-Marquardt damping
            diag_idx = cp.arange(6)
            JtWJd_g[diag_idx, diag_idx] += lam * cp.maximum(cp.diag(JtWJ_g), 1e-6)
            
            try:
                dxi_g = cp.linalg.solve(JtWJd_g, -JtWr_g)
            except Exception:
                # If singular, try simple gradient descent step
                dxi_g = -JtWr_g / (lam + 1e-6)
            
            if not cp.all(cp.isfinite(dxi_g)):
                break
                
            delta_T_g = batch_exp_se3_gpu(dxi_g[None,:])[0]
            T_cw_new_g = delta_T_g @ se3_inv_gpu(T_gpu)
            T_new_g = se3_inv_gpu(T_cw_new_g)
            
            # Evaluate new cost
            rn, _ = self._residuals_and_jacobian_gpu(pts3d_gpu, pts2d_gpu, T_new_g)
            Wn = self._huber_weights_gpu(rn)
            cost_new = float(0.5 * cp.dot(rn, Wn * rn))
            
            if np.isfinite(cost_new) and cost_new < cost:
                T_gpu = T_new_g
                lam = max(lam * 0.1, 1e-10)
                if abs(prev_cost - cost_new) < self.conv_delta:
                    break
                prev_cost = cost_new
            else:
                lam = min(lam * 10.0, 1e6)
                
        # Final sanity check
        res_T = cp.asnumpy(T_gpu)
        if not np.all(np.isfinite(res_T)):
            return T_init
        return res_T

    def _refine_cpu(self, T_init, pts3d_w, pts2d):
        T = T_init.copy()
        lam = self.lam0
        prev_cost = np.inf
        N = len(pts3d_w)
        
        for _ in range(self.max_iter):
            Ti = PoseTransform.inverse(T)
            pc = PoseTransform.transform_points(Ti, pts3d_w)
            
            # Stability: clip z
            z = np.maximum(pc[:, 2], 1e-4)
            iz = 1.0 / z
            iz2 = iz * iz
            
            up = self.fx * pc[:, 0] * iz + self.cx
            vp = self.fy * pc[:, 1] * iz + self.cy
            
            r = np.empty(2 * N)
            r[0::2] = up - pts2d[:, 0]
            r[1::2] = vp - pts2d[:, 1]
            
            # Safety: clip residuals to prevent overflow during sum(r**2)
            r = np.clip(r, -1e6, 1e6)
            
            # Huber weighting
            err = np.sqrt(r[0::2]**2 + r[1::2]**2)
            w = np.ones(N)
            m = err > self.huber_c
            w[m] = self.huber_c / np.maximum(err[m], 1e-10)
            W = np.empty(2 * N)
            W[0::2] = w; W[1::2] = w
            
            cost = 0.5 * np.sum(W * r**2)
            if not np.isfinite(cost):
                break
                
            Jp = np.zeros((N, 2, 3))
            Jc = np.zeros((N, 3, 6))
            
            Jp[:, 0, 0] = self.fx * iz
            Jp[:, 0, 2] = -self.fx * pc[:, 0] * iz2
            Jp[:, 1, 1] = self.fy * iz
            Jp[:, 1, 2] = -self.fy * pc[:, 1] * iz2
            
            Jc[:, 0, 0] = 1; Jc[:, 1, 1] = 1; Jc[:, 2, 2] = 1
            Jc[:, 0, 4] = pc[:, 2]; Jc[:, 0, 5] = -pc[:, 1]
            Jc[:, 1, 3] = -pc[:, 2]; Jc[:, 1, 5] = pc[:, 0]
            Jc[:, 2, 3] = pc[:, 1]; Jc[:, 2, 4] = -pc[:, 0]
            
            Jf = np.einsum('nij,njk->nik', Jp, Jc)
            J = np.empty((2 * N, 6))
            J[0::2] = Jf[:, 0, :]
            J[1::2] = Jf[:, 1, :]
            
            WJ = J * W[:, None]
            JtWJ = WJ.T @ J
            JtWr = WJ.T @ r
            
            JtWJd = JtWJ.copy()
            np.fill_diagonal(JtWJd, JtWJd.diagonal() + lam * np.maximum(np.diag(JtWJ), 1e-6))
            
            try:
                dxi = np.linalg.solve(JtWJd, -JtWr)
            except np.linalg.LinAlgError:
                break
                
            if not np.all(np.isfinite(dxi)):
                break
                
            T_cw_new = PoseTransform.exp_se3(dxi) @ PoseTransform.inverse(T)
            T_new = PoseTransform.inverse(T_cw_new)
            
            # Cost check for T_new
            Tin = PoseTransform.inverse(T_new)
            pcn = PoseTransform.transform_points(Tin, pts3d_w)
            zn = np.maximum(pcn[:, 2], 1e-4)
            upn = self.fx * pcn[:, 0] / zn + self.cx
            vpn = self.fy * pcn[:, 1] / zn + self.cy
            rn = np.empty(2 * N)
            rn[0::2] = upn - pts2d[:, 0]
            rn[1::2] = vpn - pts2d[:, 1]
            
            # Evaluate new cost with same weights for comparison
            cost_new = 0.5 * np.sum(W * rn**2)
            
            if np.isfinite(cost_new) and cost_new < cost:
                T = T_new
                lam = max(lam * 0.1, 1e-10)
                if abs(prev_cost - cost_new) < self.conv_delta:
                    break
                prev_cost = cost_new
            else:
                lam = min(lam * 10.0, 1e6)
                
        if not np.all(np.isfinite(T)):
            return T_init
        return T