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
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
        self.max_iter = max_iter
        self.lam0 = lam0
        self.conv_delta = conv_delta
        self.huber_c = huber_thresh
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
        if self.use_gpu and len(pts3d_w) > 10:
            return self._refine_gpu(T_init, pts3d_w, pts2d)
        return self._refine_cpu(T_init, pts3d_w, pts2d)

    def _project_gpu(self, pts3d_gpu, T_gpu):
        Ti = se3_inv_gpu(T_gpu)
        Ri = Ti[:3,:3]; ti = Ti[:3,3]
        pc = cp.dot(pts3d_gpu, Ri.T) + ti
        z = pc[:,2]
        u = self.fx * pc[:,0] / z + self.cx
        v = self.fy * pc[:,1] / z + self.cy
        return u, v, pc, z

    def _residuals_and_jacobian_gpu(self, pts3d_gpu, pts2d_gpu, T_gpu):
        N = len(pts3d_gpu)
        u, v, pc, z = self._project_gpu(pts3d_gpu, T_gpu)
        self._r_gpu[:N*2:2] = u - pts2d_gpu[:,0]
        self._r_gpu[1:N*2:2] = v - pts2d_gpu[:,1]
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
        self._J_full_gpu[:N] = cp.matmul(self._J_proj_gpu[:N], self._J_cam_gpu[:N])
        self._J_gpu[:N*2:2] = self._J_full_gpu[:N,0,:]
        self._J_gpu[1:N*2:2] = self._J_full_gpu[:N,1,:]
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
            WJ = J * W[:, None]
            JtWJ_g = cp.dot(WJ.T, J)
            JtWr_g = cp.dot(WJ.T, r)
            cost = float(0.5 * cp.dot(r, W * r))
            JtWJd_g = JtWJ_g.copy()
            JtWJd_g[cp.arange(6), cp.arange(6)] += lam
            try:
                dxi_g = cp.linalg.solve(JtWJd_g, -JtWr_g)
            except Exception:
                break
            delta_T_g = batch_exp_se3_gpu(dxi_g[None,:])[0]
            T_cw_new_g = delta_T_g @ se3_inv_gpu(T_gpu)
            T_new_g = se3_inv_gpu(T_cw_new_g)
            rn, _ = self._residuals_and_jacobian_gpu(pts3d_gpu, pts2d_gpu, T_new_g)
            Wn = self._huber_weights_gpu(rn)
            cost_new = float(0.5 * cp.dot(rn, Wn * rn))
            if cost_new < cost:
                T_gpu = T_new_g
                lam = max(lam * 0.1, 1e-10)
                if abs(prev_cost - cost_new) < self.conv_delta:
                    break
                prev_cost = cost_new
            else:
                lam = min(lam * 10.0, 1e6)
        return cp.asnumpy(T_gpu)

    def _refine_cpu(self, T_init, pts3d_w, pts2d):
        T = T_init.copy()
        lam = self.lam0
        prev_cost = np.inf
        for _ in range(self.max_iter):
            Ti = PoseTransform.inverse(T)
            pc = PoseTransform.transform_points(Ti, pts3d_w)
            z = pc[:, 2]
            up = self.fx * pc[:, 0] / z + self.cx
            vp = self.fy * pc[:, 1] / z + self.cy
            N = len(pts3d_w)
            r = np.empty(2 * N)
            r[0::2] = up - pts2d[:, 0]
            r[1::2] = vp - pts2d[:, 1]
            iz = 1.0 / z
            iz2 = iz * iz
            Jp = np.zeros((N, 2, 3))
            Jc = np.zeros((N, 3, 6))
            Jp[:, 0, 0] = self.fx * iz
            Jp[:, 0, 2] = -self.fx * pc[:, 0] * iz2
            Jp[:, 1, 1] = self.fy * iz
            Jp[:, 1, 2] = -self.fy * pc[:, 1] * iz2
            Jc[:, 0, 0] = 1
            Jc[:, 1, 1] = 1
            Jc[:, 2, 2] = 1
            Jc[:, 0, 4] = pc[:, 2]
            Jc[:, 0, 5] = -pc[:, 1]
            Jc[:, 1, 3] = -pc[:, 2]
            Jc[:, 1, 5] = pc[:, 0]
            Jc[:, 2, 3] = pc[:, 1]
            Jc[:, 2, 4] = -pc[:, 0]
            Jf = np.einsum('nij,njk->nik', Jp, Jc)
            J = np.empty((2 * N, 6))
            J[0::2] = Jf[:, 0, :]
            J[1::2] = Jf[:, 1, :]
            err = np.sqrt(r[0::2] ** 2 + r[1::2] ** 2)
            w = np.ones(N)
            m = err > self.huber_c
            w[m] = self.huber_c / np.maximum(err[m], 1e-10)
            W = np.empty(2 * N)
            W[0::2] = w
            W[1::2] = w
            WJ = J * W[:, None]
            JtWJ = WJ.T @ J
            JtWr = WJ.T @ r
            cost = 0.5 * (r * W * r).sum()
            JtWJd = JtWJ.copy()
            np.fill_diagonal(JtWJd, JtWJd.diagonal() + lam)
            try:
                dxi = np.linalg.solve(JtWJd, -JtWr)
            except np.linalg.LinAlgError:
                break
            T_cw_new = PoseTransform.exp_se3(dxi) @ PoseTransform.inverse(T)
            T_new = PoseTransform.inverse(T_cw_new)
            Tin = PoseTransform.inverse(T_new)
            pcn = PoseTransform.transform_points(Tin, pts3d_w)
            zn = pcn[:, 2]
            upn = self.fx * pcn[:, 0] / zn + self.cx
            vpn = self.fy * pcn[:, 1] / zn + self.cy
            rn = np.empty(2 * N)
            rn[0::2] = upn - pts2d[:, 0]
            rn[1::2] = vpn - pts2d[:, 1]
            errn = np.sqrt(rn[0::2] ** 2 + rn[1::2] ** 2)
            wn = np.ones(N)
            wn[errn > self.huber_c] = self.huber_c / np.maximum(errn[errn > self.huber_c], 1e-10)
            Wn = np.empty(2 * N)
            Wn[0::2] = wn
            Wn[1::2] = wn
            cost_new = 0.5 * (rn * Wn * rn).sum()
            if cost_new < cost:
                T = T_new
                lam = max(lam * 0.1, 1e-10)
                if abs(prev_cost - cost_new) < self.conv_delta:
                    break
                prev_cost = cost_new
            else:
                lam = min(lam * 10.0, 1e6)
        return T