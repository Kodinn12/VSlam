"""SE(3) particle filter with IMU and gravity constraints."""

import numpy as np
import math
from ..utils.logger import get_logger
from ..utils.se3_ops import (
    batch_exp_se3_gpu, batch_log_se3_gpu, batch_matmul_gpu,
    se3_inv_gpu, PoseTransform, USE_CUPY
)

logger = get_logger(__name__)

try:
    import cupy as cp
    if not USE_CUPY:
        raise ImportError
except ImportError:
    cp = np
    USE_CUPY = False


class SE3ParticleFilter:
    def __init__(self, K, num_particles=80, sigma_t_base=0.012, sigma_r_base=0.015,
                 sigma_t_min=0.003, sigma_r_min=0.004, obs_sigma=3.5,
                 resample_thresh=0.5, karcher_iters=6, confidence_scale=40.0,
                 huber_thresh=2.0):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.N = num_particles
        self.sigma_t_base = sigma_t_base
        self.sigma_r_base = sigma_r_base
        self.sigma_t_min = sigma_t_min
        self.sigma_r_min = sigma_r_min
        self.confidence_scale = confidence_scale
        self.obs_sigma = obs_sigma
        self.obs_sigma_base = obs_sigma
        self.huber_c = huber_thresh
        self.resample_thresh = resample_thresh
        self.karcher_iters = karcher_iters
        self.particles = np.tile(np.eye(4), (self.N, 1, 1))
        self.weights = np.full(self.N, 1.0 / self.N)
        self.velocity = np.zeros(6)
        self._prev_mean = np.eye(4)
        self._prev_prev_mean = np.eye(4)
        self._frame_count = 0
        if USE_CUPY:
            self._parts_gpu = cp.asarray(self.particles, dtype=cp.float64)
            self._weights_gpu = cp.asarray(self.weights)
        logger.info(f"PF: {self.N} particles, GPU={'CuPy' if USE_CUPY else 'CPU'}")

    def reset(self, T, spread_t=0.005, spread_r=0.008):
        self.particles = self._scatter_around(T, self.N, spread_t, spread_r)
        self.weights = np.full(self.N, 1.0 / self.N)
        self.velocity = np.zeros(6)
        self._prev_mean = T.copy()
        self._prev_prev_mean = T.copy()
        self._frame_count = 0
        if USE_CUPY:
            self._parts_gpu[:] = cp.asarray(self.particles, dtype=cp.float64)
            self._weights_gpu[:] = cp.asarray(self.weights)

    def predict(self, num_inliers=0):
        conf = min(num_inliers / self.confidence_scale, 1.0)
        st = self.sigma_t_base - conf * (self.sigma_t_base - self.sigma_t_min)
        sr = self.sigma_r_base - conf * (self.sigma_r_base - self.sigma_r_min)
        if USE_CUPY:
            noise_gpu = cp.empty((self.N, 6), dtype=cp.float64)
            noise_gpu[:, :3] = cp.random.normal(0, st, (self.N, 3))
            noise_gpu[:, 3:] = cp.random.normal(0, sr, (self.N, 3))
            vel_gpu = cp.asarray(self.velocity[None, :], dtype=cp.float64)
            xi_pred_gpu = vel_gpu + noise_gpu
            delta_T_gpu = batch_exp_se3_gpu(xi_pred_gpu)
            self._parts_gpu = batch_matmul_gpu(self._parts_gpu, delta_T_gpu)
        else:
            noise = np.empty((self.N, 6))
            noise[:, :3] = np.random.normal(0, st, (self.N, 3))
            noise[:, 3:] = np.random.normal(0, sr, (self.N, 3))
            xi_pred = self.velocity[None, :] + noise
            delta_T = batch_exp_se3_gpu(xi_pred)
            self.particles = batch_matmul_gpu(self.particles, delta_T)

    def predict_with_imu(self, imu_delta_R, imu_delta_p, num_inliers=0, imu_weight=0.7):
        conf = min(num_inliers / self.confidence_scale, 1.0)
        st = self.sigma_t_base - conf * (self.sigma_t_base - self.sigma_t_min)
        sr = self.sigma_r_base - conf * (self.sigma_r_base - self.sigma_r_min)

        # Build mean SE3 delta from IMU
        xi_imu = np.zeros(6, dtype=np.float64)
        cos_a = np.clip((np.trace(imu_delta_R) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(cos_a)
        if angle > 1e-9:
            RmRt = imu_delta_R - imu_delta_R.T
            s_inv = angle / (2.0 * math.sin(angle))
            omega = np.empty(3, dtype=np.float64)
            omega[0] = s_inv * RmRt[2, 1]
            omega[1] = s_inv * RmRt[0, 2]
            omega[2] = s_inv * RmRt[1, 0]
            xi_imu[3:] = omega
            omega_n = omega / angle
            Kn = np.array([[0, -omega_n[2], omega_n[1]],
                           [omega_n[2], 0, -omega_n[0]],
                           [-omega_n[1], omega_n[0], 0]], dtype=np.float64)
            Kn2 = Kn @ Kn
            skw = np.array([[0, -omega[2], omega[1]],
                            [omega[2], 0, -omega[0]],
                            [-omega[1], omega[0], 0]], dtype=np.float64)
            half = angle * math.sin(angle) / (2.0 * (1.0 - math.cos(angle)))
            V_inv = np.eye(3) - 0.5 * skw + (1.0 / (angle * angle)) * (1.0 - half) * Kn2
            xi_imu[:3] = V_inv @ imu_delta_p
        else:
            xi_imu[:3] = imu_delta_p

        noise_scale_t = max(1.0 - imu_weight * 0.70,
                            self.sigma_t_min / (self.sigma_t_base + 1e-12) * 1.2)
        noise_scale_r = max(1.0 - imu_weight * 0.70,
                            self.sigma_r_min / (self.sigma_r_base + 1e-12) * 1.2)
        sig_t = max(st * noise_scale_t, self.sigma_t_min)
        sig_r = max(sr * noise_scale_r, self.sigma_r_min)

        if USE_CUPY:
            noise_gpu = cp.empty((self.N, 6), dtype=cp.float64)
            noise_gpu[:, :3] = cp.random.normal(0, sig_t, (self.N, 3))
            noise_gpu[:, 3:] = cp.random.normal(0, sig_r, (self.N, 3))
            xi_imu_gpu = cp.asarray(xi_imu[None, :], dtype=cp.float64)
            xi_pred_gpu = imu_weight * xi_imu_gpu + noise_gpu
            delta_T_gpu = batch_exp_se3_gpu(xi_pred_gpu)
            self._parts_gpu = batch_matmul_gpu(self._parts_gpu, delta_T_gpu)
        else:
            noise = np.empty((self.N, 6))
            noise[:, :3] = np.random.normal(0, sig_t, (self.N, 3))
            noise[:, 3:] = np.random.normal(0, sig_r, (self.N, 3))
            xi_pred = imu_weight * xi_imu[None, :] + noise
            delta_T = batch_exp_se3_gpu(xi_pred)
            self.particles = batch_matmul_gpu(self.particles, delta_T)

        self.velocity = imu_weight * xi_imu + (1.0 - imu_weight) * self.velocity

    def update_with_gravity_constraint(self, gravity_cam, gravity_sigma=0.12, gravity_world=None):
        if gravity_cam is None:
            return
        gn = np.linalg.norm(gravity_cam)
        if gn < 1.0:
            return
        g_ref = gravity_cam / gn
        use_abs = (gravity_world is not None and np.linalg.norm(gravity_world) > 1.0)

        if USE_CUPY:
            g_ref_gpu = cp.asarray(g_ref, dtype=cp.float64)
            R_wcs_gpu = self._parts_gpu[:, :3, :3]
            g_w_parts = cp.matmul(R_wcs_gpu, g_ref_gpu[:, None])[:, :, 0]
            norms = cp.linalg.norm(g_w_parts, axis=1, keepdims=True) + 1e-9
            g_w_norm = g_w_parts / norms
            if use_abs:
                gwr_gpu = cp.asarray(gravity_world / np.linalg.norm(gravity_world), dtype=cp.float64)
            else:
                gwr_gpu = g_w_parts.mean(axis=0)
                gwr_n2 = float(cp.dot(gwr_gpu, gwr_gpu))
                if gwr_n2 < 0.01:
                    return
                gwr_gpu = gwr_gpu / cp.sqrt(cp.asarray(gwr_n2))
            dots = cp.clip(g_w_norm @ gwr_gpu, -1.0, 1.0)
            angles = cp.arccos(dots)
            grav_w = cp.exp(-0.5 * (angles / gravity_sigma) ** 2)
            grav_w = cp.maximum(grav_w, 1e-6)
            self._weights_gpu *= grav_w
            ws = float(self._weights_gpu.sum())
            if ws > 1e-12:
                self._weights_gpu /= ws
            else:
                self._weights_gpu[:] = 1.0 / self.N
            n_eff = float(1.0 / cp.sum(self._weights_gpu * self._weights_gpu))
        else:
            R_wcs = self.particles[:, :3, :3]
            g_world_particles = np.einsum('nij,j->ni', R_wcs, g_ref)
            if use_abs:
                g_world_ref = gravity_world / np.linalg.norm(gravity_world)
            else:
                g_world_ref = g_world_particles.mean(axis=0)
                gwr_n = np.linalg.norm(g_world_ref)
                if gwr_n < 0.1:
                    return
                g_world_ref /= gwr_n
            dots = np.clip(np.einsum('ni,i->n', g_world_particles /
                                     (np.linalg.norm(g_world_particles, axis=1, keepdims=True) + 1e-9),
                                     g_world_ref), -1.0, 1.0)
            angles = np.arccos(dots)
            grav_weights = np.exp(-0.5 * (angles / gravity_sigma) ** 2)
            grav_weights = np.maximum(grav_weights, 1e-6)
            self.weights *= grav_weights
            ws = self.weights.sum()
            if ws > 1e-12:
                self.weights /= ws
            else:
                self.weights[:] = 1.0 / self.N
            n_eff = 1.0 / np.sum(self.weights * self.weights)

        if n_eff < self.resample_thresh * self.N:
            self._systematic_resample()

    def apply_zupt(self):
        T_mean = self._karcher_mean()
        spread_t = self.sigma_t_min * 0.5
        spread_r = self.sigma_r_min * 0.5
        self.particles = self._scatter_around(T_mean, self.N, spread_t, spread_r)
        self.velocity = np.zeros(6)
        if USE_CUPY:
            self._parts_gpu[:] = cp.asarray(self.particles, dtype=cp.float64)
            self._weights_gpu[:] = cp.full(self.N, 1.0 / self.N, dtype=cp.float64)
            self.weights = cp.asnumpy(self._weights_gpu)

    def update(self, pts3d_w, pts2d):
        if pts3d_w is None or len(pts3d_w) < 4:
            return
        if USE_CUPY and len(pts3d_w) >= 4:
            M = min(len(pts3d_w), 2500)
            pts3d_g = cp.asarray(pts3d_w[:M], dtype=cp.float64)
            pts2d_g = cp.asarray(pts2d[:M], dtype=cp.float64)
            parts_gpu = self._parts_gpu
            T_inv_gpu = se3_inv_gpu(parts_gpu)
            R_inv = T_inv_gpu[:, :3, :3]
            t_inv = T_inv_gpu[:, :3, 3]
            pts_cam = cp.matmul(R_inv, pts3d_g.T[None]).swapaxes(1, 2) + t_inv[:, None, :]
            Z = pts_cam[:, :, 2]
            inv_Z = cp.where(Z > 0.01, 1.0 / Z, 0.0)
            u_proj = self.fx * pts_cam[:, :, 0] * inv_Z + self.cx
            v_proj = self.fy * pts_cam[:, :, 1] * inv_Z + self.cy
            du = u_proj - pts2d_g[None, :, 0]
            dv = v_proj - pts2d_g[None, :, 1]
            err_sq = du * du + dv * dv
            err = cp.sqrt(err_sq)
            hw = cp.where(err > self.huber_c, self.huber_c / cp.maximum(err, 1e-10), 1.0)
            cost = cp.sum(hw * err_sq, axis=1)
            behind = cp.any(Z < 0.01, axis=1)
            cost[behind] = 1e12
            log_w_g = -cost / (2.0 * self.obs_sigma * self.obs_sigma)
            log_w_g -= log_w_g.max()
            w_g = cp.exp(log_w_g)
            ws_g = w_g.sum()
            if float(ws_g) > 0:
                w_g /= ws_g
            else:
                w_g[:] = 1.0 / self.N
            self._weights_gpu[:] = w_g
            n_eff = float(1.0 / cp.sum(w_g * w_g))
        else:
            T_inv = se3_inv_gpu(self.particles)
            R_inv = T_inv[:, :3, :3]
            t_inv = T_inv[:, :3, 3]
            pts_cam = np.einsum('nij,mj->nmi', R_inv, pts3d_w) + t_inv[:, None, :]
            Z = pts_cam[:, :, 2]
            inv_Z = np.where(Z > 0.01, 1.0 / Z, 0.0)
            u_proj = self.fx * pts_cam[:, :, 0] * inv_Z + self.cx
            v_proj = self.fy * pts_cam[:, :, 1] * inv_Z + self.cy
            du = u_proj - pts2d[None, :, 0]
            dv = v_proj - pts2d[None, :, 1]
            err_sq = du * du + dv * dv
            err = np.sqrt(err_sq)
            hw = np.where(err > self.huber_c, self.huber_c / np.maximum(err, 1e-10), 1.0)
            cost = np.sum(hw * err_sq, axis=1)
            behind = np.any(Z < 0.01, axis=1)
            cost[behind] = np.inf
            log_w = -cost / (2.0 * self.obs_sigma * self.obs_sigma)
            log_w -= log_w.max()
            self.weights = np.exp(log_w)
            ws = self.weights.sum()
            if ws > 0:
                self.weights /= ws
            else:
                self.weights[:] = 1.0 / self.N
            n_eff = 1.0 / np.sum(self.weights * self.weights)

        if n_eff < self.resample_thresh * self.N:
            self._systematic_resample()

    def estimate(self):
        T_mean = self._karcher_mean()
        self._frame_count += 1
        if self._frame_count >= 2:
            T_rel = PoseTransform.inverse(self._prev_mean) @ T_mean
            v_visual = PoseTransform.log_se3(T_rel)
            self.velocity = 0.4 * v_visual + 0.6 * self.velocity
        self._prev_prev_mean = self._prev_mean.copy()
        self._prev_mean = T_mean.copy()
        return T_mean

    def _karcher_mean(self):
        if USE_CUPY and self.N >= 60:
            best = int(cp.argmax(self._weights_gpu))
            T_mean_g = self._parts_gpu[best].copy()
            w_gpu = self._weights_gpu
            for _ in range(self.karcher_iters):
                Ti_g = se3_inv_gpu(T_mean_g)
                T_rel_g = cp.matmul(Ti_g, self._parts_gpu)
                xi_all_g = batch_log_se3_gpu(T_rel_g)
                xi_sum_g = (w_gpu[:, None] * xi_all_g).sum(axis=0)
                xi_sum_norm = float(cp.linalg.norm(xi_sum_g))
                delta_g = batch_exp_se3_gpu(xi_sum_g[None, :])[0]
                T_mean_g = T_mean_g @ delta_g
                if xi_sum_norm < 1e-9:
                    break
            return cp.asnumpy(T_mean_g)
        else:
            best = np.argmax(self.weights)
            T_mean = self.particles[best].copy()
            for _ in range(self.karcher_iters):
                Ti = PoseTransform.inverse(T_mean)
                Tit = np.tile(Ti, (self.N, 1, 1))
                T_rel = batch_matmul_gpu(Tit, self.particles)
                xi_all = batch_log_se3_gpu(T_rel)
                xi_sum = (self.weights[:, None] * xi_all).sum(axis=0)
                T_mean = T_mean @ PoseTransform.exp_se3(xi_sum)
                if np.linalg.norm(xi_sum) < 1e-9:
                    break
            return T_mean

    def _systematic_resample(self):
        if USE_CUPY:
            cumsum_gpu = cp.cumsum(self._weights_gpu)
            cumsum_gpu[-1] = 1.0
            u0 = float(cp.random.uniform(0, 1.0 / self.N))
            u_gpu = u0 + cp.arange(self.N, dtype=cp.float64) / self.N
            idx_gpu = cp.searchsorted(cumsum_gpu, u_gpu)
            idx_gpu = cp.clip(idx_gpu, 0, self.N - 1)
            self._parts_gpu = self._parts_gpu[idx_gpu]
            self._weights_gpu[:] = 1.0 / self.N
            self.particles = cp.asnumpy(self._parts_gpu)
            self.weights = cp.asnumpy(self._weights_gpu)
        else:
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0
            u0 = np.random.uniform(0, 1.0 / self.N)
            u = u0 + np.arange(self.N) / self.N
            idx = np.searchsorted(cumsum, u)
            idx = np.clip(idx, 0, self.N - 1)
            self.particles = self.particles[idx].copy()
            self.weights = np.full(self.N, 1.0 / self.N)

    @staticmethod
    def _scatter_around(T_center, N, spread_t, spread_r):
        if USE_CUPY:
            noise_gpu = cp.empty((N, 6), dtype=cp.float64)
            noise_gpu[:, :3] = cp.random.normal(0, spread_t, (N, 3))
            noise_gpu[:, 3:] = cp.random.normal(0, spread_r, (N, 3))
            delta_gpu = batch_exp_se3_gpu(noise_gpu)
            center_gpu = cp.tile(cp.asarray(T_center, dtype=cp.float64), (N, 1, 1))
            return cp.asnumpy(batch_matmul_gpu(center_gpu, delta_gpu))
        noise = np.zeros((N, 6))
        noise[:, :3] = np.random.normal(0, spread_t, (N, 3))
        noise[:, 3:] = np.random.normal(0, spread_r, (N, 3))
        delta = batch_exp_se3_gpu(noise)
        center = np.tile(T_center, (N, 1, 1))
        return batch_matmul_gpu(center, delta)