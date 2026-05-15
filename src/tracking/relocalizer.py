"""Ghost particle relocalizer using depth, photometric and feature alignment."""

import numpy as np
import math
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.logger import get_logger
from ..utils.se3_ops import batch_exp_se3_gpu, batch_matmul_gpu, PoseTransform, USE_CUPY
from ..utils.depth_utils import bilinear_depth_gpu, bilinear_depth
from ..utils.cupy_utils import to_numpy_safe

logger = get_logger(__name__)

try:
    import cupy as cp
    if not USE_CUPY:
        raise ImportError
except ImportError:
    cp = np
    USE_CUPY = False

try:
    from cupyx.scipy.spatial import KDTree as CupyKDTree
    HAS_CUPY_KDTREE = True
except ImportError:
    HAS_CUPY_KDTREE = False
    CupyKDTree = None

from scipy.spatial import cKDTree

class GhostParticleRelocalizer:
    def __init__(self, config, K, voxel_manager, bubble_map):
        self.config = config
        self.K = K
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
        self.voxel_manager = voxel_manager
        self.bubble_map = bubble_map
        self.particles = []
        self.best_particle = None
        self.best_score = -np.inf
        self.iteration = 0
        self.max_iterations = config.get("relocalizer_max_iters", 60)
        self.base_spread_t = config.get("relocalizer_spread_t", 0.5)
        self.base_spread_r = config.get("relocalizer_spread_r", 0.8)
        self.convergence_threshold = config.get("relocalizer_converge_thresh", 0.4)
        self.min_score_threshold = config.get("relocalizer_min_score", 0.4)
        self.min_overlap_pixels = config.get("relocalizer_min_overlap_pixels", 100)
        self._score_executor = ThreadPoolExecutor(max_workers=4)
        self._reloc_cached_kf_id = None
        self._reloc_cached_kf_tree = None
        self._reloc_cached_kf_pts_world = None

    # Methods: scatter_hypotheses, _render_bubbles_to_depth, _score_depth_alignment,
    # _score_photometric_consistency, _score_feature_alignment, score_hypothesis,
    # evolve_ghosts, get_best_pose
    # Implementation identical to original, with logger instead of print.

    def scatter_hypotheses(self, seed_pose, num_particles=256, mode="local"):
        self.particles = []
        self.iteration = 0
        self.best_score = -np.inf
        if mode in ["local", "hybrid"]:
            nl = num_particles if mode == "local" else num_particles // 2
            if USE_CUPY:
                # GPU zone: use only CuPy arrays
                xi_g = cp.concatenate([
                    cp.random.normal(0, self.base_spread_t, (nl, 3), dtype=cp.float32),
                    cp.random.normal(0, self.base_spread_r, (nl, 3), dtype=cp.float32)], axis=1).astype(cp.float32)
                seed_g = cp.tile(cp.asarray(seed_pose, dtype=cp.float32), (nl, 1, 1))
                deltas_g = batch_exp_se3_gpu(xi_g)
                T_arr_gpu = batch_matmul_gpu(seed_g, deltas_g)
                T_arr_cpu = cp.asnumpy(T_arr_gpu)  # GPU->CPU conversion at boundary
                for T_new in T_arr_cpu:
                    self.particles.append((T_new, 0.0, np.zeros(6)))
            else:
                # CPU zone: use only NumPy arrays
                nt = np.random.normal(0, self.base_spread_t, (nl, 3))
                nr = np.random.normal(0, self.base_spread_r, (nl, 3))
                xi_batch = np.concatenate([nt, nr], axis=1)
                deltas = batch_exp_se3_gpu(xi_batch)
                seed_tiled = np.tile(seed_pose, (nl, 1, 1))
                T_arr = np.einsum('nij,njk->nik', seed_tiled, deltas)
                for T_new in T_arr:
                    self.particles.append((T_new, 0.0, np.zeros(6)))
        if mode in ["global", "hybrid"] and len(self.voxel_manager.keyframes) > 0:
            ng = num_particles // 2 if mode == "hybrid" else num_particles
            kfs = self.voxel_manager.keyframes
            idx = np.random.choice(len(kfs), min(ng, len(kfs) * 2), replace=True)
            for i in idx[:ng]:
                xi = np.random.normal(0, [0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
                self.particles.append((kfs[i].pose.copy() @ PoseTransform.exp_se3(xi), 0.0, np.zeros(6)))

    def _render_bubbles_to_depth(self, T_wc, h, w, return_gpu=False):
        T_cw = PoseTransform.inverse(T_wc)
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        if len(self.bubble_map) == 0:
            return (cp.zeros((h, w), dtype=cp.float32) if (return_gpu and USE_CUPY)
                    else np.zeros((h, w), dtype=np.float32))
        if USE_CUPY:
            # GPU zone: use only CuPy arrays
            _mu_already_gpu = isinstance(self.bubble_map.mu, cp.ndarray)
            mu_g = self.bubble_map.mu.astype(cp.float32) if _mu_already_gpu else cp.asarray(self.bubble_map.mu, dtype=cp.float32)
            R_g = cp.asarray(R_cw, dtype=cp.float32)
            t_g = cp.asarray(t_cw, dtype=cp.float32)
            pts_c = (R_g @ mu_g.T).T + t_g
            x_g, y_g, z_g = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
            u_g = self.fx * x_g / (z_g + 1e-9) + self.cx
            v_g = self.fy * y_g / (z_g + 1e-9) + self.cy
            valid_g = (z_g > 0.1) & (u_g >= 0) & (u_g < w) & (v_g >= 0) & (v_g < h)
            if not bool(cp.any(valid_g)):
                return (cp.zeros((h, w), dtype=cp.float32) if return_gpu
                        else np.zeros((h, w), dtype=np.float32))
            ui_g = cp.clip(u_g[valid_g].astype(cp.int32), 0, w - 1)
            vi_g = cp.clip(v_g[valid_g].astype(cp.int32), 0, h - 1)
            zv_g = z_g[valid_g]
            order_g = cp.argsort(-zv_g)
            dm_g = cp.zeros((h, w), dtype=cp.float32)
            dm_g[vi_g[order_g], ui_g[order_g]] = zv_g[order_g]
            return dm_g if return_gpu else cp.asnumpy(dm_g)  # GPU->CPU conversion at boundary
        # CPU zone: use only NumPy arrays
        pts_c = (R_cw @ self.bubble_map.mu.T).T + t_cw
        x, y, z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        valid = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(valid):
            return np.zeros((h, w), dtype=np.float32)
        ui = np.clip(u[valid].astype(int), 0, w - 1)
        vi = np.clip(v[valid].astype(int), 0, h - 1)
        zv = z[valid]
        dm = np.zeros((h, w), dtype=np.float32)
        order = np.argsort(-zv)
        dm[vi[order], ui[order]] = zv[order]
        return dm

    def _score_depth_alignment(self, T_wc, depth, use_voxel=True):
        h, w = depth.shape
        if len(self.bubble_map) < 100:
            return 0.0
        if USE_CUPY:
            # GPU zone: use only CuPy arrays
            rd_g = self._render_bubbles_to_depth(T_wc, h, w, return_gpu=True)
            depth_g = cp.asarray(depth, dtype=cp.float32) if isinstance(depth, np.ndarray) else depth
            vm = rd_g > 0.01
            obs_valid = (depth_g > 0.1) & (depth_g < 10.0)
            both = vm & obs_valid
            n_both = int(cp.sum(both))
            if n_both < self.min_overlap_pixels:
                return 0.0
            ir = 1.0 / rd_g[both]
            io = 1.0 / depth_g[both]
            diff = cp.abs(ir - io)
            hc = cp.float32(0.1)
            wh = cp.where(diff > hc, hc / diff, cp.float32(1.0))
            err = float(cp.mean(wh * diff))
            cov = n_both / max(1, int(cp.sum(obs_valid)))
            return float(cov * math.exp(-err * 5.0))
        else:
            # CPU zone: use only NumPy arrays
            rd = self._render_bubbles_to_depth(T_wc, h, w)
            vm = rd > 0.01
            obs_valid = (depth > 0.1) & (depth < 10.0)
            both = vm & obs_valid
            if np.sum(both) < self.min_overlap_pixels:
                return 0.0
            ir = 1.0 / rd[both]
            io = 1.0 / depth[both]
            diff = np.abs(ir - io)
            hc = 0.1
            wh = np.where(diff > hc, hc / diff, 1.0)
            err = np.mean(wh * diff)
            cov = np.sum(both) / np.sum(obs_valid)
            return float(cov * math.exp(-err * 5.0))

    def _score_photometric_consistency(self, T_wc, image, observed_depth):
        if len(self.bubble_map) < 100:
            return 0.0
        h, w = image.shape[:2]
        n_samples = min(500, len(self.bubble_map))
        if n_samples == 0:
            return 0.0
        if USE_CUPY and isinstance(self.bubble_map.weight, cp.ndarray):
            w_g = self.bubble_map.weight
            w_sum = float(cp.sum(w_g))
            if w_sum > 0:
                probs_g = w_g / w_sum
                idx_g = cp.random.choice(len(self.bubble_map.mu), n_samples, replace=False, p=probs_g)
                indices = cp.asnumpy(idx_g)
            else:
                indices = np.random.choice(len(self.bubble_map.mu), n_samples, replace=False)
        else:
            w_sum = float(np.sum(self.bubble_map.weight))
            probs = (self.bubble_map.weight / w_sum).astype(np.float32) if w_sum > 0 else None
            indices = np.random.choice(len(self.bubble_map.mu), n_samples, p=probs, replace=False)
        colors = self.bubble_map.color[indices]
        if USE_CUPY:
            T_cw_g = cp.asarray(PoseTransform.inverse(T_wc), dtype=cp.float32)
            R_cw_g, t_cw_g = T_cw_g[:3, :3], T_cw_g[:3, 3]
            _mu_gpu = isinstance(self.bubble_map.mu, cp.ndarray)
            mu_sel = self.bubble_map.mu[indices].astype(cp.float32) if _mu_gpu else cp.asarray(self.bubble_map.mu[indices], dtype=cp.float32)
            pts_c_g = (R_cw_g @ mu_sel.T).T + t_cw_g
            x_g, y_g, z_g = pts_c_g[:, 0], pts_c_g[:, 1], pts_c_g[:, 2]
            u_g = (self.fx * x_g / (z_g + 1e-9)) + self.cx
            v_g = (self.fy * y_g / (z_g + 1e-9)) + self.cy
            valid_g = (z_g > 0.1) & (z_g < 10.0) & (u_g >= 3) & (u_g < w - 3) & (v_g >= 3) & (v_g < h - 3)
            if int(cp.sum(valid_g)) < 20:
                return 0.0
            u_i = cp.asnumpy(u_g[cp.asnumpy(valid_g)]).astype(int)
            v_i = cp.asnumpy(v_g[cp.asnumpy(valid_g)]).astype(int)
            z_i = cp.asnumpy(z_g[cp.asnumpy(valid_g)])
            if isinstance(colors, cp.ndarray):
                colors_valid_g = colors[cp.asnumpy(valid_g)]
            else:
                colors_gpu_full = cp.asarray(colors, dtype=cp.float32)
                colors_valid_g = colors_gpu_full[cp.asnumpy(valid_g)]
        else:
            T_cw = PoseTransform.inverse(T_wc)
            R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
            _mu = self.bubble_map.mu[indices]
            if isinstance(_mu, cp.ndarray):
                _mu = cp.asnumpy(_mu)
            pts_c_n = (R_cw @ _mu.T).T + t_cw
            x, y, z = pts_c_n[:, 0], pts_c_n[:, 1], pts_c_n[:, 2]
            u = (self.fx * x / (z + 1e-9)) + self.cx
            v = (self.fy * y / (z + 1e-9)) + self.cy
            valid = (z > 0.1) & (z < 10.0) & (u >= 3) & (u < w - 3) & (v >= 3) & (v < h - 3)
            if np.sum(valid) < 20:
                return 0.0
            u_i, v_i, z_i = u[valid].astype(int), v[valid].astype(int), z[valid]
            colors_valid_g = None
            colors_valid = colors[valid] if not isinstance(colors, cp.ndarray) else cp.asnumpy(colors)[valid]
        obs_z = observed_depth[v_i, u_i]
        depth_valid = np.abs(obs_z - z_i) < 0.2
        if np.sum(depth_valid) < 10:
            return 0.0
        ui_dv = u_i[depth_valid]
        vi_dv = v_i[depth_valid]
        if USE_CUPY and colors_valid_g is not None:
            dv_idx = cp.asarray(depth_valid)
            bubble_gray_g = colors_valid_g[dv_idx].astype(cp.float32).mean(axis=1)
        else:
            bubble_gray_np = np.mean(
                colors_valid[depth_valid] if colors_valid_g is None else cp.asnumpy(colors_valid_g)[depth_valid],
                axis=1)
        if USE_CUPY and len(ui_dv) > 0:
            if isinstance(image, cp.ndarray):
                img_f = image.astype(cp.float32)
                img_gray_g = (0.114 * img_f[:, :, 0] +
                              0.587 * img_f[:, :, 1] +
                              0.299 * img_f[:, :, 2]) / 255.0
            else:
                img_gray_cpu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                img_gray_g = cp.asarray(img_gray_cpu)
            h_img, w_img = img_gray_g.shape
            ui_g = cp.asarray(ui_dv, dtype=cp.int32)
            vi_g = cp.asarray(vi_dv, dtype=cp.int32)
            dv_r = cp.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=cp.int32)
            du_c = cp.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=cp.int32)
            vi_nb = cp.clip(vi_g[:, None] + dv_r[None, :], 0, h_img - 1)
            ui_nb = cp.clip(ui_g[:, None] + du_c[None, :], 0, w_img - 1)
            patch_means = img_gray_g[vi_nb, ui_nb].mean(axis=1)
            bg_g = bubble_gray_g if 'bubble_gray_g' in locals() else cp.asarray(bubble_gray_np, dtype=cp.float32)
            score_arr = 1.0 - cp.abs(patch_means - bg_g)
            return float(cp.mean(score_arr)) if len(score_arr) > 0 else 0.0
        else:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            h_img, w_img = img_gray.shape
            dv_r = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int32)
            du_c = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
            vi_nb = np.clip(vi_dv[:, None] + dv_r[None, :], 0, h_img - 1)
            ui_nb = np.clip(ui_dv[:, None] + du_c[None, :], 0, w_img - 1)
            patch_means = img_gray[vi_nb, ui_nb].mean(axis=1)
            score_arr = 1.0 - np.abs(patch_means - bubble_gray_np)
            return float(np.mean(score_arr)) if len(score_arr) > 0 else 0.0

    def _score_feature_alignment(self, T_wc, feats_curr, keyframes):
        if len(keyframes) == 0 or feats_curr is None:
            return 0.0
        kf_poses = np.array([kf.pose[:3, 3] for kf in keyframes])
        dists = np.linalg.norm(kf_poses - T_wc[:3, 3], axis=1)
        nearest_idx = np.argmin(dists)
        nearest_kf = keyframes[nearest_idx]
        if nearest_kf.keypoints is None:
            return 0.0
        if nearest_kf.depth is None or nearest_kf.intrinsics is None:
            return 0.0
        K_kf = nearest_kf.intrinsics
        fx_kf = float(K_kf[0, 0])
        fy_kf = float(K_kf[1, 1])
        cx_kf = float(K_kf[0, 2])
        cy_kf = float(K_kf[1, 2])
        kpts_kf = nearest_kf.keypoints
        if USE_CUPY and nearest_kf.depth is not None:
            _d_g = cp.asarray(nearest_kf.depth) if not isinstance(nearest_kf.depth, cp.ndarray) else nearest_kf.depth
            kp_g = cp.asarray(kpts_kf, dtype=cp.float32)
            z_kf_g = bilinear_depth_gpu(_d_g, kp_g[:, 0], kp_g[:, 1], return_gpu=True)
            valid_z_g = (z_kf_g > 0.05) & (z_kf_g < 20.0)
            if int(cp.sum(valid_z_g)) < 10:
                return 0.0
            kp_g64 = kp_g.astype(cp.float64)
            z_kf_g64 = z_kf_g.astype(cp.float64)
            u_v_g = kp_g64[valid_z_g, 0]
            v_v_g = kp_g64[valid_z_g, 1]
            z_v_g = z_kf_g64[valid_z_g]
            pts_cam_g = cp.column_stack([(u_v_g - cx_kf) * z_v_g / fx_kf,
                                         (v_v_g - cy_kf) * z_v_g / fy_kf, z_v_g])
            kfpose_g = cp.asarray(nearest_kf.pose, dtype=cp.float64)
            pts_world_g = (kfpose_g[:3, :3] @ pts_cam_g.T).T + kfpose_g[:3, 3]
            T_cw_g = cp.asarray(PoseTransform.inverse(T_wc), dtype=cp.float64)
            pts3d_cam_g = (T_cw_g[:3, :3] @ pts_world_g.T).T + T_cw_g[:3, 3]
            z3_g = pts3d_cam_g[:, 2]
            u_proj_g = (self.fx * pts3d_cam_g[:, 0] / (z3_g + 1e-9)) + self.cx
            v_proj_g = (self.fy * pts3d_cam_g[:, 1] / (z3_g + 1e-9)) + self.cy
            N_proj = int(pts3d_cam_g.shape[0])
            _use_gpu_kd = HAS_CUPY_KDTREE
        else:
            z_kf = bilinear_depth(nearest_kf.depth, kpts_kf[:, 0], kpts_kf[:, 1])
            valid_z = (z_kf > 0.05) & (z_kf < 20.0)
            if np.sum(valid_z) < 10:
                return 0.0
            u_v = kpts_kf[valid_z, 0]
            v_v = kpts_kf[valid_z, 1]
            z_v = z_kf[valid_z]
            pts_cam = np.column_stack([(u_v - cx_kf) * z_v / fx_kf,
                                       (v_v - cy_kf) * z_v / fy_kf, z_v])
            pts_world = PoseTransform.transform_points(nearest_kf.pose, pts_cam)
            T_cw = PoseTransform.inverse(T_wc)
            pts3d_cam = PoseTransform.transform_points(T_cw, pts_world)
            u_proj = (self.fx * pts3d_cam[:, 0] / pts3d_cam[:, 2]) + self.cx
            v_proj = (self.fy * pts3d_cam[:, 1] / pts3d_cam[:, 2]) + self.cy
            N_proj = len(pts3d_cam)
            _use_gpu_kd = False
        kpts_curr = feats_curr['keypoints'].cpu().numpy()
        try:
            kf_id = id(nearest_kf)
            if _use_gpu_kd:
                proj_g = cp.column_stack([u_proj_g, v_proj_g]).astype(cp.float32)
                if self._reloc_cached_kf_id != kf_id:
                    kpts_curr_g = cp.asarray(kpts_curr, dtype=cp.float32)
                    self._reloc_cached_kf_tree = CupyKDTree(kpts_curr_g)
                    self._reloc_cached_kf_id = kf_id
                dists_g, _ = self._reloc_cached_kf_tree.query(proj_g, k=1)
                inliers = int(cp.sum(dists_g.ravel() < 5.0))
            else:
                if USE_CUPY and 'u_proj_g' in locals():
                    u_proj = cp.asnumpy(u_proj_g)
                    v_proj = cp.asnumpy(v_proj_g)
                proj_pts = np.stack([u_proj, v_proj], axis=1)
                if self._reloc_cached_kf_id != kf_id:
                    self._reloc_cached_kf_tree = cKDTree(kpts_curr)
                    self._reloc_cached_kf_id = kf_id
                dists_kd, _ = self._reloc_cached_kf_tree.query(proj_pts, k=1)
                inliers = np.sum(dists_kd < 5.0)
            score = inliers / N_proj if N_proj > 0 else 0.0
            return float(score)
        except Exception:
            return 0.0

    def score_hypothesis(self, T_wc, depth, image, feats):
        score_geom = self._score_depth_alignment(T_wc, depth, use_voxel=True)
        if score_geom < 0.1:
            return score_geom
        score_photo = self._score_photometric_consistency(T_wc, image, depth)
        score_feat = self._score_feature_alignment(T_wc, feats, self.voxel_manager.keyframes)
        return 0.5 * score_geom + 0.3 * score_photo + 0.2 * score_feat

    def evolve_ghosts(self, depth, image, feats):
        if len(self.particles) == 0:
            return None, 0.0

        # Ensure bubble map cache is valid before scoring multiple particles
        if hasattr(self.bubble_map, "validate_cache"):
            # Trigger property access to fill cache
            _ = self.bubble_map.mu
            _ = self.bubble_map.weight
            _ = self.bubble_map.color
            _ = self.bubble_map.Sigma
            self.bubble_map.validate_cache()

        self.iteration += 1

        def _score_one(item):
            T, _, vel = item
            s = self.score_hypothesis(T, depth, image, feats)
            return T, s, vel

        futures = {self._score_executor.submit(_score_one, p): i for i, p in enumerate(self.particles)}
        scored = [None] * len(self.particles)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                scored[idx] = fut.result()
            except Exception:
                T, _, vel = self.particles[idx]
                scored[idx] = (T, 0.0, vel)
        for T, score, vel in scored:
            if score > self.best_score:
                self.best_score = score
                self.best_particle = T.copy()
        scored.sort(key=lambda x: x[1], reverse=True)
        elite_count = max(4, len(scored) // 5)
        elites = scored[:elite_count]
        progress = self.iteration / self.max_iterations
        st = self.base_spread_t * (1.0 - progress * 0.8)
        sr = self.base_spread_r * (1.0 - progress * 0.8)
        new_particles = list(elites)
        n_needed = len(self.particles) - len(new_particles)
        if n_needed > 0:
            if USE_CUPY:
                parent_idx_arr = np.random.randint(0, len(elites), n_needed)
                parents_np = np.stack([elites[i][0] for i in parent_idx_arr])
                xi_g = cp.concatenate([
                    cp.random.normal(0, st, (n_needed, 3), dtype=cp.float64),
                    cp.random.normal(0, sr, (n_needed, 3), dtype=cp.float64)], axis=1)
                parents_g = cp.asarray(parents_np, dtype=cp.float64)
                deltas_g = batch_exp_se3_gpu(xi_g)
                scattered = cp.asnumpy(batch_matmul_gpu(parents_g, deltas_g))
                for T_new in scattered:
                    new_particles.append((T_new, 0.0, np.zeros(6)))
            else:
                while len(new_particles) < len(self.particles):
                    pT, _, _ = elites[np.random.randint(len(elites))]
                    xi = np.concatenate([np.random.normal(0, st, 3), np.random.normal(0, sr, 3)])
                    new_particles.append((pT @ PoseTransform.exp_se3(xi), 0.0, np.zeros(6)))
        self.particles = new_particles
        if self.best_score >= self.min_score_threshold and self.best_score >= self.convergence_threshold:
            return self.best_particle, self.best_score
        return None, self.best_score

    def visualize_ghosts_pyvista(self, max_ghosts=50):
        if len(self.particles) == 0:
            return None
        sorted_particles = sorted(self.particles, key=lambda x: x[1], reverse=True)
        step = max(1, len(sorted_particles) // max_ghosts)
        points = []
        for i, (T, score, _) in enumerate(sorted_particles[::step][:max_ghosts]):
            points.append(T[:3, 3])
        return np.array(points) if points else None

    def get_best_pose(self, min_score=None):
        thr = min_score if min_score is not None else self.min_score_threshold
        if self.best_score >= thr and self.best_particle is not None:
            return self.best_particle.copy()
        return None