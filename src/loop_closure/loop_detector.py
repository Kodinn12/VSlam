"""Loop closure detection using LightGlue and GPU PnP."""

import numpy as np
import cv2
import torch
from ..utils.logger import get_logger
from ..utils.depth_utils import bilinear_depth_gpu, bilinear_depth
from ..utils.se3_ops import PoseTransform, USE_CUPY

logger = get_logger(__name__)

try:
    import cupy as cp
    if not USE_CUPY:
        raise ImportError
except ImportError:
    cp = np
    USE_CUPY = False

# Import the GPU PnP function from utils (defined in the original script)
# We assume it is available as _batched_gpu_pnp_ransac in utils.pnp
try:
    from ..utils.pnp import batched_gpu_pnp_ransac  # type: ignore
except ImportError:
    # If pnp module doesn't exist, define a placeholder
    def batched_gpu_pnp_ransac(*args, **kwargs):
        raise NotImplementedError("batched_gpu_pnp_ransac not available")

class LoopClosureDetector:
    def __init__(self, config, K_proc, sp_lg):
        self.config = config
        self.K = K_proc
        self.sp_lg = sp_lg
        self.kf_counter = 0

    def detect_and_verify(self, current_kf, history_kfs):
        self.kf_counter += 1
        if self.kf_counter % self.config.get("loop_check_frequency", 12) != 0:
            return False, None
        if len(history_kfs) < self.config["loop_temporal_window"]:
            return False, None
        candidates = history_kfs[:-self.config["loop_temporal_window"]]
        if not candidates:
            return False, None
        limit = self.config.get("loop_candidate_limit", 20)
        if len(candidates) > limit:
            candidates = candidates[-limit:]
        curr_feats = current_kf.get_gpu_feats(self.sp_lg.device)
        if curr_feats is None:
            return False, None
        best_kf = None
        max_inliers = 0
        best_pose_w = None
        for past_kf in candidates:
            if past_kf.keypoints is None or len(past_kf.keypoints) < 50:
                continue
            past_feats = past_kf.get_gpu_feats(self.sp_lg.device)
            if past_feats is None:
                continue
            matches, scores = self.sp_lg.match(curr_feats, past_feats)
            vm = (matches > -1) & (scores > self.config['match_threshold'])
            vi = torch.where(vm)[0]
            if len(vi) < self.config['loop_min_matches']:
                continue
            mn = matches.cpu().numpy()
            vn = vi.cpu().numpy()
            pm = mn[vn].astype(np.int64)
            kp_c = current_kf.keypoints[vn]
            kp_p = past_kf.keypoints[pm]
            fx = self.K[0, 0]
            cx = self.K[0, 2]
            fy = self.K[1, 1]
            cy = self.K[1, 2]
            if USE_CUPY and past_kf.depth is not None:
                _dep_g = past_kf.depth if isinstance(past_kf.depth, cp.ndarray) else cp.asarray(past_kf.depth)
                kp_p_g = cp.asarray(kp_p, dtype=cp.float32)
                z_g = bilinear_depth_gpu(_dep_g, kp_p_g[:, 0], kp_p_g[:, 1], return_gpu=True)
                vz_g = (z_g > 0.1) & (z_g < 10.0)
                if int(cp.sum(vz_g)) < self.config['lm_min_inliers']:
                    continue
                u_vg = kp_p_g[vz_g, 0]
                v_vg = kp_p_g[vz_g, 1]
                z_vg = z_g[vz_g]
                pts_cam_g = cp.column_stack([(u_vg - cx) * z_vg / fx,
                                             (v_vg - cy) * z_vg / fy, z_vg])
                pose_g = cp.asarray(past_kf.pose, dtype=cp.float64)
                pts_w = cp.asnumpy((pose_g[:3, :3] @ pts_cam_g.T).T + pose_g[:3, 3])
                vz = cp.asnumpy(vz_g)
                curr_2d = kp_c[vz]
            else:
                z = bilinear_depth(past_kf.depth, kp_p[:, 0], kp_p[:, 1])
                vz = (z > 0.1) & (z < 10.0)
                if np.sum(vz) < self.config['lm_min_inliers']:
                    continue
                u_v = kp_p[vz, 0]
                v_v = kp_p[vz, 1]
                z_v = z[vz]
                pts_cam = np.column_stack([(u_v - cx) * z_v / fx,
                                           (v_v - cy) * z_v / fy, z_v])
                pts_w = PoseTransform.transform_points(past_kf.pose, pts_cam)
                curr_2d = kp_c[vz]

            # GPU PnP path
            if hasattr(batched_gpu_pnp_ransac, '__call__'):
                pts_w_t = torch.as_tensor(pts_w.astype(np.float32), device='cuda')
                c2d_t = torch.as_tensor(curr_2d.astype(np.float32), device='cuda')
                K_t = torch.as_tensor(self.K[:3, :3].astype(np.float32), device='cuda')
                ok_g, T_cw_g, inl_mask_g, n_inl_g = batched_gpu_pnp_ransac(
                    pts_w_t, c2d_t, K_t,
                    reproj_thresh=self.config['loop_ransac_thresh'],
                    n_iter=200)
                if ok_g and n_inl_g >= self.config['loop_min_matches'] // 2:
                    ok = ok_g
                    inliers = np.where(inl_mask_g)[0].reshape(-1, 1)
                    T_cw = T_cw_g
                else:
                    # fallback to OpenCV
                    ok, rvec, tvec, inl = cv2.solvePnPRansac(
                        pts_w.astype(np.float64), curr_2d.astype(np.float64).reshape(-1, 2),
                        self.K.astype(np.float64), None,
                        reprojectionError=self.config['loop_ransac_thresh'],
                        iterationsCount=200,
                        flags=cv2.SOLVEPNP_SQPNP if hasattr(cv2, 'SOLVEPNP_SQPNP') else cv2.SOLVEPNP_EPNP)
                    if ok and inl is not None:
                        inl_flat = inl.flatten()
                        try:
                            rvec, tvec = cv2.solvePnPRefineLM(
                                pts_w[inl_flat].astype(np.float64),
                                curr_2d[inl_flat].astype(np.float64).reshape(-1, 2),
                                self.K.astype(np.float64), None, rvec, tvec)
                        except Exception:
                            pass
                        R_l, _ = cv2.Rodrigues(rvec)
                        T_cw = np.eye(4)
                        T_cw[:3, :3] = R_l
                        T_cw[:3, 3] = tvec.flatten()
                    else:
                        ok = False
                        T_cw = None
            else:
                # CPU PnP only
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    pts_w.astype(np.float64), curr_2d.astype(np.float64).reshape(-1, 2),
                    self.K.astype(np.float64), None,
                    reprojectionError=self.config['loop_ransac_thresh'],
                    iterationsCount=200,
                    flags=cv2.SOLVEPNP_SQPNP if hasattr(cv2, 'SOLVEPNP_SQPNP') else cv2.SOLVEPNP_EPNP)
                if ok and inl is not None:
                    inl_flat = inl.flatten()
                    try:
                        rvec, tvec = cv2.solvePnPRefineLM(
                            pts_w[inl_flat].astype(np.float64),
                            curr_2d[inl_flat].astype(np.float64).reshape(-1, 2),
                            self.K.astype(np.float64), None, rvec, tvec)
                    except Exception:
                        pass
                    R_l, _ = cv2.Rodrigues(rvec)
                    T_cw = np.eye(4)
                    T_cw[:3, :3] = R_l
                    T_cw[:3, 3] = tvec.flatten()
                else:
                    T_cw = None
            if not ok or inl is None or len(inl) < self.config['loop_min_matches'] // 2:
                continue
            if len(inl) > max_inliers:
                max_inliers = len(inl)
                best_pose_w = PoseTransform.inverse(T_cw)
                best_kf = past_kf
        if best_kf is None or best_pose_w is None:
            return False, None
        logger.info(f"Loop detected: KF#{current_kf.id} ↔ KF#{best_kf.id} ({max_inliers} inliers)")
        return True, best_pose_w