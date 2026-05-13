import numpy as np
import torch
import math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict as _dd

from ..utils.logger import get_logger
from ..utils.se3_ops import batch_exp_se3_gpu, batch_matmul_gpu, USE_CUPY
from ..utils.linear_algebra import batch_inv3_gpu
from ..utils.depth_utils import bilinear_depth_gpu

from ..utils.cupy_utils import cupy_manager, cp, USE_CUPY
xp = cp

logger = get_logger(__name__)

# Helper function for scatter add (GPU or CPU)
if USE_CUPY:
    try:
        import importlib
        _cupyx = importlib.import_module("cupyx")
        _cupyx_scatter_add = _cupyx.scatter_add
        def _cp_scatter_add(a, slices, value):
            _cupyx_scatter_add(a, slices, value)
    except (ImportError, AttributeError):
        def _cp_scatter_add(a, slices, value):
            cp.scatter_add(a, slices, value)
else:
    def _cp_scatter_add(a, slices, value):
        np.add.at(a, slices, value)

# Function aliases
_batch_exp_se3_gpu = batch_exp_se3_gpu
_batch_matmul_gpu = batch_matmul_gpu
_batch_inv3_gpu = batch_inv3_gpu

# CPU fallback for batch inverse
def _batch_inv3(H_pp_blocks):
    """Batch invert 3x3 blocks (CPU fallback)."""
    try:
        return np.linalg.inv(H_pp_blocks)
    except np.linalg.LinAlgError:
        # Singular block; use pseudo-inverse
        return np.linalg.pinv(H_pp_blocks)

class CuPyBundleAdjuster:
    """
    Sliding-window bundle adjustment over the last N keyframes.

    Algorithm (Schur-complement LM):
    ─────────────────────────────────
    Variables:
        xi_cam[i]  ∈ ℝ^6   (se(3) correction for camera i, i=0..N_cam-1)
        dP[k]      ∈ ℝ^3   (correction for 3D landmark k, k=0..N_pts-1)

    Per observation (camera i, landmark k, pixel measurement u_obs):
        residual r_ik = u_proj(T_i, P_k) - u_obs   ∈ ℝ^2

    Jacobians:
        J_cam_ik  = ∂r/∂xi_i   (2×6)
        J_pts_ik  = ∂r/∂P_k    (2×3)

    Normal equations (blocked):
        [H_cc | H_cp] [Δξ ]   [-g_c]
        [H_pc | H_pp] [ΔP ]  = [-g_p]

    Schur complement (landmark variables eliminated):
        S   = H_cc - H_cp H_pp^{-1} H_pc   (6N_cam × 6N_cam dense)
        rhs = g_c - H_cp H_pp^{-1} g_p
        Solve S Δξ = -rhs  (on GPU for large systems)

    Back-substitution - CORRECT FORMULA (V15 fix, V18 doc fix):
        Normal eqns row 2:  H_cp^T Δξ + H_pp ΔP = -g_p
        -> ΔP = -H_pp^{-1} (g_p + H_cp^T Δξ)
        NOTE: sign is MINUS outside, PLUS inside (NOT g_p - H_cp^T Δξ).
        The old docstring had g_p - H_pc Δξ which was wrong.

    H_pp is block-diagonal (each 3×3 block per landmark), so inversion
    is cheap: O(3^3 × N_pts) rather than O(N_pts^3).

    GPU:  all H_cc, H_cp, g_c/g_p accumulation done in CuPy.
          Schur solve via cp.linalg.solve or np.linalg.solve (6N_cam × 6N_cam).
          _batch_inv3_gpu handles singular H_pp blocks safely (V18 fix).
    """

    def __init__(self, K: np.ndarray, config: dict):
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
        self.K           = K.copy()
        self.cfg         = config
        self.win_size    = config.get("ba_window_size", 8)
        self.max_iter    = config.get("ba_max_iterations", 12)
        self.lam_init    = config.get("ba_lambda_init", 1e-3)
        self.conv_delta  = config.get("ba_convergence_delta", 1e-6)
        self.huber_c     = config.get("ba_huber_thresh", 2.0)
        self.huber_c_depth = config.get("ba_depth_huber_thresh", 0.12)   # V33: separate depth Huber (metres)
        self.ba_max_step_norm = config.get("ba_max_step_norm", 5.0)      # V33: per-DOF step clamp
        self.min_track   = config.get("ba_min_track_length", 2)
        self.max_reproj  = config.get("ba_max_reproj_error", 5.0)
        # V30: CuPy memory pool warm-up - pre-touch the default pool so the
        # first BA call doesn't pay the pool initialization penalty.
        if USE_CUPY:
            try:
                _warmup = cp.zeros(1, dtype=cp.float64)
                del _warmup
            except Exception:
                pass
        print(f" [BA] CuPy Bundle Adjuster  "
              f"window={self.win_size}  max_iter={self.max_iter}  "
              f"GPU={'CuPy' if USE_CUPY else 'NumPy'}  "
              f"huber_reproj={self.huber_c}px  huber_depth={self.huber_c_depth}m  (V33)")

    # ------------------------------------------------------------------
    def run(self, keyframes: List, sp_lg) -> Tuple[List[np.ndarray], bool]:
        """
        Run windowed bundle adjustment over the last window_size keyframes.

        Parameters
        ----------
        keyframes : list[Keyframe] - full history; BA uses the last win_size.
        sp_lg     : SuperPointLightGlue - used for cross-KF matching.

        Returns
        -------
        corrected_poses : list[np.ndarray] - corrected 4×4 poses for the
                          window keyframes (same order as the slice used).
        converged       : bool
        """
        if len(keyframes) < 2:
            return [kf.pose.copy() for kf in keyframes], False

        # Slice the window
        win_kfs = keyframes[-self.win_size:]
        N_cam   = len(win_kfs)

        # ── Build landmark tracks across window keyframes ─────────────
        pts3d_list, obs_list = self._build_tracks(win_kfs, sp_lg)

        N_pts = len(pts3d_list)
        if N_pts < 6:
            return [kf.pose.copy() for kf in win_kfs], False

        # CPU zone: build arrays using NumPy
        pts3d = np.array(pts3d_list, dtype=np.float64)   # (N_pts, 3)
        # obs_list: list of (cam_idx, pt_idx, u, v) - one entry per observation
        if len(obs_list) < 8:
            return [kf.pose.copy() for kf in win_kfs], False

        # CPU zone: LM bundle adjustment using NumPy (GPU transfer happens in _lm_optimize)
        poses = np.array([kf.pose.copy() for kf in win_kfs])   # (N_cam,4,4)
        poses, pts3d, converged = self._lm_optimize(poses, pts3d, obs_list, N_cam, N_pts)

        # Fix scale / first pose so the window doesn't drift
        # (keep first camera fixed; only corrections to others are returned)
        return list(poses), converged

    # ------------------------------------------------------------------
    def _build_tracks(self, win_kfs: List, sp_lg) -> Tuple[List, List]:
        """
        Build 3D landmark tracks by matching keyframe pairs with LightGlue.

        Strategy: match every consecutive pair and transitively chain into
        multi-view tracks using a union-find structure.

        Returns:
            pts3d_list : list of (3,) world-space landmark positions
            obs_list   : list of (cam_idx, pt_idx, u, v)
        """
        N = len(win_kfs)
        # Per-keyframe: lift 2D keypoints to 3D using depth
        kf_pts3d  = []   # kf_pts3d[i] = (M_i, 3) local 3D points in world
        kf_pts2d  = []   # kf_pts2d[i] = (M_i, 2) pixel coords
        kf_valid  = []   # boolean mask

        for kf in win_kfs:
            if (kf.keypoints is None or kf.depth is None or
                    kf.intrinsics is None):
                kf_pts3d.append(np.empty((0, 3))); kf_pts2d.append(np.empty((0, 2)))
                kf_valid.append(np.zeros(0, dtype=bool))
                continue
            pts_w, valid_idx = kf.get_3d_points(kf.intrinsics)
            if pts_w is None or len(pts_w) == 0:
                kf_pts3d.append(np.empty((0, 3))); kf_pts2d.append(np.empty((0, 2)))
                kf_valid.append(np.zeros(0, dtype=bool))
                continue
            valid_idx = np.array(valid_idx, dtype=np.int32)
            kf_pts3d.append(pts_w)
            kf_pts2d.append(kf.keypoints[valid_idx])
            kf_valid.append(valid_idx)

        # Union-Find for track merging
        # Each local observation = (i, local_pt_idx) -> global landmark
        obs_to_global: Dict[Tuple[int,int], int] = {}
        global_pts3d: List[np.ndarray] = []
        parent: List[int] = []

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        # Register all local points as initial global landmarks
        kf_g_start = []
        for i in range(N):
            kf_g_start.append(len(global_pts3d))
            for li in range(len(kf_pts3d[i])):
                gid = len(global_pts3d)
                global_pts3d.append(kf_pts3d[i][li])
                parent.append(gid)
                obs_to_global[(i, li)] = gid

        thresh = self.cfg.get("match_threshold", 0.10)
        # Match consecutive AND skip-one keyframe pairs for richer constraints.
        # V21: added (i, i+2) pairs - doubles long-range constraints within the
        # window, significantly improving BA convergence and reducing drift for
        # window sizes >= 4.  Consecutive (i, i+1) pairs handle small-baseline
        # geometry; skip-one (i, i+2) pairs catch wider-baseline correspondences
        # that are often more stable for depth-sensitive operations.
        pairs_to_match = []
        for i in range(N - 1):
            pairs_to_match.append((i, i + 1))          # consecutive
        for i in range(N - 2):
            pairs_to_match.append((i, i + 2))          # skip-one

        for (i, j) in pairs_to_match:
            kf_i, kf_j = win_kfs[i], win_kfs[j]
            fi = kf_i.get_gpu_feats(sp_lg.device)
            fj = kf_j.get_gpu_feats(sp_lg.device)
            if fi is None or fj is None:
                continue
            try:
                matches, scores = sp_lg.match(fi, fj)
            except Exception:
                continue

            # V11: filter matches on GPU before D2H - only download valid rows.
            # Replaces matches.cpu().numpy() + scores.cpu().numpy() (full N_kp arrays)
            # with filtered index arrays (only valid matches).
            valid_m = (matches > -1) & (scores > thresh)
            feat_i_t = torch.where(valid_m)[0]               # indices with valid match
            if len(feat_i_t) == 0:
                continue
            feat_j_t = matches[feat_i_t]                      # corresponding match indices
            feat_i_np = feat_i_t.cpu().numpy()                # smaller D2H
            feat_j_np = feat_j_t.cpu().numpy()

            # Map feature indices -> local 3D point indices
            vi_valid = kf_valid[i]
            vj_valid = kf_valid[j]
            # Build reverse maps: feat_idx -> local 3D idx
            feat_to_local_i = {}
            if len(vi_valid) > 0:
                for li, fi_feat in enumerate(vi_valid):
                    feat_to_local_i[int(fi_feat)] = li
            feat_to_local_j = {}
            if len(vj_valid) > 0:
                for lj, fj_feat in enumerate(vj_valid):
                    feat_to_local_j[int(fj_feat)] = lj

            for feat_i, feat_j in zip(feat_i_np, feat_j_np):
                feat_i = int(feat_i); feat_j = int(feat_j)
                li_idx = feat_to_local_i.get(feat_i)
                lj_idx = feat_to_local_j.get(feat_j)
                if li_idx is None or lj_idx is None:
                    continue
                gi = obs_to_global.get((i, li_idx))
                gj = obs_to_global.get((j, lj_idx))
                if gi is None or gj is None:
                    continue
                union(gi, gj)

        # Collect final track roots and filter by min track length
        root_to_obs: Dict[int, List[Tuple[int,int,float,float]]] = {}
        for (cam_i, local_i), gid in obs_to_global.items():
            root = find(gid)
            if cam_i >= len(kf_pts2d) or local_i >= len(kf_pts2d[cam_i]):
                continue
            u, v = float(kf_pts2d[cam_i][local_i][0]), float(kf_pts2d[cam_i][local_i][1])
            root_to_obs.setdefault(root, []).append((cam_i, local_i, u, v))

        # Only keep tracks visible in >= min_track keyframes
        pts3d_out = []
        obs_out   = []
        dw        = self.cfg.get("ba_depth_weight", 0.5)

        # V42: pre-build a GPU depth map per keyframe for vectorized bilinear sampling.
        # V41 called scalar Python math inside the observation loop - for N_obs=1000
        # that is 1000 Python iterations of floor/clip/multiply.  Instead, group
        # all observations by cam_i and call bilinear_depth_gpu once per keyframe.
        # bilinear_depth_gpu accepts np/cp u,v arrays and returns CPU float64 array.
        # This replaces the entire scalar inner loop with 1 GPU kernel per keyframe.
        _depth_cache: Dict[int, Optional[np.ndarray]] = {}   # cam_i -> depth CPU array
        def _get_kf_depth(cam_i: int):
            if cam_i not in _depth_cache:
                kf = win_kfs[cam_i]
                _depth_cache[cam_i] = kf.depth if kf.depth is not None else None
            return _depth_cache[cam_i]

        # Group obs by cam_i so we can batch the bilinear lookups per KF
        from collections import defaultdict as _dd
        _obs_by_cam: Dict[int, List[Tuple[int, float, float, int]]] = _dd(list)
        # obs_indexed: (root_idx, cam_i, local_i, u, v) flattened
        _obs_indexed = []
        for root, obs in root_to_obs.items():
            cam_set = {o[0] for o in obs}
            if len(cam_set) < self.min_track:
                continue
            pt_idx_local = len(pts3d_out)
            all_pts = [global_pts3d[obs_to_global[(o[0], o[1])]]
                       for o in obs if (o[0], o[1]) in obs_to_global]
            if not all_pts:
                continue
            pts3d_out.append(np.mean(all_pts, axis=0))
            for (cam_i, local_i, u, v) in obs:
                _obs_indexed.append((pt_idx_local, cam_i, u, v))
                _obs_by_cam[cam_i].append((len(_obs_indexed) - 1, u, v))

        # Vectorized bilinear depth per keyframe - one GPU kernel call per KF
        _z_map: Dict[int, float] = {}   # obs linear index -> z_obs
        for cam_i, entries in _obs_by_cam.items():
            depth_kf = _get_kf_depth(cam_i)
            if dw <= 0.0 or depth_kf is None:
                for (idx, u, v) in entries:
                    _z_map[idx] = -1.0
                continue
            idxs_e = [e[0] for e in entries]
            u_arr  = np.array([e[1] for e in entries], dtype=np.float32)
            v_arr  = np.array([e[2] for e in entries], dtype=np.float32)
            # Use existing GPU bilinear kernel (handles GPU depth arrays directly)
            _d_g   = (depth_kf if isinstance(depth_kf, cp.ndarray)
                      else cp.asarray(depth_kf)) if USE_CUPY else depth_kf
            z_arr  = bilinear_depth_gpu(_d_g, u_arr, v_arr, return_gpu=False)  # CPU float64
            for i, idx in enumerate(idxs_e):
                zv = float(z_arr[i])
                _z_map[idx] = zv if 0.1 < zv < 10.0 else -1.0

        for obs_i, (pt_idx_local, cam_i, u, v) in enumerate(_obs_indexed):
            obs_out.append((cam_i, pt_idx_local, u, v, _z_map.get(obs_i, -1.0)))

        return pts3d_out, obs_out

    # ------------------------------------------------------------------
    def _lm_optimize(self, poses, pts3d, obs_list, N_cam, N_pts):
        """
        Levenberg-Marquardt with Schur complement.

        V13 FIXES:
          • cam_idx_per_res / pt_idx_per_res now correctly have one entry per
            residual row (2*M_valid), not one per observation, so _schur_solve
            receives consistent index arrays after valid-mask filtering.
          • Standard LM flow: compute step -> evaluate new cost -> accept/reject.
            The previous code checked cost BEFORE computing the step and skipped
            the step on upward cost, which is not LM.
          • All heavy ops run in CuPy when available.

        V26 FIXES:
          • Pre-upload u_obs, v_obs, cam_idx, pt_idx to GPU ONCE here.
            Previously these four arrays were re-uploaded inside every call to
            _residuals_jacobians (cp.asarray cost = M×8B + M×8B + M×4B + M×4B
            = ~24M B per iteration H2D, EVERY iteration including rejected steps).
            Pre-uploading eliminates ALL of these transfers - _residuals_jacobians
            receives xp arrays that cp.asarray() recognises as already on-device.
          • cam_idx_per_res / pt_idx_per_res also pre-uploaded to GPU.
            _schur_solve's xp.asarray() calls for these become no-ops.
          • On-device threshold unified: when USE_CUPY, always use GPU path
            regardless of M to keep all tensor ops on-device and avoid the
            NumPy->CuPy ping-pong when M crosses the 32-obs boundary mid-run.

        V28 FIXES:
          • N_RES = 3: residuals are now (u, v, z_depth) per observation.
            cam/pt per-res arrays use repeat(…, 3) instead of repeat(…, 2).
          • NaN REVERT BUG FIXED (CRITICAL): previously `poses` and `pts3d`
            were overwritten to `_new` BEFORE the NaN check, making it
            impossible to recover.  Now `poses_prev` / `pts3d_prev` are saved
            before any update so the genuine revert path is available.
          • z_obs pre-uploaded to GPU alongside other observation arrays.
        """
        N_RES     = 3     # residuals per observation: (u, v, z_depth)
        lam       = self.lam_init
        converged = False

        obs_arr = np.array(obs_list, dtype=np.float64)  # (M, 5): cam,pt,u,v,z
        cam_idx = obs_arr[:, 0].astype(np.int32)
        pt_idx  = obs_arr[:, 1].astype(np.int32)
        u_obs   = obs_arr[:, 2]
        v_obs   = obs_arr[:, 3]
        z_obs   = obs_arr[:, 4] if obs_arr.shape[1] >= 5 else np.full(len(obs_arr), -1.0)
        M       = len(obs_arr)

        # Per-residual-row cam/pt index arrays (length N_RES*M, interleaved u,v,z).
        # _schur_solve expects: cam_idx_per_res[N_RES*k : N_RES*k+N_RES] all equal.
        # V42 FIX: eliminated the GPU->CPU->GPU round-trip that V40 still had.
        # V40 built the repeated arrays on GPU with cp.repeat, then immediately
        # called cp.asnumpy() "for later re-use", then re-uploaded them at L4797-4798
        # via cp.asarray().  That was a full D2H + H2D per BA call for nothing.
        # Fix: keep GPU arrays as the primary reference; the CPU versions are only
        # needed in the NumPy fallback branch (line ~4841) so build them lazily there.
        if USE_CUPY and M > 0:
            cam_idx_g_tmp   = cp.asarray(cam_idx,  dtype=cp.int32)
            pt_idx_g_tmp    = cp.asarray(pt_idx,   dtype=cp.int32)
            cam_idx_res_xp  = cp.repeat(cam_idx_g_tmp, N_RES)  # stays on GPU
            pt_idx_res_xp   = cp.repeat(pt_idx_g_tmp,  N_RES)  # stays on GPU
            # CPU mirrors - only allocated here so the else-branch can share the name
            cam_idx_per_res = None   # built lazily below if needed (NumPy fallback)
            pt_idx_per_res  = None
        else:
            cam_idx_per_res = np.repeat(cam_idx, N_RES)   # (N_RES*M,) NumPy int32
            pt_idx_per_res  = np.repeat(pt_idx,  N_RES)   # (N_RES*M,) NumPy int32
            cam_idx_res_xp  = cam_idx_per_res
            pt_idx_res_xp   = pt_idx_per_res

        prev_cost = np.inf
        _accepted_steps = 0   # V27: track accepted steps for SO(3) re-ortho cadence
        _consec_rejects = 0   # V35: consecutive rejection counter for lam bail-out

        # V26: Pre-upload ALL observation arrays and pose/point arrays to GPU
        # in one batch.  This eliminates repeated H2D transfers inside both
        # _residuals_jacobians (u_obs, v_obs, cam_idx, pt_idx) and _schur_solve
        # (cam_idx_per_res, pt_idx_per_res) every LM iteration.
        # cp.asarray() on an already-resident CuPy array is a zero-copy no-op.
        use_gpu_path = USE_CUPY and M > 0
        if use_gpu_path:
            pts3d           = cp.asarray(pts3d,           dtype=cp.float64)  # H2D once
            poses           = cp.asarray(poses,           dtype=cp.float64)  # H2D once
            u_obs_xp        = cp.asarray(u_obs,           dtype=cp.float64)  # H2D once
            v_obs_xp        = cp.asarray(v_obs,           dtype=cp.float64)  # H2D once
            z_obs_xp        = cp.asarray(z_obs,           dtype=cp.float64)  # H2D once (V28)
            cam_idx_xp      = cp.asarray(cam_idx,         dtype=cp.int32)    # H2D once
            pt_idx_xp       = cp.asarray(pt_idx,          dtype=cp.int32)    # H2D once
            # V42: cam_idx_res_xp / pt_idx_res_xp already built on GPU above -
            # cp.asarray() here is a zero-copy no-op (they ARE cp.ndarrays).
            # No H2D transfer occurs; the cp.asarray guard is kept only for
            # future-proofing if the code path is ever reached with CPU arrays.
        else:
            u_obs_xp       = u_obs
            v_obs_xp       = v_obs
            z_obs_xp       = z_obs
            cam_idx_xp     = cam_idx
            pt_idx_xp      = pt_idx
            cam_idx_res_xp = cam_idx_per_res
            pt_idx_res_xp  = pt_idx_per_res

        for iteration in range(self.max_iter):
            # ── Build residuals + Jacobians ───────────────────────────
            # V26: passes pre-uploaded xp arrays - all xp.asarray() inside
            # _residuals_jacobians are no-ops (zero PCIe cost per iteration).
            (r_vec, J_cam_block, J_pts_block,
             weights, valid_mask) = self._residuals_jacobians(
                poses, pts3d, cam_idx_xp, pt_idx_xp,
                u_obs_xp, v_obs_xp, z_obs_xp, N_cam, N_pts)

            # ── Device-aware any() check and valid_np for NumPy cam/pt idx ──
            on_gpu = USE_CUPY and isinstance(r_vec, cp.ndarray)
            if on_gpu:
                if not bool(cp.any(valid_mask)):
                    break
                valid_np = cp.asnumpy(valid_mask)   # NumPy bool - for cam/pt index arrays
            else:
                valid_np = np.asarray(valid_mask, dtype=bool)
                if not np.any(valid_np):
                    break

            # ── Filter to valid residual rows (on device) ─────────────
            r_valid  = r_vec[valid_mask]          # CuPy or NumPy bool indexing ✓
            Jc_valid = J_cam_block[valid_mask]
            Jp_valid = J_pts_block[valid_mask]
            W_valid  = weights[valid_mask]

            # V26: cam/pt index arrays are pre-uploaded xp arrays.
            # Filter on device (CuPy bool mask) when on GPU path,
            # otherwise fall back to NumPy bool mask on NumPy arrays.
            if on_gpu:
                ci_valid = cam_idx_res_xp[valid_mask]   # CuPy bool fancy-indexing
                pi_valid = pt_idx_res_xp[valid_mask]
            else:
                # V42: cam_idx_per_res / pt_idx_per_res are None in the GPU path.
                # Build CPU arrays lazily - this branch is only reached on CPU fallback.
                if cam_idx_per_res is None:
                    cam_idx_per_res = cp.asnumpy(cam_idx_res_xp)
                    pt_idx_per_res  = cp.asnumpy(pt_idx_res_xp)
                ci_valid = cam_idx_per_res[valid_np]    # NumPy fallback
                pi_valid = pt_idx_per_res[valid_np]

            # ── Device-aware cost ─────────────────────────────────────
            xp_dev = cp if on_gpu else np
            cost = float(0.5 * xp_dev.sum(W_valid * r_valid * r_valid))

            # ── Schur complement solve ────────────────────────────────
            dx_cam, dx_pts = self._schur_solve(
                r_valid, Jc_valid, Jp_valid, W_valid,
                ci_valid, pi_valid, N_cam, N_pts, lam, N_RES)

            if dx_cam is None:
                lam = min(lam * 10.0, 1e6)
                continue

            # Guard against NaN/Inf in step vectors BEFORE applying.
            on_gpu_dx = USE_CUPY and isinstance(dx_cam, cp.ndarray)
            xp_step   = cp if on_gpu_dx else np
            if not bool(xp_step.all(xp_step.isfinite(dx_cam))):
                lam = min(lam * 10.0, 1e6)
                continue

            # V33 FIX: Per-DOF step norm clipping for BA camera increments.
            # When the window starts far from convergence (first BA call, or
            # after a large loop closure), the unclamped first Gauss-Newton
            # step can be enormous, pushing poses into degenerate regions where
            # exp_se3 produces near-singular R blocks and _residuals_jacobians
            # subsequently returns NaN costs.  LM damping alone cannot prevent
            # this because lam starts at lam_init=1e-3 which is far too small
            # relative to the Schur diagonal for very-large residual scenes.
            # Clipping per-DOF step norm (||dx_cam||/sqrt(6*N_cam)) to
            # ba_max_step_norm preserves direction while bounding step size;
            # the step is still accepted/rejected by the cost comparison below.
            _cam_norm_per_dof = float(xp_step.linalg.norm(dx_cam)) / max(
                1.0, math.sqrt(6 * N_cam))
            if _cam_norm_per_dof > self.ba_max_step_norm:
                _scale_cam = self.ba_max_step_norm / _cam_norm_per_dof
                dx_cam = dx_cam * _scale_cam   # new array (safe; no aliasing)

            # V35 FIX (CRITICAL): dx_pts must be clamped INDEPENDENTLY.
            # V33 incorrectly applied the camera-derived _scale to dx_pts.
            # This is WRONG: landmark increments live in metres (world coords)
            # and may be large/small independent of camera increments.
            # Scaling dx_pts by a camera-derived factor biases landmark updates
            # whenever cameras have large first-iteration residuals.
            # Fix: clamp dx_pts by its own per-DOF norm (||dx_pts||/sqrt(3*N_pts)).
            # Use ba_max_step_norm (same config key) - landmark DOFs are 3 vs 6,
            # so the per-DOF tolerance is the same physical magnitude threshold.
            xp_pts_step     = cp if (USE_CUPY and isinstance(dx_pts, cp.ndarray)) else np
            _pts_norm_per_dof = float(xp_pts_step.linalg.norm(dx_pts)) / max(
                1.0, math.sqrt(3 * N_pts))
            if _pts_norm_per_dof > self.ba_max_step_norm:
                _scale_pts = self.ba_max_step_norm / _pts_norm_per_dof
                dx_pts = dx_pts * _scale_pts

            # V30 FIX / V35 CLEANUP: guard dx_pts for NaN/Inf before applying update.
            # dx_pts can carry NaN from degenerate H_pp blocks not caught by
            # _batch_inv3_gpu's 1e-12 clamp (extreme conditioning edge cases).
            # xp_pts_step already defined above in V35 independent clamp block.
            on_gpu_pts_step = USE_CUPY and isinstance(dx_pts, cp.ndarray)
            xp_pts_step     = cp if on_gpu_pts_step else np
            if not bool(xp_pts_step.all(xp_pts_step.isfinite(dx_pts))):
                lam = min(lam * 10.0, 1e6)
                continue

            # ── V28 FIX: save prior state BEFORE applying the step ────────
            # Previously, poses and pts3d were overwritten BEFORE the NaN check,
            # making a genuine revert impossible.  Saving here enables the true
            # rollback path in the NaN guard below.
            poses_prev  = poses
            pts3d_prev  = pts3d

            # ── Try candidate update ──────────────────────────────────
            xi_batch   = dx_cam.reshape(N_cam, 6)             # CuPy or NumPy view
            exp_xi     = _batch_exp_se3_gpu(xi_batch)         # (N_cam,4,4) CuPy or NumPy
            poses_new  = _batch_matmul_gpu(poses, exp_xi)     # (N_cam,4,4) - matmul allocates new

            # V32 FIX: Guard poses_new for NaN/Inf BEFORE cost evaluation.
            # _batch_exp_se3_gpu uses Rodrigues which can produce non-finite
            # R/t for extreme dx_cam (e.g. near-singular Schur matrix producing
            # very large but finite dx_cam that explodes after exp map).
            # Without this guard, _residuals_jacobians receives a non-finite
            # poses_new, producing NaN cost and residuals - the step is rejected
            # (cost_new > cost) but the iteration wastes compute and can
            # occasionally corrupt pts3d_prev/poses_prev if accepted erroneously.
            on_gpu_poses_new = USE_CUPY and isinstance(poses_new, cp.ndarray)
            xp_poses_new     = cp if on_gpu_poses_new else np
            if not bool(xp_poses_new.all(xp_poses_new.isfinite(poses_new))):
                lam = min(lam * 10.0, 1e6)
                continue

            # pts3d_new: keep on-device if pts3d is CuPy and dx_pts is CuPy
            if USE_CUPY and isinstance(pts3d, cp.ndarray) and isinstance(dx_pts, cp.ndarray):
                pts3d_new = pts3d + dx_pts.reshape(N_pts, 3)   # (N_pts,3) CuPy
            elif USE_CUPY and isinstance(pts3d, cp.ndarray):
                pts3d_new = pts3d + cp.asarray(dx_pts.reshape(N_pts, 3), dtype=cp.float64)
            else:
                pts3d_new = pts3d + dx_pts.reshape(N_pts, 3)   # NumPy

            # V32 FIX: Guard pts3d_new for NaN/Inf BEFORE residual evaluation.
            # dx_pts may be finite yet pts3d_new can still overflow if pts3d
            # already carried extreme values from a prior bad step. Evaluating
            # _residuals_jacobians on non-finite pts3d_new produces NaN cost;
            # the step is rejected but wastes a full Jacobian/residual pass.
            on_gpu_pts_new = USE_CUPY and isinstance(pts3d_new, cp.ndarray)
            xp_pts_new     = cp if on_gpu_pts_new else np
            if not bool(xp_pts_new.all(xp_pts_new.isfinite(pts3d_new))):
                lam = min(lam * 10.0, 1e6)
                continue

            (r_new, _, _, w_new, vm_new) = self._residuals_jacobians(
                poses_new, pts3d_new, cam_idx_xp, pt_idx_xp,
                u_obs_xp, v_obs_xp, z_obs_xp, N_cam, N_pts)

            on_gpu_new = USE_CUPY and isinstance(r_new, cp.ndarray)
            xp_new     = cp if on_gpu_new else np
            cost_new   = float(0.5 * xp_new.sum(w_new[vm_new] * r_new[vm_new] ** 2))

            if cost_new < cost:
                # Accept step
                poses     = poses_new
                pts3d     = pts3d_new
                lam       = max(lam * 0.1, 1e-10)
                _accepted_steps += 1

                # V27 FIX: SO(3) re-orthogonalization every 3 accepted steps.
                if _accepted_steps % 3 == 0:
                    xp_ortho = cp if (USE_CUPY and isinstance(poses, cp.ndarray)) else np
                    # Must detach from matmul output BEFORE writing columns
                    poses = poses.copy()
                    R_blk = poses[:, :3, :3]                        # (N_cam,3,3) view
                    # Column 0: normalize
                    c0 = R_blk[:, :, 0].copy()
                    n0 = xp_ortho.linalg.norm(c0, axis=1, keepdims=True)
                    c0 = c0 / xp_ortho.maximum(n0, 1e-9)
                    # Column 1: remove c0 component, normalize
                    c1 = R_blk[:, :, 1].copy()
                    c1 = c1 - c0 * (c0 * c1).sum(axis=1, keepdims=True)
                    n1 = xp_ortho.linalg.norm(c1, axis=1, keepdims=True)
                    c1 = c1 / xp_ortho.maximum(n1, 1e-9)
                    # Column 2: cross product - guaranteed orthonormal
                    c2 = xp_ortho.cross(c0, c1)
                    poses[:, :3, 0] = c0
                    poses[:, :3, 1] = c1
                    poses[:, :3, 2] = c2

                # V28 FIX (NaN revert CORRECTED): detect NaN in pts3d and
                # properly revert to the saved prior state (poses_prev/pts3d_prev).
                # Previously this path had no valid rollback because prior values
                # were gone - now they are saved above before the update.
                on_gpu_pts = USE_CUPY and isinstance(pts3d, cp.ndarray)
                xp_pts     = cp if on_gpu_pts else np
                if not bool(xp_pts.all(xp_pts.isfinite(pts3d))):
                    # Genuine revert - restore saved pre-step state
                    poses = poses_prev
                    pts3d = pts3d_prev
                    lam   = min(lam * 10.0, 1e6)
                    # Do not update prev_cost
                else:
                    # V17: relative convergence check.
                    if np.isfinite(prev_cost):
                        rel_delta = abs(prev_cost - cost_new) / max(abs(prev_cost), 1e-10)
                        if rel_delta < self.conv_delta:
                            converged = True
                            break
                    prev_cost = cost_new
                _consec_rejects = 0   # V35: reset on accept
            else:
                # Reject step - increase damping
                lam = min(lam * 10.0, 1e6)
                _consec_rejects += 1
                # V35 FIX: Early bail-out when damping is maxed for 4+ consecutive
                # rejections.  At lam=1e6 the Schur step is ~0 - further iterations
                # pay full residual+Jacobian cost with zero useful update.
                if _consec_rejects >= 4 and lam >= 1e5:
                    break

        # ── Ensure poses are always returned as NumPy (4,4) float64 ──────────
        if USE_CUPY and isinstance(poses, cp.ndarray):
            poses = cp.asnumpy(poses)
        if USE_CUPY and isinstance(pts3d, cp.ndarray):
            pts3d = cp.asnumpy(pts3d)

        return poses, pts3d, converged

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _residuals_jacobians(self, poses, pts3d, cam_idx, pt_idx,
                              u_obs, v_obs, z_obs, N_cam, N_pts):
        """
        Compute 3-channel interleaved residuals r (3M,), Jc (3M,6), Jp (3M,3),
        weights W (3M,), and valid_mask (3M,) - one block of 3 per observation.

        Channel layout per observation k  (rows 3k, 3k+1, 3k+2):
          channel 0 (u): r_u = u_proj - u_obs
          channel 1 (v): r_v = v_proj - v_obs
          channel 2 (z): r_z = P_c[2] - z_obs   (depth; zero-weighted if invalid)

        Depth channel weight = ba_depth_weight * Huber(|r_z|).
        Channel 2 is zero-weighted (and marked invalid) when z_obs <= 0.1.

        V13 FIXES:
          1. ∂P_c/∂δρ = -I  (was -R_cw - wrong convention).
          2. All ops use xp (GPU-path was dead in earlier versions).

        V26 FIXES: inputs may be pre-uploaded CuPy; xp.asarray() is no-op.

        V28 FIXES:
          • z_obs parameter added (stereo depth per observation).
          • 3rd residual channel (depth) with correct Jacobians:
              J_c_z[row] = Jc_Pc[row 2] = [0, 0, -1, Py, -Px, 0]
              J_p_z[row] = R_cw[row 2, :]
          • valid_mask channel 2 also requires dw > 0 and has_depth.
        """
        M = len(cam_idx)
        if M == 0:
            return (np.zeros(0), np.zeros((0,6)), np.zeros((0,3)),
                    np.ones(0), np.zeros(0, dtype=bool))

        # V32 FIX: Select xp from actual input type (mirrors _schur_solve V30 fix).
        # Previously used `cp if USE_CUPY else np` - this picks CuPy even when
        # poses is a NumPy array (possible in CPU fallback / hybrid paths).
        # Using isinstance(poses, cp.ndarray) guarantees xp matches the memory
        # space of the input arrays and keeps all ops on the correct device.
        xp = cp if (USE_CUPY and isinstance(poses, cp.ndarray)) else np
        dw = float(self.cfg.get("ba_depth_weight", 0.5))

        # Upload inputs to device (no-op if already CuPy - V26)
        poses_xp   = xp.asarray(poses,   dtype=xp.float64)
        pts3d_xp   = xp.asarray(pts3d,   dtype=xp.float64)
        cam_idx_xp = xp.asarray(cam_idx, dtype=xp.int32)
        pt_idx_xp  = xp.asarray(pt_idx,  dtype=xp.int32)
        u_obs_xp   = xp.asarray(u_obs,   dtype=xp.float64)
        v_obs_xp   = xp.asarray(v_obs,   dtype=xp.float64)
        z_obs_xp   = xp.asarray(z_obs,   dtype=xp.float64)

        # ── Gather transforms ──────────────────────────────────────────────
        T_wc_batch = poses_xp[cam_idx_xp]              # (M, 4, 4)
        R_wc = T_wc_batch[:, :3, :3]                   # (M, 3, 3)
        t_wc = T_wc_batch[:, :3, 3]                    # (M, 3)
        R_cw = R_wc.transpose(0, 2, 1)                 # (M, 3, 3)
        t_cw = -xp.matmul(R_cw, t_wc[:, :, None])[:, :, 0]  # (M, 3)

        # ── Landmark to camera frame ───────────────────────────────────────
        P_w  = pts3d_xp[pt_idx_xp]                     # (M, 3)
        P_c  = xp.matmul(R_cw, P_w[:, :, None])[:, :, 0] + t_cw  # (M, 3)

        z      = P_c[:, 2]
        safe   = z > 0.05
        z_s    = xp.where(safe, z, xp.ones_like(z))
        inv_z  = xp.where(safe, 1.0 / z_s, xp.zeros_like(z))
        inv_z2 = inv_z * inv_z

        # ── Reprojection ───────────────────────────────────────────────────
        u_proj = self.fx * P_c[:, 0] * inv_z + self.cx
        v_proj = self.fy * P_c[:, 1] * inv_z + self.cy
        ru  = u_proj - u_obs_xp
        rv  = v_proj - v_obs_xp
        err = xp.sqrt(xp.maximum(ru*ru + rv*rv, 1e-20))

        # ── Huber weights (reprojection) ───────────────────────────────────
        obs_valid = safe & (err < self.max_reproj)
        w_reproj  = xp.where(obs_valid,
                        xp.where(err <= self.huber_c,
                                 xp.ones_like(err),
                                 self.huber_c / xp.maximum(err, 1e-10)),
                        xp.zeros_like(err))

        # ── Projection Jacobian ∂(u,v)/∂P_c  (M,2,3) ─────────────────────
        Jp_proj = xp.zeros((M, 2, 3), dtype=xp.float64)
        Jp_proj[:, 0, 0] =  self.fx * inv_z
        Jp_proj[:, 0, 2] = -self.fx * P_c[:, 0] * inv_z2
        Jp_proj[:, 1, 1] =  self.fy * inv_z
        Jp_proj[:, 1, 2] = -self.fy * P_c[:, 1] * inv_z2

        # ── P_c Jacobian wrt right-perturbation ξ on T_wc  (M,3,6) ───────
        #   Translation block (cols 0-2): ∂P_c/∂δρ = -I
        #   Rotation block    (cols 3-5): ∂P_c/∂δω = -[P_c]×
        Jc_Pc = xp.zeros((M, 3, 6), dtype=xp.float64)
        Jc_Pc[:, 0, 0] = -1.0
        Jc_Pc[:, 1, 1] = -1.0
        Jc_Pc[:, 2, 2] = -1.0
        Jc_Pc[:, 0, 4] =  P_c[:, 2];  Jc_Pc[:, 0, 5] = -P_c[:, 1]
        Jc_Pc[:, 1, 3] = -P_c[:, 2];  Jc_Pc[:, 1, 5] =  P_c[:, 0]
        Jc_Pc[:, 2, 3] =  P_c[:, 1];  Jc_Pc[:, 2, 4] = -P_c[:, 0]

        # ── Full Jacobians (chain rule) ────────────────────────────────────
        Jc_full = xp.matmul(Jp_proj, Jc_Pc)    # (M,2,3)@(M,3,6) -> (M,2,6)
        Jp_full = xp.matmul(Jp_proj, R_cw)     # (M,2,3)@(M,3,3) -> (M,2,3)

        # ── V28: Depth residual (channel 2) ───────────────────────────────
        # r_z = P_c[:,2] - z_obs
        # J_c_z = Jc_Pc[row 2] = [0, 0, -1, P_c[y], -P_c[x], 0]
        # J_p_z = R_cw[row 2, :]
        #
        # V33 FIX: Use self.huber_c_depth (metres) instead of self.huber_c (pixels)
        # for the depth Huber threshold.  ba_huber_thresh=2.0 px is far too large
        # for depth residuals in metres (typical outlier threshold = 0.12 m).
        # Previously every depth residual |r_z| < 2.0 m was treated as an inlier
        # with weight 1.0, giving no robustness against bad stereo pixels or
        # occlusion-boundary depth errors.  Using 0.12 m correctly down-weights
        # depth observations with |r_z| > 12 cm.
        has_depth = z_obs_xp > 0.1
        rz        = xp.where(has_depth, P_c[:, 2] - z_obs_xp, xp.zeros_like(z))
        err_z     = xp.abs(rz)
        w_depth   = xp.where(
            has_depth & obs_valid,
            dw * xp.where(err_z <= self.huber_c_depth,
                          xp.ones_like(err_z),
                          self.huber_c_depth / xp.maximum(err_z, 1e-10)),
            xp.zeros_like(err_z))

        Jc_z = Jc_Pc[:, 2, :]      # (M, 6)  depth cam Jacobian = Jc_Pc row 2
        Jp_z = R_cw[:, 2, :]        # (M, 3)  depth pt  Jacobian = R_cw row 2

        # ── Interleave into 3M-row arrays ─────────────────────────────────
        r_vec       = xp.zeros(3 * M, dtype=xp.float64)
        J_cam_block = xp.zeros((3 * M, 6), dtype=xp.float64)
        J_pts_block = xp.zeros((3 * M, 3), dtype=xp.float64)
        weights     = xp.zeros(3 * M, dtype=xp.float64)
        valid_mask  = xp.zeros(3 * M, dtype=bool)

        ri0 = xp.arange(M) * 3       # u-rows
        ri1 = ri0 + 1                 # v-rows
        ri2 = ri0 + 2                 # z-rows

        r_vec[ri0] = ru;  r_vec[ri1] = rv;  r_vec[ri2] = rz
        J_cam_block[ri0] = Jc_full[:, 0, :]
        J_cam_block[ri1] = Jc_full[:, 1, :]
        J_cam_block[ri2] = Jc_z
        J_pts_block[ri0] = Jp_full[:, 0, :]
        J_pts_block[ri1] = Jp_full[:, 1, :]
        J_pts_block[ri2] = Jp_z
        weights[ri0] = w_reproj;  weights[ri1] = w_reproj;  weights[ri2] = w_depth
        valid_mask[ri0] = obs_valid
        valid_mask[ri1] = obs_valid
        # V29 CRITICAL FIX: ri2 MUST use the same obs_valid as ri0/ri1.
        # V28 had: valid_mask[ri2] = obs_valid & has_depth & (dw > 0.0)
        # That excluded depth rows for obs without stereo depth, making
        # the total number of valid rows NOT divisible by N_RES=3 whenever
        # any observation lacked depth.  _schur_solve then does:
        #   M = MN // N_RES          -> truncated (wrong M)
        #   Jc_obs = Jc_xp.reshape(M, N_RES, 6)  -> ValueError crash
        # Fix: always keep all 3 rows of each obs together in the mask.
        # The depth channel weight w_depth is already 0 when has_depth=False
        # or dw=0 (see w_depth = xp.where(has_depth & obs_valid, ...)),
        # so depth-less rows contribute exactly 0 to H, g, and cost.
        valid_mask[ri2] = obs_valid

        # Return on-device; no forced D2H.
        return r_vec, J_cam_block, J_pts_block, weights, valid_mask

    # ------------------------------------------------------------------
    def _schur_solve(self, r, Jc, Jp, W, cam_idx_per_res, pt_idx_per_res,
                     N_cam, N_pts, lam, N_RES=3):
        """
        Schur complement solve - fully vectorized (V13), GPU-clean (V23).

        V28 CHANGES:
          • N_RES parameter (default 3): residuals per observation.
            With N_RES=3 (u,v,z) the per-row weight W allows different weights
            per channel (depth weight ≠ reprojection weight).
          • W_obs shape (M, N_RES): per-row weights used for weighted J products.
            H_cc_blk = J_c^T diag(W_row) J_c via broadcast:
              wJc  = W_obs[:,:,None] * Jc_obs       (M, N_RES, 6)
              H_cc = Jc_obs_T @ wJc = (M,6,N_RES)@(M,N_RES,6) -> (M,6,6)
          • ci_obs / pi_obs extracted via [::N_RES][:M].
          • Jc_obs / Jp_obs / r_obs reshaped to (M, N_RES, *).

        V23 CLARIFICATION/FIX - gradient vector shape:
          gc_blk = J_c^T @ diag(W_row) r: (M,6,N_RES) @ (M,N_RES,1) -> (M,6)
          gp_blk = J_p^T @ diag(W_row) r: (M,3,N_RES) @ (M,N_RES,1) -> (M,3)

        V15 FIXES (preserved):
          • Back-substitution sign CORRECTED (CRITICAL):
            ΔP = -H_pp^{-1}(g_p + H_cp^T Δξ)  - V14 had wrong sign
          • All ops stay on CuPy device - no forced D2H.

        Inputs are already filtered to valid residuals only.
        cam_idx_per_res / pt_idx_per_res have one entry per residual row.
        """
        MN = len(r)
        # V30 FIX: guard against non-divisible MN before reshape.
        # Caused by any upstream valid_mask bug that breaks the N_RES grouping.
        # Truncate to largest multiple of N_RES that is safe.
        if MN % N_RES != 0:
            MN = (MN // N_RES) * N_RES
            r  = r[:MN];  Jc = Jc[:MN];  Jp = Jp[:MN]
            W  = W[:MN]
            cam_idx_per_res = cam_idx_per_res[:MN]
            pt_idx_per_res  = pt_idx_per_res[:MN]
        M  = MN // N_RES
        if M == 0:
            return None, None

        # V30 FIX: select xp from actual input type, not USE_CUPY flag.
        # Handles hybrid calls (e.g. CPU fallback with USE_CUPY=True).
        if USE_CUPY and isinstance(r, cp.ndarray):
            xp = cp
        else:
            xp = np

        # Upload inputs once (no-op if already xp arrays - V26)
        r_xp  = xp.asarray(r,               dtype=xp.float64)  # (N_RES*M,)
        Jc_xp = xp.asarray(Jc,              dtype=xp.float64)  # (N_RES*M,6)
        Jp_xp = xp.asarray(Jp,              dtype=xp.float64)  # (N_RES*M,3)
        W_xp  = xp.asarray(W,               dtype=xp.float64)  # (N_RES*M,)
        ci_xp = xp.asarray(cam_idx_per_res, dtype=xp.int32)    # (N_RES*M,)
        pi_xp = xp.asarray(pt_idx_per_res,  dtype=xp.int32)    # (N_RES*M,)

        # ── Per-observation data ───────────────────────────────────────────
        ci_obs = ci_xp[::N_RES][:M]            # (M,) cam index per obs
        pi_obs = pi_xp[::N_RES][:M]            # (M,) pt  index per obs

        Jc_obs = Jc_xp.reshape(M, N_RES, 6)   # (M, N_RES, 6)
        Jp_obs = Jp_xp.reshape(M, N_RES, 3)   # (M, N_RES, 3)
        r_obs  = r_xp.reshape(M, N_RES)        # (M, N_RES)
        W_obs  = W_xp.reshape(M, N_RES)        # (M, N_RES) - per-row weights

        # ── Weighted Jacobian products (on device) ────────────────────────
        # V28: broadcast per-row weight (M,N_RES,1) onto each Jacobian row
        wJc = W_obs[:, :, None] * Jc_obs       # (M, N_RES, 6)
        wJp = W_obs[:, :, None] * Jp_obs       # (M, N_RES, 3)

        Jc_obs_T = Jc_obs.transpose(0, 2, 1)  # (M, 6, N_RES)
        Jp_obs_T = Jp_obs.transpose(0, 2, 1)  # (M, 3, N_RES)

        # H blocks: (M,6,N_RES)@(M,N_RES,6) -> (M,6,6)
        Hcc_blk = xp.matmul(Jc_obs_T, wJc)    # (M,6,6)
        Hpp_blk = xp.matmul(Jp_obs_T, wJp)    # (M,3,3)
        Hcp_blk = xp.matmul(Jc_obs_T, wJp)    # (M,6,3)

        # Gradient: J^T @ (W*r).  (M,N_RES,1) weight-scaled residual.
        wr_vec  = (W_obs * r_obs)[:, :, None]              # (M, N_RES, 1)
        gc_blk  = xp.matmul(Jc_obs_T, wr_vec)[:, :, 0]   # (M, 6)
        gp_blk  = xp.matmul(Jp_obs_T, wr_vec)[:, :, 0]   # (M, 3)

        # ── Scatter-accumulate on device ──────────────────────────────────
        # CuPy supports fancy-index scatter via += but needs 1D index scatter.
        # We use a helper that works for both CuPy and NumPy.
        H_cc        = xp.zeros((6*N_cam, 6*N_cam), dtype=xp.float64)
        g_c         = xp.zeros(6*N_cam,             dtype=xp.float64)
        H_pp_blocks = xp.zeros((N_pts,  3, 3),      dtype=xp.float64)
        g_p         = xp.zeros(3*N_pts,             dtype=xp.float64)
        H_cp        = xp.zeros((6*N_cam, 3*N_pts),  dtype=xp.float64)

        # V24: use int64 throughout for all flat scatter indices.
        # int32 would overflow for large N_cam or N_pts (e.g. N_cam=200 gives
        # max flat index = 200*6*200*6 = 1.44e6 - within int32, but int64 is
        # safer and CuPy scatter_add requires int64 for large arrays).
        idx6 = xp.arange(6, dtype=xp.int64)
        idx3 = xp.arange(3, dtype=xp.int64)

        # V24: pre-allocate flat views (reshape(-1) guaranteed-view on C-order arrays)
        H_cc_flat = H_cc.reshape(-1)
        H_pp_flat = H_pp_blocks.reshape(-1)
        H_cp_flat = H_cp.reshape(-1)

        # H_cc scatter: (M,6,6) blocks into (6N_cam, 6N_cam)
        # row = ci*6+di,  col = ci*6+dj
        row_cc = (ci_obs.astype(xp.int64)*6)[:,None,None] + idx6[None,:,None]   # (M,6,1)
        col_cc = (ci_obs.astype(xp.int64)*6)[:,None,None] + idx6[None,None,:]   # (M,1,6)
        flat_cc = (row_cc * (6*N_cam) + col_cc).reshape(-1)                      # (M*36,) int64
        if xp is cp:
            _cp_scatter_add(H_cc_flat, flat_cc, Hcc_blk.reshape(-1))
        else:
            np.add.at(H_cc_flat, flat_cc, Hcc_blk.reshape(-1))

        # g_c scatter: (M,6) into (6N_cam,)
        row_gc  = (ci_obs.astype(xp.int64)*6)[:,None] + idx6[None,:]             # (M,6)
        flat_gc = row_gc.reshape(-1)                                              # (M*6,)
        if xp is cp:
            _cp_scatter_add(g_c, flat_gc, gc_blk.reshape(-1))
        else:
            np.add.at(g_c, flat_gc, gc_blk.reshape(-1))

        # H_pp_blocks scatter: (M,3,3) into (N_pts,3,3) via obs index
        Hpp_flat_blk = Hpp_blk.reshape(M, 9)                       # (M,9)
        pi_obs_rep   = xp.repeat(pi_obs.astype(xp.int64), 9)       # (M*9,) - base pt idx repeated
        off9         = xp.tile(xp.arange(9, dtype=xp.int64), M)    # (M*9,) - offset 0..8
        flat_hpp     = pi_obs_rep * 9 + off9                        # (M*9,) int64
        if xp is cp:
            _cp_scatter_add(H_pp_flat, flat_hpp, Hpp_flat_blk.reshape(-1))
        else:
            np.add.at(H_pp_flat, flat_hpp, Hpp_flat_blk.reshape(-1))

        # g_p scatter: (M,3) into (3N_pts,)
        row_gp  = (pi_obs.astype(xp.int64)*3)[:,None] + idx3[None,:]             # (M,3)
        flat_gp = row_gp.reshape(-1)                                              # (M*3,)
        if xp is cp:
            _cp_scatter_add(g_p, flat_gp, gp_blk.reshape(-1))
        else:
            np.add.at(g_p, flat_gp, gp_blk.reshape(-1))

        # H_cp scatter: (M,6,3) into (6N_cam, 3N_pts)
        # row = ci*6+di,  col = pi*3+dj
        row_cp = (ci_obs.astype(xp.int64)*6)[:,None,None] + idx6[None,:,None]   # (M,6,1)
        col_cp = (pi_obs.astype(xp.int64)*3)[:,None,None] + idx3[None,None,:]   # (M,1,3)
        flat_cp = (row_cp * (3*N_pts) + col_cp).reshape(-1)                      # (M*18,) int64
        if xp is cp:
            _cp_scatter_add(H_cp_flat, flat_cp, Hcp_blk.reshape(-1))
        else:
            np.add.at(H_cp_flat, flat_cp, Hcp_blk.reshape(-1))

        # ── H_pp block inversion - batch, no loop ─────────────────────────
        if xp is cp:
            H_pp_reg = H_pp_blocks + lam * cp.eye(3, dtype=cp.float64)[None]
            H_pp_inv = _batch_inv3_gpu(H_pp_reg)                # (N_pts,3,3) GPU
        else:
            H_pp_reg = H_pp_blocks + lam * np.eye(3)[None]
            H_pp_inv = _batch_inv3(H_pp_reg)                    # (N_pts,3,3) CPU

        # ── Schur complement - vectorized via batched matmul ──────────────
        # V22: replaced einsum('knj,njm->knm') with batched matmul on the
        # N_pts axis.  cuBLAS batched GEMM dispatches through the (N_pts,
        # 6*N_cam, 3) × (N_pts, 3, 3) path, giving ~3-5× throughput vs
        # the element-wise einsum for large N_pts.
        #
        # V30 FIX: verify reshape semantics:
        #   H_cp has shape (6*N_cam, 3*N_pts).
        #   reshape(6*N_cam, N_pts, 3) treats columns as (N_pts, 3) blocks.
        #   Transpose to (N_pts, 6*N_cam, 3): each of N_pts landmark blocks
        #   becomes a (6*N_cam, 3) slice -> matmul with H_pp_inv[n] (3,3).
        #
        # V31 FIX: force H_cp to be C-contiguous before reshape.
        # After scatter_add, H_cp is always C-contiguous (allocated via
        # xp.zeros), but adding ascontiguousarray here guards against any
        # future upstream change that produces a non-contiguous H_cp
        # (e.g. a slice or transpose view).  reshape() on a non-contiguous
        # CuPy array silently copies data in some CuPy versions, meaning
        # H_cp_3d and H_cp_flat would no longer share memory - the matmul
        # would operate on stale data with no error.
        H_cp_contig = xp.ascontiguousarray(H_cp)               # no-op if already C-order
        H_cp_3d      = H_cp_contig.reshape(6*N_cam, N_pts, 3)
        HcpHppinv_3d = xp.matmul(
            H_cp_3d.transpose(1, 0, 2),   # (N_pts, 6*N_cam, 3)
            H_pp_inv                       # (N_pts, 3, 3)
        ).transpose(1, 0, 2)               # (6*N_cam, N_pts, 3)
        HcpHppinv    = xp.ascontiguousarray(HcpHppinv_3d.reshape(6*N_cam, 3*N_pts))

        # ── Add LM damping to camera block diagonal ──────────────────────
        diag_idx = xp.arange(6*N_cam)
        H_cc[diag_idx, diag_idx] += lam

        # ── V22 GAUGE FIX: pin first camera with FIXED large constant ────────
        # Without this the BA window has 6 DOF of gauge freedom (global
        # rigid-body motion of the entire window leaves all reprojection
        # errors unchanged).  This makes the Schur-complement matrix S
        # singular or poorly-conditioned.
        #
        # V21 used `lam * 1e6` as the gauge weight.  This is CRITICALLY WRONG:
        #   lam_init = 1e-3  -> gauge weight = 1e3
        #   lam after 3 accepted steps (~0.1³ × 1e-3) = 1e-6 -> weight = 1.0
        #   normal-equation diagonal ≈ fx² × M ≈ 400² × 100 = 1.6e7
        # So at convergence the gauge weight (1.0) is 7 orders of magnitude
        # below H_cc's diagonal - effectively allowing full 6-DOF gauge drift.
        # This caused the window to freely translate/rotate, corrupt all camera
        # and landmark updates, and appear numerically converged while producing
        # completely wrong corrections.
        #
        # Fix: use a FIXED large constant (1e6).  This is always ≫ H_cc diag
        # for typical problems, correctly anchoring the first camera regardless
        # of the LM damping schedule.
        H_cc[:6, :6] += xp.eye(6, dtype=xp.float64) * 1e6

        # V24 FIX: Pre-compute contiguous H_pc = H_cp^T once and reuse.
        # H_cp.T is a non-contiguous (Fortran-order) view in CuPy.  cuBLAS can
        # handle non-contiguous arrays via transposition flags, but materialising
        # the contiguous copy here avoids any potential slowdown and ensures the
        # two uses of H_pc below (S and g_p back-sub) use identical data.
        H_pc = xp.ascontiguousarray(H_cp.T)      # (3*N_pts, 6*N_cam) - contiguous

        S   = H_cc - HcpHppinv @ H_pc            # (6N_cam, 6N_cam) Schur matrix
        rhs = g_c  - HcpHppinv @ g_p             # (6N_cam,)

        # V35 FIX: Enforce numerical symmetry of S before solve.
        # Floating-point accumulation during scatter_add and the three matrix
        # products (HcpHppinv @ H_pc) can introduce asymmetry at the ~1e-14
        # level.  cp.linalg.solve / np.linalg.solve tolerates this, but Cholesky-
        # based solvers would fail.  Symmetrizing costs 1 transpose + 1 element-
        # wise add + 1 scalar multiply: O((6N_cam)²) - negligible vs the GEMM.
        S = xp.ascontiguousarray(0.5 * (S + S.T))

        # V24 FIX: Add small Tikhonov regularization on the Schur matrix S.
        # The gauge fix (1e6 on H_cc[:6,:6]) prevents pure gauge drift, but S
        # can still be numerically ill-conditioned for nearly-degenerate camera
        # configurations (collinear cameras, narrow baseline, or very few tracks).
        # Adding 1e-6 * I ensures the smallest eigenvalue of S is >= 1e-6,
        # which keeps cp.linalg.solve well-conditioned at negligible accuracy cost
        # (camera increments are typically 0.01-1.0 pixels - 1e-6 noise is imperceptible).
        S_diag = xp.arange(6 * N_cam, dtype=xp.int64)
        S[S_diag, S_diag] += 1e-6

        # ── Solve for camera increments ────────────────────────────────────
        # V17: dx_cam stays on-device (CuPy) when running the GPU path.
        # Eliminates the D2H(6N_cam) + H2D(6N_cam) round-trip that V16 paid:
        #   cp.asnumpy(solve(...))  then  cp.asarray(dx_cam) in _lm_optimize.
        # _batch_exp_se3_gpu and _batch_matmul_gpu both accept CuPy - the
        # D2H only occurs once at _lm_optimize return (cp.asnumpy(poses)).
        #
        # V33 FIX: For small S (6*N_cam <= 72, i.e. N_cam <= 12), solve on CPU.
        # cp.linalg.solve for a 48×48 matrix incurs ~0.4 ms kernel-launch
        # overhead - 10-40× slower than np.linalg.solve (~0.01 ms).  D2H for
        # a (72,72) float64 array is 72²×8 = 41 KB, well below PCIe bandwidth.
        # The returned dx_cam is immediately re-uploaded as cp.asarray() in
        # _lm_optimize (no-op if already CuPy), so the net PCIe cost is minimal.
        _S_size = 6 * N_cam
        if xp is cp and _S_size <= 72:
            # Small S: solve on CPU, avoids GPU kernel-launch overhead
            try:
                S_cpu   = cp.asnumpy(S)
                rhs_cpu = cp.asnumpy(rhs)
                dx_cam  = cp.asarray(np.linalg.solve(S_cpu, -rhs_cpu),
                                     dtype=cp.float64)
            except np.linalg.LinAlgError:
                return None, None
        else:
            try:
                dx_cam = xp.linalg.solve(S, -rhs)   # CuPy on GPU path, NumPy on CPU path
            except Exception:
                return None, None

        # V24 FIX: NaN/Inf guard on dx_cam.
        # _batch_inv3_gpu clamps det to ±1e-12 so degenerate H_pp blocks produce
        # large-but-finite inverses; however extreme cases can still propagate
        # NaN through the Schur complement.  Detect and bail before the update.
        if xp is cp:
            if not bool(cp.all(cp.isfinite(dx_cam))):
                return None, None
        else:
            if not np.all(np.isfinite(dx_cam)):
                return None, None

        # ── Back-substitute for landmark increments - vectorized ──────────
        # From normal equations row 2:  H_pc Δξ + H_pp ΔP = -g_p
        #   -> ΔP = -H_pp^{-1} (g_p + H_pc Δξ)
        # NOTE: sign is PLUS inside the parens (not minus).  V14 had
        # g_p - H_cp.T @ dx_cam (sign error) - corrected in V15.
        # V25 FIX (CRITICAL): H_pc is (3*N_pts, 6*N_cam); dx_cam is (6*N_cam,).
        # Correct matmul: H_pc @ dx_cam -> (3*N_pts,).
        # V24 had H_pc.T @ dx_cam = (6*N_cam, 3*N_pts) @ (6*N_cam,) which is a
        # shape MISMATCH - raises LinAlgError/ValueError at runtime whenever
        # 3*N_pts ≠ 6*N_cam.  The variable H_pc = ascontiguousarray(H_cp.T)
        # is (3*N_pts, 6*N_cam); using it directly (not transposed) is correct.
        # Both dx_cam and dx_pts are returned as xp arrays (CuPy or NumPy),
        # keeping all arithmetic on-device until _lm_optimize's final D2H.
        g_p_corr  = g_p + H_pc @ dx_cam              # (3*N_pts,)  - V25 FIX: was H_pc.T (shape error)
        g_p_3d    = g_p_corr.reshape(N_pts, 3)
        # V22: matmul matvec (N_pts,3,3) @ (N_pts,3,1) -> (N_pts,3,1) -> (N_pts,3)
        # uses cuBLAS batched GEMV - faster than einsum('nij,nj->ni') for large N_pts
        dx_pts_3d = -xp.matmul(H_pp_inv, g_p_3d[:, :, None])[:, :, 0]
        dx_pts    = dx_pts_3d.reshape(-1)             # (N_pts*3,)  - stays on-device

        # V24 FIX: NaN/Inf guard on dx_pts (degenerate H_pp blocks).
        if xp is cp:
            if not bool(cp.all(cp.isfinite(dx_pts))):
                return None, None
        else:
            if not np.all(np.isfinite(dx_pts)):
                return None, None

        return dx_cam, dx_pts
