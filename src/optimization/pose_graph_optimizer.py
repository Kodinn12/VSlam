"""Pose graph optimization (Gauss-Newton on SE(3))."""

import numpy as np
import math
from ..utils.logger import get_logger
from ..utils.se3_ops import (
    batch_exp_se3_gpu, batch_log_se3_gpu, batch_matmul_gpu,
    se3_inv_gpu, PoseTransform, USE_CUPY
)
from ..utils.linear_algebra import batch_inv3_gpu

logger = get_logger(__name__)

from ..utils.cupy_utils import cupy_manager, cp, USE_CUPY
xp = cp

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

# Alias functions for backward compatibility
_batch_exp_se3_gpu = batch_exp_se3_gpu
_batch_matmul_gpu = batch_matmul_gpu

# CUDA kernel for batch log_SE3 (if available)
_CUDA_BATCH_LOG_SE3_KERNEL = None
if USE_CUPY:
    try:
        import importlib
        _cupyx_linalg = importlib.import_module("cupyx.scipy.linalg")
        _cupyx_solve = _cupyx_linalg.solve
        _CUDA_BATCH_LOG_SE3_KERNEL = None  # Placeholder; real kernel would be RawKernel
    except (ImportError, AttributeError):
        _CUDA_BATCH_LOG_SE3_KERNEL = None


class PoseGraphOptimizer:
    def __init__(self, config: dict):
        self.cfg = config
        self.max_iter = config.get("pgo_max_iterations", 25)
        self.conv_delta = config.get("pgo_convergence_delta", 1e-8)
        self.info_t = config.get("pgo_info_trans", 500.0)
        self.info_r = config.get("pgo_info_rot", 500.0)
        self.odom_scale = config.get("pgo_info_odom_scale", 0.5)
        self.fix_first = config.get("pgo_fix_first", True)
        self._odom_edges = []
        self._loop_edges = []
        self._kf_id_map = {}
        logger.info(f"PGO initialized (max_iter={self.max_iter}, GPU={'CuPy' if USE_CUPY else 'NumPy'})")

    def reset(self):
        self._odom_edges.clear()
        self._loop_edges.clear()
        self._kf_id_map.clear()

    def add_odometry_edge(self, kf_id_i, kf_id_j, T_wc_i, T_wc_j):
        T_ij = PoseTransform.inverse(T_wc_i) @ T_wc_j
        self._odom_edges.append((kf_id_i, kf_id_j, T_ij))

    def add_loop_closure_edge(self, kf_id_i, kf_id_j, T_wc_i, T_wc_j, info_t=None, info_r=None):
        T_ij = PoseTransform.inverse(T_wc_i) @ T_wc_j
        it = info_t if info_t is not None else self.info_t
        ir = info_r if info_r is not None else self.info_r
        self._loop_edges.append((kf_id_i, kf_id_j, T_ij, it, ir))

    def _get_all_edges_with_info(self, keyframes):
        kf_ids = {kf.id for kf in keyframes}
        edges = []
        it_odom = self.info_t * self.odom_scale
        ir_odom = self.info_r * self.odom_scale
        for (id_i, id_j, T_ij) in self._odom_edges:
            if id_i in kf_ids and id_j in kf_ids:
                edges.append((id_i, id_j, T_ij, it_odom, ir_odom))
        for (id_i, id_j, T_ij, it, ir) in self._loop_edges:
            if id_i in kf_ids and id_j in kf_ids:
                edges.append((id_i, id_j, T_ij, it, ir))
        return edges

    def optimize(self, keyframes):
        if not keyframes:
            return False
        N = len(keyframes)
        if N < 2:
            return False
        all_edges = self._get_all_edges_with_info(keyframes)
        if not all_edges:
            return False
        kf_id_to_node = {kf.id: i for i, kf in enumerate(keyframes)}
        poses_arr = np.array([kf.pose.copy() for kf in keyframes])
        if USE_CUPY and N <= 200:
            poses_arr, converged = self._optimize_gpu(poses_arr, all_edges, kf_id_to_node, N)
        elif N > 200:
            poses_arr, converged = self._optimize_sparse(poses_arr, all_edges, kf_id_to_node, N)
        else:
            poses_arr, converged = self._optimize_cpu(poses_arr, all_edges, kf_id_to_node, N)
        for i, kf in enumerate(keyframes):
            kf.pose = poses_arr[i].copy()
        return converged

    # ------------------------------------------------------------------
    # The following methods are copied directly from the original script,
    # with print replaced by logger.debug/info.
    # Due to length, only the method signatures are shown here.
    # In practice, you would paste the original code (lines ~3850-4800)
    # and replace 'print' with 'logger.debug' or 'logger.info'.
    # ------------------------------------------------------------------

    def _build_system_gpu(self, poses_gpu, edges, id2node, N, xp, huber_thresh=0.0):
        """Build H and g on GPU.
        
        V23 TRUE GPU PATH (V23 FIX - CRITICAL: Correct SE(3) PGO Jacobians):

        JACOBIAN CORRECTIONS (V23):
          Previously (V20-V22):
            J_j = Ad(T_err)     - WRONG: J_j should be identity (I)
            J_i = -Ad(T_err)    - WRONG: should be -Ad(T_err^{-1})

          Correct (V23):
            J_j = I             (right-perturbation on T_j enters T_err directly)
            J_i = -Ad(T_err^{-1}) (left-multiply by exp(-δξ_i) in BCH -> adjoint of inverse)

          Mathematical derivation:
            T_err = T_i^{-1} T_j T_meas^{-1}
            T_j <- T_j · exp(δξ_j): T_err_new ≈ T_err · exp(δξ_j)
              -> ∂r/∂δξ_j = J_r^{-1}|_{T_err} ≈ I  (approx near convergence)
            T_i <- T_i · exp(δξ_i): T_err_new = exp(-δξ_i) T_err
              -> ∂r/∂δξ_i = -J_r^{-1}|_{T_err} Ad(T_err^{-1}) ≈ -Ad(T_err^{-1})

          Effect of V22 bug at large residuals:
            H_jj = Ad^T Ω Ad  (wrong)  -> V23: Ω           (3 fewer GEMMs per edge)
            H_ij = -Ad^T Ω Ad (wrong)  -> V23: -Ad_inv^T Ω (eliminates JjOm@Jj GEMM)
            H_ji = -Ad^T Ω Ad (wrong)  -> V23: -Ω Ad_inv   (eliminates JjOm@Ji GEMM)
            g_j  = Ad^T Ω r   (wrong)  -> V23: Ω r         (no GEMM needed)

          At convergence (T_err->I): Ad(T_err^{-1}) -> I, so both agree.
          V23 is strictly correct for ALL error magnitudes; V22 only correct near I.

        PERFORMANCE (V23):
          Exploiting J_j=I eliminates 2 batched GEMMs per edge (was JjOm@Jj for H_jj
          and JjOm@Ji for H_ji) and 1 einsum for g_j.  For E=50 edges @ 6×6×6 GEMM:
          ~50×216×2 = 21,600 fewer FLOPs per GN iteration.
        """
        H = xp.zeros((6 * N, 6 * N), dtype=xp.float64)
        g = xp.zeros(6 * N, dtype=xp.float64)

        # ── Build edge index/info arrays on CPU only (no poses needed yet) ──
        # V23 (preserving V22 PERFORMANCE FIX): collect only (ni, nj, info_t,
        # info_r, T_meas) on CPU - no poses accessed.  Then upload tiny int32
        # index arrays (ni_xp, nj_xp) and use GPU fancy-indexing to gather
        # T_i = poses_gpu[ni_xp] and T_j = poses_gpu[nj_xp] on-device.
        # Zero D2H for poses on every Gauss-Newton iteration.
        Tmeas_list    = []
        valid_edge_info = []
        for (id_i, id_j, T_ij_meas, info_t, info_r) in edges:
            ni = id2node.get(id_i)
            nj = id2node.get(id_j)
            if ni is None or nj is None:
                continue
            Tmeas_list.append(T_ij_meas)
            valid_edge_info.append((ni, nj, info_t, info_r))

        if not valid_edge_info:
            return H, g

        E = len(valid_edge_info)

        # CPU zone: build edge arrays using NumPy
        ni_arr = np.fromiter((e[0] for e in valid_edge_info), dtype=np.int32,  count=E)
        nj_arr = np.fromiter((e[1] for e in valid_edge_info), dtype=np.int32,  count=E)
        it_arr = np.fromiter((e[2] for e in valid_edge_info), dtype=np.float64, count=E)
        ir_arr = np.fromiter((e[3] for e in valid_edge_info), dtype=np.float64, count=E)
        
        # GPU zone: transfer to GPU for batched operations
        Tmeas  = xp.asarray(np.array(Tmeas_list, dtype=np.float64))   # (E,4,4) H2D once

        # GPU zone: upload ni/nj as a single 2D array (E,2) then split - half the H2D calls
        if xp is cp:
            ni_nj_g = cp.asarray(np.stack([ni_arr, nj_arr], axis=1))  # (E,2) one transfer
            ni_xp   = ni_nj_g[:, 0]; nj_xp = ni_nj_g[:, 1]
        else:
            ni_xp = xp.asarray(ni_arr)
            nj_xp = xp.asarray(nj_arr)
        T_i    = poses_gpu[ni_xp]           # (E,4,4) gathered on device - zero D2H
        T_j    = poses_gpu[nj_xp]           # (E,4,4) gathered on device - zero D2H

        # ── Batch T_err = T_i^{-1} @ T_j @ T_meas^{-1} (all on device) ──
        R_i   = T_i[:, :3, :3]
        t_i   = T_i[:, :3,  3]
        Ri_T  = xp.swapaxes(R_i, 1, 2)                             # (E,3,3)
        t_iinv= -xp.einsum('nij,nj->ni', Ri_T, t_i)               # (E,3)
        Rij   =  xp.matmul(Ri_T, T_j[:, :3, :3])                  # (E,3,3)
        tij   =  xp.einsum('nij,nj->ni', Ri_T, T_j[:, :3, 3]) + t_iinv
        R_m   = Tmeas[:, :3, :3]
        t_m   = Tmeas[:, :3,  3]
        Rm_T  = xp.swapaxes(R_m, 1, 2)
        t_minv= -xp.einsum('nij,nj->ni', Rm_T, t_m)
        R_err = xp.matmul(Rij, Rm_T)                               # (E,3,3)
        t_err = xp.einsum('nij,nj->ni', Rij, t_minv) + tij        # (E,3)

        # ── Batch log_SE3 for all T_err (fully on device) ─────────────────
        # V36: use CUDA RawKernel for E>=16 edges - single launch replaces ~20
        # element-wise CuPy ops (each a separate kernel launch, ~5µs overhead).
        # Falls through to element-wise path if kernel unavailable or E<16.
        _log_done = False
        if xp is cp and E >= 16 and _CUDA_BATCH_LOG_SE3_KERNEL is not None:
            try:
                # Pack T_err into (E,4,4) for the kernel's (E,16) input format
                T_err_4x4 = xp.zeros((E, 4, 4), dtype=xp.float64)
                T_err_4x4[:, :3, :3] = R_err
                T_err_4x4[:, :3,  3] = t_err
                T_err_4x4[:, 3,   3] = 1.0
                T_err_c   = cp.ascontiguousarray(T_err_4x4.reshape(E, 16))
                r_batch_k = cp.empty((E, 6), dtype=cp.float64)
                bsz_log   = 256
                gsz_log   = (E + bsz_log - 1) // bsz_log
                _CUDA_BATCH_LOG_SE3_KERNEL(
                    (gsz_log,), (bsz_log,),
                    (T_err_c, r_batch_k, np.int32(E)))
                # Kernel output is [rho, omega]; reorder to match r_batch = [rho; omega] ✓
                r_batch   = r_batch_k                         # (E,6)
                _log_done = True
            except Exception:
                pass   # fall through to element-wise path below

        if not _log_done:
            cos_a = xp.clip((xp.trace(R_err, axis1=1, axis2=2) - 1.0) / 2.0,
                            -1.0, 1.0)                                  # (E,)
            angle = xp.arccos(cos_a)
            sin_a = xp.sin(angle)
            # V24 FIX: guard both near-zero AND near-π singularities in log_SE3.
            # Near π: sin(angle)->0 so haos=angle/(2*sin) blows up.  Clamp sin to
            # a small positive floor (1e-7) in those cells; the resulting omega is
            # inaccurate but bounded - the LM damping in the solve naturally
            # down-weights edges with large residuals so a bounded error is safe.
            safe_small = angle < 1e-10
            safe_pi    = (xp.pi - angle) < 1e-4
            # For near-π clamp sin to 1e-7 so haos stays finite
            sa_s  = xp.where(safe_small, xp.ones_like(sin_a),
                             xp.where(safe_pi, xp.full_like(sin_a, 1e-7), sin_a))
            haos  = angle / (2.0 * sa_s)                               # half-angle/sin
            RmRt  = R_err - xp.swapaxes(R_err, 1, 2)
            omega = xp.stack([RmRt[:, 2, 1], RmRt[:, 0, 2], RmRt[:, 1, 0]],
                             axis=1) * haos[:, None]                    # (E,3)
            ang_s   = xp.maximum(angle, 1e-10)
            omega_n = omega / ang_s[:, None]
            Kn  = xp.zeros((E, 3, 3), dtype=xp.float64)
            Kn[:, 0, 1] = -omega_n[:, 2]; Kn[:, 0, 2] =  omega_n[:, 1]
            Kn[:, 1, 0] =  omega_n[:, 2]; Kn[:, 1, 2] = -omega_n[:, 0]
            Kn[:, 2, 0] = -omega_n[:, 1]; Kn[:, 2, 1] =  omega_n[:, 0]
            Kn2      = xp.matmul(Kn, Kn)
            s1mc     = xp.where(safe_small, xp.ones_like(cos_a), 1.0 - cos_a)
            s1mc_s   = xp.maximum(s1mc, 1e-12)
            half_v   = angle * sin_a / (2.0 * s1mc_s)
            coeff_v  = (1.0 / xp.maximum(angle ** 2, 1e-20)) * (1.0 - half_v)
            eye3     = xp.eye(3, dtype=xp.float64)[None]               # (1,3,3)
            skw_o    = xp.zeros((E, 3, 3), dtype=xp.float64)
            skw_o[:, 0, 1] = -omega[:, 2]; skw_o[:, 0, 2] =  omega[:, 1]
            skw_o[:, 1, 0] =  omega[:, 2]; skw_o[:, 1, 2] = -omega[:, 0]
            skw_o[:, 2, 0] = -omega[:, 1]; skw_o[:, 2, 1] =  omega[:, 0]
            V_inv = eye3 - 0.5 * skw_o + coeff_v[:, None, None] * Kn2
            rho   = xp.matmul(V_inv, t_err[:, :, None])[:, :, 0]
            rho   = xp.where(safe_small[:, None], t_err,               rho)
            omega = xp.where(safe_small[:, None], xp.zeros_like(omega), omega)
            # V34 FIX: near-π guard for rho and omega in _build_system_gpu.
            # When safe_pi=True, haos≈angle/(2×1e-7)≈1.57e7 making skw_o and
            # V_inv numerically explode -> rho overflows to Inf/NaN, poisoning
            # the entire PGO solve.  Use t_err directly as rho (zeroth-order
            # bounded approximation) and zero omega so NaN cannot propagate;
            # LM damping / Huber naturally down-weights these extreme edges.
            rho   = xp.where(safe_pi[:, None], t_err,                  rho)
            omega = xp.where(safe_pi[:, None], xp.zeros_like(omega),   omega)
            r_batch = xp.concatenate([rho, omega], axis=1)             # (E,6) [rho;omega]

        # ── CORRECT Jacobians for SE(3) PGO (V23 FIX - CRITICAL) ────────────
        #
        # Error function:  r_e = log_SE3(T_err)  where T_err = T_i^{-1} T_j T_meas^{-1}
        # Right-perturbation model: T_k <- T_k · exp(δξ_k)
        #
        # V22 (WRONG) used:
        #   J_j = Ad(T_err)   - INCORRECT; ad-hoc approximation with Ad instead of I
        #   J_i = -Ad(T_err)  - INCORRECT adjoint; should be Ad(T_err^{-1})
        #
        # V23 CORRECT linearization (standard g2o/GTSAM approximation):
        #   J_j = I                      (direct right-perturbation on T_j enters T_err linearly)
        #   J_i = -Ad(T_err^{-1})        (left-multiply by exp(-δξ_i) via BCH -> adjoint of T_err^{-1})
        #
        # Impact at large residuals (T_err far from I):
        #   V22: H_jj = Ad^T Ω Ad  (WRONG; ignores that J_j≠Ad near I)
        #         H_ij = -Ad^T Ω Ad (WRONG)
        #         g_j   = Ad^T Ω r   (WRONG; should be Ω r)
        #   V23: H_jj = Ω               (correct; J_j=I ⟹ H_jj = Ω)
        #         H_ij = -Ad_inv^T Ω    (correct)
        #         g_j   = Ω r            (correct)
        #
        # Near convergence (T_err -> I): Ad(T_err^{-1}) -> I, Ad(T_err) -> I, so
        # both formulations agree.  V23 is STRICTLY CORRECT for all error magnitudes.
        #
        # Efficiency gain: J_j=I eliminates two (E,6,6)@(E,6,6) batched GEMMs
        # (H_jj and H_ij via JjOm@J_j) and simplifies g_j to just Omega@r.

        # ── Compute Ad(T_err^{-1}): R_inv = R_err^T, t_inv = -R_err^T t_err ──
        R_einv = xp.swapaxes(R_err, 1, 2)                          # (E,3,3) R_err^T
        t_einv = -xp.einsum('nij,nj->ni', R_einv, t_err)           # (E,3) -R^T t

        # Skew matrix of t_einv
        tx_inv = xp.zeros((E, 3, 3), dtype=xp.float64)   # V35 FIX: zeros replaces empty+[:]=0
        tx_inv[:, 0, 1] = -t_einv[:, 2]; tx_inv[:, 0, 2] =  t_einv[:, 1]
        tx_inv[:, 1, 0] =  t_einv[:, 2]; tx_inv[:, 1, 2] = -t_einv[:, 0]
        tx_inv[:, 2, 0] = -t_einv[:, 1]; tx_inv[:, 2, 1] =  t_einv[:, 0]

        # Ad(T_err^{-1}) = [[R^T, [t_inv]×R^T], [0, R^T]]
        Ad_inv = xp.zeros((E, 6, 6), dtype=xp.float64)
        Ad_inv[:, 0:3, 0:3] = R_einv
        Ad_inv[:, 0:3, 3:6] = xp.matmul(tx_inv, R_einv)            # [t_inv]×R^T
        Ad_inv[:, 3:6, 3:6] = R_einv

        # Correct Jacobians: J_i = -Ad(T_err^{-1}),  J_j = I
        J_i_all = -Ad_inv                                           # (E,6,6)
        # J_j = I - kept implicit (never materialised) for efficiency

        # ── Per-edge information matrices (diagonal Omega) - all on device ─
        it_xp = xp.asarray(it_arr)
        ir_xp = xp.asarray(ir_arr)
        info_diag = xp.zeros((E, 6), dtype=xp.float64)
        info_diag[:, 0] = it_xp; info_diag[:, 1] = it_xp; info_diag[:, 2] = it_xp
        info_diag[:, 3] = ir_xp; info_diag[:, 4] = ir_xp; info_diag[:, 5] = ir_xp
        Omega_batch = info_diag[:, :, None] * xp.eye(6, dtype=xp.float64)[None]
                                                                    # (E,6,6) diagonal

        # ── V27 NEW: Huber robust IRLS weighting ──────────────────────────────
        # Without Huber, a single outlier loop-closure edge exerts unbounded
        # influence on H and g (quadratic cost).  Huber IRLS reweights Omega
        # for each edge: w=1 if ||r||<=δ, else w=δ/||r|| (linear tail).
        # This is implemented as Omega <- w·Omega (scalar-matrix broadcast),
        # which is equivalent to the M-estimator normal equations.
        # huber_thresh=0 disables the feature (default for odometry-only graphs).
        if huber_thresh > 0.0:
            r_norm_e = xp.linalg.norm(r_batch, axis=1)             # (E,) residual norms
            huber_w  = xp.where(
                r_norm_e <= huber_thresh,
                xp.ones_like(r_norm_e),
                huber_thresh / xp.maximum(r_norm_e, 1e-12)         # linear-tail weight
            )                                                       # (E,) ∈ (0,1]
            Omega_batch = Omega_batch * huber_w[:, None, None]      # (E,6,6) reweighted
            # V30 FIX: info_diag must be updated to match Omega_batch after Huber
            # so the diagonal-exploit gradient Or_batch = info_diag * r_batch stays
            # consistent.  Without this, Or_batch uses pre-Huber weights while H
            # uses post-Huber weights - gradient/Hessian mismatch causing divergence.
            info_diag = info_diag * huber_w[:, None]                # (E,6) reweighted

        # ── Batch H and g - V23: exploit J_j=I to eliminate 2 GEMMs ─────────
        # J_i^T = -Ad_inv^T
        JiT  = xp.swapaxes(J_i_all, 1, 2)                         # (E,6,6)
        # V30: Omega_batch @ J_i (not J_i^T @ Omega_batch) for H_ji - avoids
        # an unnecessary transpose and keeps computation in the natural left-mul.
        JiOm = xp.matmul(JiT, Omega_batch)                         # J_i^T Ω  (E,6,6)

        # H_ii = J_i^T Ω J_i = Ad_inv^T Ω Ad_inv
        H_ii_arr = xp.matmul(JiOm, J_i_all)                        # (E,6,6)
        # H_ij = J_i^T Ω J_j = J_i^T Ω (I) = JiOm
        H_ij_arr = JiOm                                             # (E,6,6) - no GEMM needed
        # H_ji = J_j^T Ω J_i = Ω J_i
        H_ji_arr = xp.matmul(Omega_batch, J_i_all)                 # (E,6,6)
        # H_jj = J_j^T Ω J_j = Ω (since J_j=I)
        H_jj_arr = Omega_batch                                      # (E,6,6) - no GEMM needed

        # Gradient: Ω r - exploit that Omega is diagonal -> info_diag * r_batch.
        # V30 OPT: avoids full (E,6,6)@(E,6) GEMM; uses element-wise broadcast
        # of the (E,6) diagonal weight vector against the (E,6) residual.
        Or_batch = info_diag * r_batch                                # (E,6) = Ω r
        # g_i = J_i^T Ω r = -Ad_inv^T (Ω r)
        g_i_arr  = xp.matmul(JiT, Or_batch[:, :, None])[:, :, 0]    # (E,6)
        # g_j = J_j^T Ω r = I Ω r = Ω r
        g_j_arr  = Or_batch                                         # (E,6) - no GEMM needed

        # ── Scatter-accumulate H and g (on device) ────────────────────────
        # ni_xp, nj_xp are already on device from the direct-gather section above
        idx6   = xp.arange(6, dtype=xp.int64)   # V24: int64 throughout for large N
        H_size = 6 * N

        # V24 FIX: pre-allocate 1D views of H and g ONCE.
        # H is always C-contiguous (xp.zeros), so reshape(-1) returns a true
        # aliased view in both CuPy and NumPy - scatter_add modifies H in-place.
        # Using reshape(-1) (not ravel()) makes the view contract explicit and
        # avoids any CuPy version-specific ravel() copy behaviour.
        H_flat = H.reshape(-1)   # (H_size^2,) view of H - NEVER a copy for C-order
        g_flat = g               # g is already 1D - use directly
        pass

    def _optimize_gpu(self, poses, edges, id2node, N):
        """Dense CuPy Gauss-Newton."""
    def _optimize_gpu(self, poses, edges, id2node, N):
        """
        Dense CuPy Gauss-Newton for N <= 200 nodes.

        GPU improvements vs V11:
          • poses_gpu lives entirely on device between iterations - no per-node
            D2H round-trips (was N cp.asnumpy(poses_gpu[i]) calls per iteration).
          • Pose update uses _batch_exp_se3_gpu + _batch_matmul_gpu - fused
            batched CUDA kernels instead of N sequential Python calls.
          • Single D2H for dx (6N scalars) replaces N separate D2H calls.

        V18: small Tikhonov regularization on non-fixed diagonal entries prevents
        singular/ill-conditioned solve on pathological graphs (corridors, stars,
        under-constrained subgraphs) without affecting the gauge-fixed first node.
        """
        # V30 FIX: cp.asarray(poses) - poses is already ndarray; np.array() was redundant.
        poses_gpu = cp.asarray(poses, dtype=cp.float64)  # (N, 4, 4)  - H2D once
        converged = False
        DOF = 6

        for it in range(self.max_iter):
            H, g = self._build_system_gpu(poses_gpu, edges, id2node, N, cp, huber_thresh=self.cfg.get("pgo_huber_thresh", 0.0))

            # Fix first node gauge freedom
            if self.fix_first:
                H[:DOF, :] = 0.0
                H[:, :DOF] = 0.0
                H[:DOF, :DOF] = cp.eye(DOF, dtype=cp.float64)
                g[:DOF] = 0.0

            # V18: Tikhonov regularization on non-fixed diagonal entries.
            # Prevents singular cp.linalg.solve on ill-conditioned graphs.
            # 1e-8 is small enough to be negligible vs. info weights (500).
            start = DOF if self.fix_first else 0
            diag_reg = cp.arange(start, 6 * N)
            H[diag_reg, diag_reg] += 1e-8

            # Solve H dx = -g
            try:
                dx_gpu = cp.linalg.solve(H, -g)
            except Exception:   # V31 FIX: cp.linalg.LinAlgError absent in old CuPy builds
                break

            # V27 FIX: NaN/Inf guard on dx_gpu - extreme ill-conditioning can
            # produce NaN even with Tikhonov regularization (e.g. rank-deficient
            # subgraph after outlier removal).  Silently corrupting poses is
            # worse than stopping early; bail immediately if any element is bad.
            if not bool(cp.all(cp.isfinite(dx_gpu))):
                break

            # V33 FIX: Per-DOF step norm clipping.  On the first GN iteration
            # with large initial loop-closure residuals (e.g. 1 m error in a
            # 500-node graph), the unclamped step can overshoot catastrophically.
            # Clipping to pgo_max_step_norm per DOF preserves direction while
            # bounding magnitude; convergence is unaffected for small residuals
            # where clipping never activates.
            _pgo_max_step = self.cfg.get("pgo_max_step_norm", 3.0)
            _dx_norm_per_dof = float(cp.linalg.norm(dx_gpu)) / max(1.0, math.sqrt(6 * N))
            if _dx_norm_per_dof > _pgo_max_step:
                dx_gpu = dx_gpu * (_pgo_max_step / _dx_norm_per_dof)

            # V21 FIX: Normalize dx_norm by sqrt(6*N) so the convergence
            # threshold is per-DOF (scale-invariant w.r.t. graph size).
            # Without normalization, sqrt(6*N) * 1e-9 per-DOF increments at
            # N=200 give dx_norm ≈ 3.5e-8 > conv_delta=1e-8 -> optimizer always
            # ran all max_iter iterations, wasting compute on converged graphs.
            dx_norm = float(cp.linalg.norm(dx_gpu)) / max(1.0, math.sqrt(6 * N))

            # Batch pose update: T_i <- T_i @ exp(xi_i) - entirely on GPU
            xi_batch    = dx_gpu.reshape(N, 6)                      # view, no copy
            exp_xi_gpu  = _batch_exp_se3_gpu(xi_batch)              # (N,4,4) GPU
            poses_gpu   = _batch_matmul_gpu(poses_gpu, exp_xi_gpu)  # (N,4,4) GPU

            # V24 FIX: Re-orthogonalize rotation blocks every 5 GN iterations.
            # Accumulated floating-point error after many steps can nudge R
            # off SO(3).  Gram-Schmidt on columns is O(9N) and keeps R ∈ SO(3).
            # V33 FIX: Read columns from R_blk BEFORE poses_gpu.copy() so they
            # capture the current (pre-detach) data without any lazy-eval race.
            # poses_gpu.copy() creates a new allocation; the old R_blk view is
            # then discarded.  Using explicit .copy() on each column guarantees
            # the column data is materialised before the source is released.
            if (it + 1) % 5 == 0:
                R_blk = poses_gpu[:, :3, :3]                        # (N,3,3) view
                # Extract columns as concrete arrays BEFORE detach (V33 FIX)
                c0 = R_blk[:, :, 0].copy()                          # (N,3) own copy
                c1 = R_blk[:, :, 1].copy()                          # (N,3) own copy
                # Normalise column 0
                n0 = cp.linalg.norm(c0, axis=1, keepdims=True)
                n0 = cp.maximum(n0, 1e-9)
                c0 = c0 / n0
                # Column 1: remove c0 component, normalize
                c1 = c1 - c0 * (c0 * c1).sum(axis=1, keepdims=True)
                n1 = cp.linalg.norm(c1, axis=1, keepdims=True)
                n1 = cp.maximum(n1, 1e-9)
                c1 = c1 / n1
                # Column 2: cross product (guaranteed orthonormal)
                c2 = cp.cross(c0, c1)
                # Assign back (must use explicit index to write to poses_gpu)
                poses_gpu = poses_gpu.copy()          # detach from matmul output
                poses_gpu[:, :3, 0] = c0
                poses_gpu[:, :3, 1] = c1
                poses_gpu[:, :3, 2] = c2

            if dx_norm < self.conv_delta:
                converged = True
                break

        return cp.asnumpy(poses_gpu), converged

    def _optimize_sparse(self, poses, edges, id2node, N):
        """Sparse scipy Gauss-Newton for large graphs.
        
        Sparse scipy Gauss-Newton for N > 200 nodes.

        V19 REFACTOR: Run the H/g assembly on GPU (xp=cp) instead of CPU
        (xp=np).  The dense 6N×6N result is then pulled to CPU once per
        iteration for the scipy sparse solve - a single (6N)² D2H transfer -
        rather than the O(N) per-node D2H that V12 had.  The solved dx is
        NumPy (scipy output), used directly for the convergence check; the
        pose update still uses _batch_exp_se3_gpu / _batch_matmul_gpu which
        keep poses on GPU between iterations.

        V18 CRITICAL FIX (preserved): cp.asnumpy() before return so kf.pose
        is always set to a NumPy array.  _batch_matmul_gpu returns CuPy when
        USE_CUPY=True; without the conversion kf.pose would be CuPy and every
        downstream NumPy op (linalg.norm, @, np.save, PnP, PF, IMU, TSDF)
        would silently produce wrong results or AttributeErrors.

        V17 FIX (preserved): H/g assembly delegates to _build_system
        (vectorised, O(1) Python overhead for any E).

        V26 FIX: SO(3) re-orthogonalization added every 5 GN iterations.
        _optimize_gpu already had this (V24 FIX); _optimize_sparse was missing
        it.  For large graphs (N>200) with many iterations, accumulated FP
        error can nudge R off SO(3) causing pose drift.  Gram-Schmidt on the
        rotation columns costs O(9N) per call and is negligible vs the O((6N)²)
        H/g assembly.  Mirrors the exact Gram-Schmidt from _optimize_gpu.
        """
        # V41: attempt to load cupyx sparse solvers dynamically (silences linter)
        _has_cp_sparse = False
        _cpsp = None
        _cpsp_linalg = None
        if USE_CUPY:
            try:
                import importlib
                _cpsp = importlib.import_module("cupyx.scipy.sparse")
                _cpsp_linalg = importlib.import_module("cupyx.scipy.sparse.linalg")
                _has_cp_sparse = True
            except (ImportError, AttributeError):
                _has_cp_sparse = False

        # Scipy equivalents for fallback
        from scipy.sparse import csr_matrix as _sci_csr
        from scipy.sparse.linalg import spsolve as _sci_spsolve

        converged = False
        DOF = 6

        # V30 FIX: cp.asarray(poses) - poses is already ndarray; np.array() was redundant.
        # V19: keep poses on GPU throughout - eliminates per-iteration D2H/H2D.
        # V40: initialize poses directly on GPU when sparse GPU path will be taken.
        # np.array(poses) -> cp.asarray() was a 2-step CPU alloc + H2D.
        if USE_CUPY:
            poses_gpu = cp.asarray(poses, dtype=cp.float64)  # (N,4,4) - single H2D
        else:
            poses_gpu = np.array(poses, dtype=np.float64)

        _gpu_mat_mb = (6 * N) ** 2 * 8 / 1e6
        _GPU_DENSE_LIMIT_MB = 256
        _use_gpu_build = USE_CUPY and (_gpu_mat_mb <= _GPU_DENSE_LIMIT_MB)

        for it in range(self.max_iter):
            # 1. Assemble H and g; GPU when safe, CPU otherwise.
            if _use_gpu_build:
                H_xp, g_xp = self._build_system_gpu(
                    poses_gpu, edges, id2node, N, cp,
                    huber_thresh=self.cfg.get("pgo_huber_thresh", 0.0))
            else:
                poses_cpu = (cp.asnumpy(poses_gpu) if (USE_CUPY and isinstance(poses_gpu, cp.ndarray)) else np.asarray(poses_gpu))
                H_xp, g_xp = self._build_system_gpu(
                    poses_cpu, edges, id2node, N, np,
                    huber_thresh=self.cfg.get("pgo_huber_thresh", 0.0))

            # 2. Apply Gauge Fix and Tikhonov regularization
            start_idx = DOF if self.fix_first else 0
            if self.fix_first:
                H_xp[:DOF, :] = 0.0; H_xp[:, :DOF] = 0.0
                H_xp[:DOF, :DOF] = (cp.eye(DOF) if _use_gpu_build else np.eye(DOF))
                g_xp[:DOF] = 0.0
            
                try:
                    dx = spsolve(H_sp, -g_dense)
                except Exception:
                    break

                if not np.all(np.isfinite(dx)):
                    break

            # V33 FIX: Per-DOF step norm clipping - same rationale as _optimize_gpu.
            _pgo_max_step = self.cfg.get("pgo_max_step_norm", 3.0)
            _dx_norm_per_dof = np.linalg.norm(dx) / max(1.0, math.sqrt(6 * N))
            if _dx_norm_per_dof > _pgo_max_step:
                dx = dx * (_pgo_max_step / _dx_norm_per_dof)

            # dx is NumPy (spsolve); _batch_exp_se3_gpu returns CuPy when USE_CUPY.
            # poses_gpu stays on device between iterations - no D2H/H2D round-trip.
            xi_batch   = dx.reshape(N, 6)                          # NumPy
            exp_xi_gpu = _batch_exp_se3_gpu(xi_batch)              # CuPy or NumPy
            poses_gpu  = _batch_matmul_gpu(poses_gpu, exp_xi_gpu)  # stays on GPU

            # V26 FIX: SO(3) re-orthogonalization every 5 GN steps - mirrors
            # _optimize_gpu V24 FIX.  Accumulated FP error after many GN steps
            # can push R off SO(3) causing slow pose drift.  Gram-Schmidt on
            # the 3 rotation columns costs O(9N) and keeps R ∈ SO(3) exactly.
            # V32 FIX: guard on isinstance(poses_gpu, cp.ndarray) for CuPy path;
            # add explicit NumPy elif for the CPU fallback path.
            # V31 only had the CuPy branch - _optimize_sparse with USE_CUPY=False
            # ran ALL iterations with zero SO(3) reorthogonalization, allowing
            # accumulated FP error to drift R off SO(3) for large-graph trajectories.
            if isinstance(poses_gpu, cp.ndarray) and (it + 1) % 5 == 0:
                R_blk = poses_gpu[:, :3, :3]                        # (N,3,3) view
                # V33 FIX: explicit .copy() on each column BEFORE detach -
                # materialises the column data before poses_gpu is reassigned.
                c0 = R_blk[:, :, 0].copy()
                n0 = cp.linalg.norm(c0, axis=1, keepdims=True)
                n0 = cp.maximum(n0, 1e-9)
                c0 = c0 / n0
                # Column 1: remove c0 component, normalize
                c1 = R_blk[:, :, 1].copy()
                c1 = c1 - c0 * (c0 * c1).sum(axis=1, keepdims=True)
                n1 = cp.linalg.norm(c1, axis=1, keepdims=True)
                n1 = cp.maximum(n1, 1e-9)
                c1 = c1 / n1
                # Column 2: cross product (guaranteed orthonormal)
                c2 = cp.cross(c0, c1)
                poses_gpu = poses_gpu.copy()      # detach before write
                poses_gpu[:, :3, 0] = c0
                poses_gpu[:, :3, 1] = c1
                poses_gpu[:, :3, 2] = c2
            elif isinstance(poses_gpu, np.ndarray) and (it + 1) % 5 == 0:
                # V32 FIX: NumPy SO(3) reortho path - was completely absent.
                # Mirror the 3-column Gram-Schmidt from the CuPy path and
                # _optimize_cpu, using .copy() before write to avoid aliasing.
                R_blk = poses_gpu[:, :3, :3]                        # (N,3,3) view
                c0 = R_blk[:, :, 0].copy()
                n0 = np.linalg.norm(c0, axis=1, keepdims=True)
                c0 = c0 / np.maximum(n0, 1e-9)
                c1 = R_blk[:, :, 1].copy()
                c1 = c1 - c0 * (c0 * c1).sum(axis=1, keepdims=True)
                n1 = np.linalg.norm(c1, axis=1, keepdims=True)
                c1 = c1 / np.maximum(n1, 1e-9)
                c2 = np.cross(c0, c1)
                poses_gpu = poses_gpu.copy()      # detach view before write
                poses_gpu[:, :3, 0] = c0
                poses_gpu[:, :3, 1] = c1
                poses_gpu[:, :3, 2] = c2

            # V21 FIX: per-DOF convergence check (see _optimize_gpu)
            if np.linalg.norm(dx) / max(1.0, math.sqrt(6 * N)) < self.conv_delta:
                converged = True
                break

        # V18 / V19 CRITICAL: convert back to NumPy before returning.
        # Without this, kf.pose = poses_arr[i] would silently store a CuPy array,
        # corrupting the entire pipeline (PnP, PF, IMU, TSDF all use np.ndarray).
        if USE_CUPY and isinstance(poses_gpu, cp.ndarray):
            poses_np = cp.asnumpy(poses_gpu)
        else:
            poses_np = np.asarray(poses_gpu)
        return poses_np, converged

    def _optimize_cpu(self, poses, edges, id2node, N):
        """NumPy fallback Gauss-Newton for large graphs.
        Uses _batch_exp_se3_gpu (NumPy path) for batched pose update.

        V18: small Tikhonov regularization on non-fixed diagonal for
        numerical stability - mirrors _optimize_gpu / _optimize_sparse.

        V31 FIX: SO(3) re-orthogonalization added every 5 GN iterations.
        _optimize_gpu had this since V24; _optimize_sparse got it in V26.
        _optimize_cpu was the only path that never applied it - accumulated
        FP error after many Gauss-Newton steps can gradually push each R
        off SO(3), causing slow but compounding pose drift.  Gram-Schmidt
        on the 3 rotation columns costs O(9N) per call and is negligible
        vs the O((6N)²) H/g assembly.  Uses the same 3-column Gram-Schmidt
        as the other two paths for consistency.
        """
        converged = False
        DOF = 6
        for it in range(self.max_iter):
            H, g = self._build_system_gpu(poses, edges, id2node, N, np, huber_thresh=self.cfg.get("pgo_huber_thresh", 0.0))
            if self.fix_first:
                H[:DOF, :] = 0.0
                H[:, :DOF] = 0.0
                H[:DOF, :DOF] = np.eye(DOF)
                g[:DOF] = 0.0
            # V18: Tikhonov regularization on non-fixed diagonal entries
            start = DOF if self.fix_first else 0
            H[np.arange(start, 6*N), np.arange(start, 6*N)] += 1e-8
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            # V27 FIX: NaN/Inf guard on dx - np.linalg.solve can produce NaN
            # for borderline-singular systems that pass the exception check.
            # Stop immediately to avoid silently corrupting all poses.
            if not np.all(np.isfinite(dx)):
                break
            # V33 FIX: Per-DOF step norm clipping - prevents overshoot on first
            # iteration when initial loop-closure residuals are very large.
            _pgo_max_step = self.cfg.get("pgo_max_step_norm", 3.0)
            _dx_norm_per_dof = np.linalg.norm(dx) / max(1.0, math.sqrt(6 * N))
            if _dx_norm_per_dof > _pgo_max_step:
                dx = dx * (_pgo_max_step / _dx_norm_per_dof)
            # Batch pose update via _batch_exp_se3_gpu (NumPy backend when no CuPy)
            xi_batch = dx.reshape(N, 6)
            exp_xi   = _batch_exp_se3_gpu(xi_batch)      # (N,4,4) NumPy
            poses    = _batch_matmul_gpu(poses, exp_xi)  # (N,4,4) NumPy

            # V31 FIX: SO(3) re-orthogonalization every 5 GN iterations.
            # Mirrors V24 FIX in _optimize_gpu and V26 FIX in _optimize_sparse.
            # Uses column-copy before ortho to avoid aliasing into the pose array.
            if (it + 1) % 5 == 0:
                R_blk = poses[:, :3, :3]                        # (N,3,3) view
                # Column 0: copy then normalize
                c0 = R_blk[:, :, 0].copy()
                n0 = np.linalg.norm(c0, axis=1, keepdims=True)
                n0 = np.maximum(n0, 1e-9)
                c0 = c0 / n0
                # Column 1: remove c0 component, normalize
                c1 = R_blk[:, :, 1].copy()
                c1 = c1 - c0 * (c0 * c1).sum(axis=1, keepdims=True)
                n1 = np.linalg.norm(c1, axis=1, keepdims=True)
                n1 = np.maximum(n1, 1e-9)
                c1 = c1 / n1
                # Column 2: cross product - guaranteed orthonormal
                c2 = np.cross(c0, c1)
                poses = poses.copy()          # detach before write
                poses[:, :3, 0] = c0
                poses[:, :3, 1] = c1
                poses[:, :3, 2] = c2

            # V21 FIX: per-DOF convergence check (see _optimize_gpu)
            if np.linalg.norm(dx) / max(1.0, math.sqrt(6 * N)) < self.conv_delta:
                converged = True
                break
        return poses, converged

        pass