"""GPU batched PnP RANSAC using Kornia."""

import numpy as np
import torch
from ..utils.logger import get_logger

logger = get_logger(__name__)

_HAS_KORNIA = False
try:
    import kornia.geometry
    _HAS_KORNIA = True
except ImportError:
    pass

def batched_gpu_pnp_ransac(pts3d_t, pts2d_t, K_t, reproj_thresh=2.0, n_iter=256, min_pts=6, T_cw_hint=None):
    """
    Vectorised GPU RANSAC-PnP using Kornia's batched DLT solver.

    Parameters
    ----------
    pts3d_t : torch.Tensor (N,3) float32 CUDA
    pts2d_t : torch.Tensor (N,2) float32 CUDA
    K_t     : torch.Tensor (3,3) float32 CUDA
    reproj_thresh : float (pixels)
    n_iter   : int
    min_pts  : int
    T_cw_hint : Optional torch.Tensor (4,4) float32 CUDA

    Returns
    -------
    ok : bool
    T_cw_np : np.ndarray (4,4) float64 camera-from-world
    inlier_mask : np.ndarray (N,) bool
    n_inliers : int
    """
    if not _HAS_KORNIA:
        return False, None, None, 0

    N = pts3d_t.shape[0]
    dev = pts3d_t.device
    if N < min_pts:
        return False, None, None, 0

    # Sample all hypotheses at once
    hyp_idx = torch.randint(0, N, (n_iter, min_pts), device=dev)
    pts3d_hyp = pts3d_t[hyp_idx]
    pts2d_hyp = pts2d_t[hyp_idx]
    K_b = K_t.unsqueeze(0).expand(n_iter, -1, -1).contiguous()

    try:
        T_cw_b = kornia.geometry.solve_pnp_dlt(pts3d_hyp, pts2d_hyp, K_b)  # (B,3,4)
    except Exception:
        return False, None, None, 0

    # Evaluate all hypotheses
    R_b = T_cw_b[:, :3, :3]
    t_b = T_cw_b[:, :3, 3]
    pts_cam = torch.einsum('bij,nj->bni', R_b, pts3d_t) + t_b[:, None, :]
    Z = pts_cam[..., 2]
    valid_z = Z > 0.01
    inv_Z = torch.where(valid_z, 1.0 / Z.clamp(min=1e-6), torch.zeros_like(Z))
    fx, cx = K_t[0, 0], K_t[0, 2]
    fy, cy = K_t[1, 1], K_t[1, 2]
    u_proj = fx * pts_cam[..., 0] * inv_Z + cx
    v_proj = fy * pts_cam[..., 1] * inv_Z + cy
    du = u_proj - pts2d_t[None, :, 0]
    dv = v_proj - pts2d_t[None, :, 1]
    reproj = torch.sqrt(du*du + dv*dv)
    inl_b = (reproj < reproj_thresh) & valid_z
    counts = inl_b.sum(dim=1)

    # Evaluate hint if provided
    best_hint_count = 0
    hint_inl_mask = None
    if T_cw_hint is not None:
        R_h = T_cw_hint[:3, :3].unsqueeze(0)
        t_h = T_cw_hint[:3, 3].unsqueeze(0)
        pc_h = torch.einsum('bij,nj->bni', R_h, pts3d_t) + t_h[:, None, :]
        Zh = pc_h[..., 2]
        vzh = Zh > 0.01
        iZh = torch.where(vzh, 1.0 / Zh.clamp(min=1e-6), torch.zeros_like(Zh))
        uh = fx * pc_h[..., 0] * iZh + cx
        vh = fy * pc_h[..., 1] * iZh + cy
        errh = torch.sqrt((uh - pts2d_t[None,:,0])**2 + (vh - pts2d_t[None,:,1])**2)
        inlh = (errh < reproj_thresh) & vzh
        best_hint_count = int(inlh.sum().item())
        if best_hint_count > 0:
            hint_inl_mask = inlh.squeeze(0)

    best_ransac_idx = int(counts.argmax().item())
    best_ransac_count = int(counts[best_ransac_idx].item())

    if best_hint_count > best_ransac_count and hint_inl_mask is not None:
        best_count = best_hint_count
        best_inl_mask = hint_inl_mask
        T_fallback_34 = T_cw_hint[:3, :].unsqueeze(0)
    else:
        best_count = best_ransac_count
        best_inl_mask = inl_b[best_ransac_idx]
        T_fallback_34 = T_cw_b[best_ransac_idx].unsqueeze(0)

    if best_count < min_pts:
        return False, None, None, 0

    # Refine on inliers
    inl_idx = torch.where(best_inl_mask)[0]
    T_cw_np = np.eye(4, dtype=np.float64)
    try:
        if len(inl_idx) >= min_pts:
            K_1 = K_t.unsqueeze(0)
            T_ref = kornia.geometry.solve_pnp_dlt(pts3d_t[inl_idx].unsqueeze(0),
                                                  pts2d_t[inl_idx].unsqueeze(0),
                                                  K_1)
            T_cw_np[:3, :] = T_ref[0].cpu().numpy().astype(np.float64)
        else:
            T_cw_np[:3, :] = T_fallback_34[0].cpu().numpy().astype(np.float64)
    except Exception:
        T_cw_np[:3, :] = T_fallback_34[0].cpu().numpy().astype(np.float64)

    return True, T_cw_np, best_inl_mask.cpu().numpy(), best_count

# Alias for compatibility with existing code that expects _batched_gpu_pnp_ransac
_batched_gpu_pnp_ransac = batched_gpu_pnp_ransac