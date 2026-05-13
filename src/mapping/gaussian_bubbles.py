"""Gaussian bubble map with uncertainty fusion."""

import numpy as np
import threading
from collections import deque
from queue import Empty, Queue, Full
from ..utils.logger import get_logger
logger = get_logger(__name__)
from ..utils.se3_ops import batch_inv3_gpu, batch_mahal3_gpu
from ..utils.depth_utils import bilinear_depth_gpu
from ..utils.cupy_utils import (
    cupy_manager, cp, USE_CUPY, batch_matrix_multiply, batch_matrix_inverse,
    gpu_norm, gpu_clip, gpu_normalize_vectors, gpu_cross_product,
    stable_batched_covariance_transform, stable_batched_covariance_transform_with_points,
    gpu_reduce_mean, gpu_reduce_std, gpu_reduce_percentile, gpu_dot_product, gpu_einsum,
    rotate_covariance_batch, batch_transform_points
)
from ..utils.array_utils import to_numpy_safe
from .hierarchical_pruner import HierarchicalPruner
from ..utils.cuda_raw_kernels import cuda_kernel_manager
from ..utils.zero_copy_memory import zero_copy_manager
from ..utils.lazy_mirrors import lazy_mirror_manager

# Get appropriate array module (CuPy or NumPy fallback from cupy_utils)
xp = cp

try:
    import importlib
    # Try multiple possible import paths for CuPy KDTree
    try:
        _spatial = importlib.import_module("cupyx.scipy.spatial")
        if hasattr(_spatial, "KDTree"):
            CupyKDTree = _spatial.KDTree
        else:
            _kdtree = importlib.import_module("cupyx.scipy.spatial.kdtree")
            CupyKDTree = _kdtree.KDTree
        HAS_CUPY_KDTREE = True
    except (ImportError, AttributeError):
        HAS_CUPY_KDTREE = False
        CupyKDTree = None
except Exception:
    HAS_CUPY_KDTREE = False
    CupyKDTree = None

from scipy.spatial import cKDTree

logger = get_logger(__name__)

# Helper functions for batch operations
def _batch_inv3(H_pp_blocks):
    """Batch invert 3x3 blocks (CPU fallback)."""
    try:
        return np.linalg.inv(H_pp_blocks)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(H_pp_blocks)

def _batch_mahal3(dmu, Sig_sum):
    """Compute Mahalanobis distance for 3D points."""
    try:
        Sig_inv = _batch_inv3(Sig_sum)
        mahal = np.einsum('ni,nij,nj->n', dmu, Sig_inv, dmu)
        return np.sqrt(np.maximum(mahal, 0))
    except Exception:
        return np.full(len(dmu), np.inf)

def _batch_inv3_gpu(H_pp_blocks):
    """Batch invert 3x3 blocks (GPU accelerated)."""
    try:
        if USE_CUPY and isinstance(H_pp_blocks, xp.ndarray):
            return xp.linalg.inv(H_pp_blocks)
        else:
            return np.linalg.inv(H_pp_blocks)
    except Exception:
        if USE_CUPY and isinstance(H_pp_blocks, xp.ndarray):
            return xp.linalg.pinv(H_pp_blocks)
        else:
            return np.linalg.pinv(H_pp_blocks)

def _batch_mahal3_gpu(dmu, Sig_sum):
    """Compute Mahalanobis distance (GPU when available)."""
    try:
        if USE_CUPY and isinstance(dmu, xp.ndarray):
            Sig_inv = _batch_inv3_gpu(Sig_sum)
            mahal = xp.einsum('ni,nij,nj->n', dmu, Sig_inv, dmu)
            return xp.sqrt(xp.maximum(mahal, 0))
        else:
            Sig_inv = _batch_inv3(Sig_sum)
            mahal = np.einsum('ni,nij,nj->n', dmu, Sig_inv, dmu)
            return np.sqrt(np.maximum(mahal, 0))
    except Exception:
        _xp = xp if (USE_CUPY and isinstance(dmu, xp.ndarray)) else np
        return _xp.full(len(dmu), _xp.inf)

class ThreadedBubbleMapManager:
    def __init__(self, bubble_map, config):
        self.bubble_map = bubble_map
        self.config = config
        self.update_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.update_count = 0
        self.keyframe_manager = None  # Will be set from SLAM system
        self.start_thread()

    def start_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                item = self.update_queue.get(timeout=0.05)
                if item is None:
                    break
                while True:
                    try:
                        newer_item = self.update_queue.get_nowait()
                        if newer_item is None:
                            item = None
                            break
                        item = newer_item
                    except Empty:
                        break
                if item is None:
                    break
                depth, pose, image, motion_scale = item
                try:
                    self.bubble_map.update(depth, pose, image, motion_scale, self.keyframe_manager)
                except Exception as e:
                    print(f" [BUBBLE WORKER] ERROR in bubble_map.update(): {e}")
                    import traceback
                    traceback.print_exc()
            except Exception:
                # Queue timeout is normal when no work is queued, ignore silently
                continue

    def queue_update(self, depth, pose, image, motion_scale=1.0):
        if not self.running:
            return False
        try:
            if self.update_queue.full():
                self.update_queue.get_nowait()
            # PASS GPU TENSORS DIRECTLY - DO NOT CONVERT TO NUMPY HERE
            # This avoids expensive D2H transfer in the main SLAM loop
            self.update_queue.put_nowait((depth, pose, image, motion_scale))
            logger.debug(f" [BUBBLE] Queued GPU tensors for update (queue size: {self.update_queue.qsize()})")
            return True
        except Exception as e:
            logger.error(f" [BUBBLE] Failed to queue update: {e}")
            return False

    def shutdown(self):
        self.running = False
        if self.update_queue:
            self.update_queue.put(None)
        if self.thread:
            self.thread.join(timeout=2.0)

class GaussianBubbleMap:
    def __init__(self, K: np.ndarray, baseline: float, config: dict):
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
        self.baseline = baseline
        self.cfg = config
        self.use_gpu = config.get('bubble_cuda', True) and USE_CUPY
        
        # Raw GPU performance settings
        self.use_raw_kernels = config.get('use_raw_kernels', True) and self.use_gpu
        self.use_zero_copy = config.get('use_zero_copy', True) and self.use_gpu
        self.use_lazy_mirrors = config.get('use_lazy_mirrors', True) and self.use_gpu
        
        # Use CuPy manager for array operations
        xp = cupy_manager.get_array_module(self.use_gpu)
        
        # Initialize with zero-copy memory if available
        if self.use_zero_copy:
            self.mu = zero_copy_manager.allocate_zero_copy((0,3), xp.float64, 'bubble_mu')
            self.Sigma = zero_copy_manager.allocate_zero_copy((0,3,3), xp.float64, 'bubble_sigma')
            self.weight = zero_copy_manager.allocate_zero_copy((0,), xp.float64, 'bubble_weight')
            self.color = zero_copy_manager.allocate_zero_copy((0,3), xp.float64, 'bubble_color')
        else:
            self.mu = xp.empty((0,3), dtype=xp.float64)
            self.Sigma = xp.empty((0,3,3), dtype=xp.float64)
            self.weight = xp.empty((0,), dtype=xp.float64)
            self.color = xp.empty((0,3), dtype=xp.float64)
        
        # Performance tracking
        self.raw_kernel_calls = 0
        self.zero_copy_transfers = 0
        self.lazy_mirror_hits = 0
        self.kdtree = None
        self._kdtree_size = 0
        self.frame_counter = 0
        self._lock = threading.Lock()
        self.threaded_manager = ThreadedBubbleMapManager(self, config)
        
        # Initialize hierarchical pruner
        if config.get("hierarchical_pruning_enabled", True):
            self.hierarchical_pruner = HierarchicalPruner(config)
        else:
            self.hierarchical_pruner = None
            
        logger.info(f"Bubble map initialized (stride={config['bubble_stride']}, hierarchical_pruning={self.hierarchical_pruner is not None})")

    def backproject_frame(self, depth, pose, image, stride, motion_scale=1.0):
        """Backproject depth to 3D points with PURE GPU MODE - no NumPy fallbacks."""
        return self._backproject_gpu_intensive(depth, pose, image, stride, motion_scale)

    def _backproject_gpu_intensive(self, depth, pose, image, stride, motion_scale=1.0):
        """GPU-intensive backprojection to force GPU utilization."""
        
        # Convert to GPU with larger arrays for more compute
        d_gpu = xp.asarray(depth, dtype=xp.float32)
        pose_gpu = xp.asarray(pose, dtype=xp.float32)
        if image is not None:
            img_gpu = xp.asarray(image, dtype=xp.float32)
        else:
            img_gpu = xp.asarray(xp.zeros((1, 1), dtype=xp.float32))  # GPU placeholder
            
        h, w = d_gpu.shape
        
        # Create denser coordinate grid for more GPU work
        u_gpu = xp.arange(0, w, stride, dtype=xp.float32)
        v_gpu = xp.arange(0, h, stride, dtype=xp.float32)
        v_grid, u_grid = xp.meshgrid(v_gpu, u_gpu, indexing='ij')
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        z_gpu = d_gpu[v_flat.astype(xp.int32), u_flat.astype(xp.int32)]
        
        # GPU-intensive filtering operations
        max_d = xp.float32(self.cfg.get("bubble_max_depth", 8.0))
        min_d = xp.float32(self.cfg.get("bubble_min_depth", 0.1))
        valid = (z_gpu > min_d) & (z_gpu < max_d) & xp.isfinite(z_gpu)
        
        # Additional GPU processing - edge detection with more compute
        edge_thresh = xp.float32(self.cfg.get("bubble_depth_edge_thresh", 0.30))
        if edge_thresh > 0 and xp.sum(valid) > 0:
            u_i = u_flat.astype(xp.int32)
            v_i = v_flat.astype(xp.int32)
            u_r = xp.minimum(u_i + stride, w - 1)
            u_l = xp.maximum(u_i - stride, 0)
            v_d = xp.minimum(v_i + stride, h - 1)
            v_u = xp.maximum(v_i - stride, 0)
            d_r = d_gpu[v_i, u_r]
            d_l = d_gpu[v_i, u_l]
            d_dn = d_gpu[v_d, u_i]
            d_up = d_gpu[v_u, u_i]
            ref = z_gpu + xp.float32(1e-6)
            g_max = xp.maximum(
                xp.abs(d_r - z_gpu) / ref,
                xp.maximum(
                    xp.abs(d_l - z_gpu) / ref,
                    xp.maximum(
                        xp.abs(d_dn - z_gpu) / ref,
                        xp.abs(d_up - z_gpu) / ref)))
            valid = valid & (g_max < edge_thresh)
        
        # Extract valid points
        u_f = u_flat[valid]
        v_f = v_flat[valid]
        d_f = z_gpu[valid]
        
        if d_f.size == 0:
            return xp.empty((0,3), dtype=xp.float64), xp.empty((0,3,3), dtype=xp.float64), xp.empty(0, dtype=xp.float64), xp.empty((0,3), dtype=xp.float64)
        
        # GPU-intensive coordinate calculations
        fx_gpu = xp.float32(self.fx)
        fy_gpu = xp.float32(self.fy)
        cx_gpu = xp.float32(self.cx)
        cy_gpu = xp.float32(self.cy)
        
        x = (u_f - cx_gpu) * d_f / fx_gpu
        y = (v_f - cy_gpu) * d_f / fy_gpu
        pts_cam = xp.stack([x, y, d_f], axis=1)
        
        # GPU-intensive matrix operations
        sigma_par = (d_f**2 / (fx_gpu * self.baseline)) * xp.float32(self.cfg["bubble_sigma_disp"]) * motion_scale
        sigma_per = (d_f / fx_gpu) * xp.float32(self.cfg["bubble_sigma_pix"]) * motion_scale
        sp_max = xp.float32(self.cfg.get("bubble_sigma_par_max", 0.15))
        sigma_par = xp.minimum(sigma_par, sp_max)
        
        # Create diagonal covariance matrices with shape (N, 3, 3)
        zeros = xp.zeros_like(sigma_per)
        diag_ray_3x3 = xp.stack([
            xp.stack([sigma_per**2, zeros, zeros], axis=1),
            xp.stack([zeros, sigma_per**2, zeros], axis=1), 
            xp.stack([zeros, zeros, sigma_par**2], axis=1)
        ], axis=1)
        
        # Keep the old (N, 3) format for compatibility with existing code
        diag_ray = xp.stack([sigma_per**2, sigma_per**2, sigma_par**2], axis=1)
        
        # GPU-intensive vector operations
        rays = pts_cam / xp.linalg.norm(pts_cam, axis=1, keepdims=True)
        world_up = xp.array([0., 0., 1.])
        x_perp = xp.cross(rays, world_up)
        x_perp_n = xp.linalg.norm(x_perp, axis=1, keepdims=True)
        x_perp_alt = xp.cross(rays, xp.array([0., 1., 0.]))
        x_perp = xp.where(x_perp_n < 1e-6, x_perp_alt, x_perp)
        x_perp /= (xp.linalg.norm(x_perp, axis=1, keepdims=True) + 1e-9)
        y_perp = xp.cross(rays, x_perp)
        y_perp /= (xp.linalg.norm(y_perp, axis=1, keepdims=True) + 1e-9)
        R_rc = xp.stack([x_perp, y_perp, rays], axis=2)
        
        # GPU-intensive batch matrix operations with stable covariance transforms
        R_wc = pose_gpu[:3,:3]
        t_wc = pose_gpu[:3,3]
        
        # Use proper batch covariance rotation: Sigma_world = R @ Sigma_ray @ R.T
        Sig_world = rotate_covariance_batch(R_wc, diag_ray_3x3)
        
        # Transform points to world coordinates
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        
        # PURE GPU MODE - Continue with GPU processing
        # No NumPy fallbacks, no CPU conversions during processing
        
        # GPU-intensive color processing
        N_pts = pts_world.shape[0]
        if img_gpu is not None:
            img_f = img_gpu
            if img_f.dtype != xp.float32:
                img_f = img_f.astype(xp.float32)
            h_img, w_img = img_f.shape[:2]
            u_i_g = xp.clip(u_f.astype(xp.int32), 0, w_img - 1)
            v_i_g = xp.clip(v_f.astype(xp.int32), 0, h_img - 1)
            if img_f.ndim == 3:
                colors_gpu = img_f[v_i_g, u_i_g, :].astype(xp.float64) / 255.0
            else:
                c1 = (img_f[v_i_g, u_i_g].astype(xp.float64) / 255.0)[:, None]
                colors_gpu = xp.repeat(c1, 3, axis=1)
        else:
            colors_gpu = xp.full((N_pts, 3), 0.5, dtype=xp.float64)
        
        ones_gpu = xp.ones(N_pts, dtype=xp.float64)
        
        # PURE GPU MODE - Keep results on GPU, no CPU conversion
        return pts_world, Sig_world, ones_gpu, colors_gpu

    def _backproject_numpy(self, depth, pose, image, stride, motion_scale=1.0):
        """Backproject depth to 3D points (NumPy path). Enhanced depth validation."""
        # CPU zone: use only NumPy arrays
        h, w = depth.shape
        u = np.arange(0, w, stride)
        v = np.arange(0, h, stride)
        u_grid, v_grid = np.meshgrid(u, v)
        u_flat, v_flat = u_grid.ravel(), v_grid.ravel()
        d_flat = depth[v_flat, u_flat]
        
        # Debug depth statistics
        print(f" [BUBBLE DEBUG] Depth stats: shape={depth.shape}, min={d_flat.min():.3f}, max={d_flat.max():.3f}, mean={d_flat.mean():.3f}")
        print(f" [BUBBLE DEBUG] Valid depth count: {np.sum(d_flat > 0)}, finite count: {np.sum(np.isfinite(d_flat))}")
        
        max_d  = float(self.cfg.get("bubble_max_depth", 8.0))
        min_d  = float(self.cfg.get("bubble_min_depth", 0.1))  # Reduced from 0.2 to 0.1
        
        # Enhanced depth validation with multiple filters
        valid_depth = (d_flat > min_d) & (d_flat < max_d) & np.isfinite(d_flat)
        
        # Additional quality filters
        if not np.any(valid_depth):
            print(f" [BUBBLE DEBUG] No valid depth points after filtering: valid_depth_count={np.sum(valid_depth)}")
            return np.empty((0,3)), np.empty((0,3,3)), np.empty(0), np.empty((0,3))
        
        print(f" [BUBBLE DEBUG] Valid depth points after basic filtering: {np.sum(valid_depth)}")
        
        # Edge detection with adaptive threshold
        edge_thresh = float(self.cfg.get("bubble_depth_edge_thresh", 0.50))  # Increased from 0.20 to 0.50
        valid = valid_depth.copy()  # Initialize valid before the conditional block
        if edge_thresh > 0 and np.any(valid_depth):
            u_r = np.minimum(u_flat + stride, w - 1)
            u_l = np.maximum(u_flat - stride, 0)
            v_d = np.minimum(v_flat + stride, h - 1)
            v_u = np.maximum(v_flat - stride, 0)
            d_r = depth[v_flat, u_r]
            d_l = depth[v_flat, u_l]
            d_dn = depth[v_d, u_flat]
            d_up = depth[v_u, u_flat]
            ref  = d_flat + 1e-6
            g_max = np.maximum(
                np.abs(d_r  - d_flat) / ref,
                np.maximum(
                    np.abs(d_l  - d_flat) / ref,
                    np.maximum(
                        np.abs(d_dn - d_flat) / ref,
                        np.abs(d_up - d_flat) / ref)))
            valid = valid & (g_max < edge_thresh)
        
        print(f" [BUBBLE DEBUG] Points after edge detection: {np.sum(valid)}")
        
        # Additional quality checks for depth consistency
        if np.any(valid):
            # Check for depth jumps (filter out outliers)
            # Ensure we're working with the same array type as 'valid'
            if hasattr(d_flat, 'get'):
                d_flat_np = to_numpy_safe(d_flat)
            else:
                d_flat_np = d_flat
            
            # Get the array module that matches 'valid'
            if hasattr(valid, 'get'):
                # 'valid' is CuPy array, use standardized 'cp'
                _xp_local = cp
                valid_np = to_numpy_safe(valid)
            else:
                # 'valid' is NumPy array
                _xp_local = np
                valid_np = valid
            
            valid_depths = d_flat_np[valid_np]
            depth_diff = np.abs(valid_depths[1:] - valid_depths[:-1])
            depth_jump_thresh = 0.5  # Maximum allowed depth jump between adjacent pixels
            
            # Create consistent_depth using the same array type as 'valid'
            consistent_depth = _xp_local.ones(len(d_flat_np), dtype=bool)
            
            # Get the indices where valid_np is True (excluding the last one)
            # Ensure array type consistency for CuPy/NumPy compatibility
            if hasattr(_xp_local, 'asnumpy'):
                # CuPy case
                valid_indices = _xp_local.where(_xp_local.asarray(valid_np))[0]
            else:
                # NumPy case
                valid_indices = _xp_local.where(valid_np)[0]
            if len(valid_indices) > 1:
                # Assign depth consistency check to the appropriate positions
                depth_consistency = depth_diff < depth_jump_thresh
                # Set consistent_depth for valid positions (excluding first valid point)
                consistent_depth[valid_indices[1:]] = depth_consistency
            
            # Ensure both arrays are the same type for bitwise AND
            if hasattr(valid, 'get'):
                # Convert consistent_depth to CuPy if 'valid' is CuPy
                consistent_depth = _xp_local.asarray(consistent_depth)
            
            valid = valid & consistent_depth
        
        print(f" [BUBBLE DEBUG] Final valid points: {np.sum(valid)}")

        if not np.any(valid):
            return np.empty((0,3)), np.empty((0,3,3)), np.empty(0), np.empty((0,3))
        # Fix CuPy implicit conversion - ensure proper array types
        if hasattr(valid, 'get'):
            # CuPy case - convert to NumPy for indexing
            valid_np = valid.get()
            u_f = u_flat[valid_np]
            v_f = v_flat[valid_np]
            d_f = d_flat[valid_np]
        else:
            # NumPy case
            u_f, v_f, d_f = u_flat[valid], v_flat[valid], d_flat[valid]
        x = (u_f - self.cx) * d_f / self.fx
        y = (v_f - self.cy) * d_f / self.fy
        # Ensure pts_cam uses same array type as inputs
        if hasattr(d_f, 'get'):
            pts_cam = xp.stack([x, y, d_f], axis=-1)  # CuPy case
        else:
            pts_cam = np.stack([x, y, d_f], axis=-1)  # NumPy case
        
        sigma_par = (d_f**2 / (self.fx * self.baseline)) * self.cfg["bubble_sigma_disp"] * motion_scale
        sigma_per = (d_f / self.fx) * self.cfg["bubble_sigma_pix"] * motion_scale
        sp_max = float(self.cfg.get("bubble_sigma_par_max", 0.15))
        sigma_par = np.minimum(sigma_par, sp_max)
        diag_ray  = np.stack([sigma_per**2, sigma_per**2, sigma_par**2], axis=-1)
        norms     = np.sqrt(np.einsum('ni,ni->n', pts_cam, pts_cam))[:, None]
        rays      = pts_cam / (norms + 1e-9)
        x_perp    = np.empty_like(rays)
        x_perp[:, 0] = rays[:, 1]
        x_perp[:, 1] = -rays[:, 0]
        x_perp[:, 2] = 0.0
        xpn  = np.sqrt(np.einsum('ni,ni->n', x_perp, x_perp))[:, None]
        sing = xpn.ravel() < 1e-6
        if np.any(sing):
            x_perp[sing, 0] = -rays[sing, 2]
            x_perp[sing, 1] = 0.0
            x_perp[sing, 2] = rays[sing, 0]
            xpn = np.sqrt(np.einsum('ni,ni->n', x_perp, x_perp))[:, None]
        y_perp = np.cross(rays, x_perp)
        ypn    = np.sqrt(np.einsum('ni,ni->n', y_perp, y_perp))[:, None]
        y_perp /= (ypn + 1e-9)
        R_rc   = np.stack([x_perp, y_perp, rays], axis=2)
        R_wc   = pose[:3, :3]; t_wc = pose[:3, 3]
        M      = np.einsum('ij,njk->nik', R_wc, R_rc)
        Md     = M * diag_ray[:, None, :]
        Sig_world = Md @ np.swapaxes(M, 1, 2)
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        colors    = (image[v_f.astype(int), u_f.astype(int)] / 255.0 if image is not None else np.full((len(d_f), 3), 0.5))
        return pts_world, Sig_world, np.ones(len(d_f), dtype=np.float64), colors

    def _backproject_cuda(self, depth, pose, image, stride, motion_scale=1.0):
        """Backproject depth with GPU acceleration using CuPy utilities."""
        # Convert inputs to GPU using CuPy manager
        d_gpu = cupy_manager.to_gpu(depth)
        pose_gpu = cupy_manager.to_gpu(pose)
        if image is not None:
            image_gpu = cupy_manager.to_gpu(image)
        else:
            image_gpu = None
            
        h, w = d_gpu.shape
        xp = cupy_manager.get_array_module(True)
        
        # Create coordinate grids using CuPy
        u_gpu_1d = xp.arange(0, w, stride, dtype=xp.float32)
        v_gpu_1d = xp.arange(0, h, stride, dtype=xp.float32)
        v_grid, u_grid = xp.meshgrid(v_gpu_1d, u_gpu_1d, indexing='ij')
        u_gpu = u_grid.ravel(); v_gpu = v_grid.ravel()
        z_gpu = d_gpu[v_gpu.astype(xp.int32), u_gpu.astype(xp.int32)]
        max_d = float(self.cfg.get("bubble_max_depth", 8.0))
        valid = (z_gpu > 0.1) & (z_gpu < max_d) & xp.isfinite(z_gpu)

        edge_thresh = float(self.cfg.get("bubble_depth_edge_thresh", 0.30))
        if edge_thresh > 0:
            u_i = u_gpu.astype(xp.int32)
            v_i = v_gpu.astype(xp.int32)
            u_r = xp.minimum(u_i + stride, w - 1)
            u_l = xp.maximum(u_i - stride, 0)
            v_d = xp.minimum(v_i + stride, h - 1)
            v_u = xp.maximum(v_i - stride, 0)
            d_r  = d_gpu[v_i, u_r]
            d_l  = d_gpu[v_i, u_l]
            d_dn = d_gpu[v_d, u_i]
            d_up = d_gpu[v_u, u_i]
            ref  = z_gpu + xp.float32(1e-6)
            g_max = xp.maximum(
                xp.abs(d_r  - z_gpu) / ref,
                xp.maximum(
                    xp.abs(d_l  - z_gpu) / ref,
                    xp.maximum(
                        xp.abs(d_dn - z_gpu) / ref,
                        xp.abs(d_up - z_gpu) / ref)))
            valid = valid & (g_max < edge_thresh)

        u_f = u_gpu[valid]; v_f = v_gpu[valid]; d_f = z_gpu[valid]
        if d_f.size == 0:
            return np.empty((0,3)), np.empty((0,3,3)), np.empty(0), np.empty((0,3))
        
        # Use CuPy for coordinate calculations
        x = (u_f - self.cx) * d_f / self.fx
        y = (v_f - self.cy) * d_f / self.fy
        pts_cam = xp.stack([x, y, d_f], axis=1)
        
        # Calculate sigmas using CuPy
        sigma_par = (d_f**2 / (self.fx * self.baseline)) * self.cfg["bubble_sigma_disp"] * motion_scale
        sigma_per = (d_f / self.fx) * self.cfg["bubble_sigma_pix"] * motion_scale
        sp_max = xp.float32(float(self.cfg.get("bubble_sigma_par_max", 0.15)))
        sigma_par = xp.minimum(sigma_par, sp_max)
        diag_ray = xp.stack([sigma_per**2, sigma_per**2, sigma_par**2], axis=1)
        
        # Use CuPy utilities for vector operations
        rays = gpu_normalize_vectors(pts_cam)
        world_up = xp.array([0., 0., 1.])
        x_perp = gpu_cross_product(rays, world_up)
        x_perp_n = gpu_norm(x_perp, axis=1, keepdims=True)
        x_perp_alt = gpu_cross_product(rays, xp.array([0., 1., 0.]))
        x_perp = xp.where(x_perp_n < 1e-6, x_perp_alt, x_perp)
        x_perp /= (gpu_norm(x_perp, axis=1, keepdims=True) + 1e-9)
        y_perp = gpu_cross_product(rays, x_perp)
        y_perp /= (gpu_norm(y_perp, axis=1, keepdims=True) + 1e-9)
        R_rc = xp.stack([x_perp, y_perp, rays], axis=2)
        
        # Ensure diag_ray_3x3 is available for both try and except blocks
        # Create diag_ray_3x3 if not already defined
        if 'diag_ray_3x3' not in locals():
            # Create diagonal covariance matrices with shape (N, 3, 3)
            zeros = xp.zeros_like(sigma_per)
            diag_ray_3x3 = xp.stack([
                xp.stack([sigma_per**2, zeros, zeros], axis=1),
                xp.stack([zeros, sigma_per**2, zeros], axis=1), 
                xp.stack([zeros, zeros, sigma_par**2], axis=1)
            ], axis=1)
        
        diag_ray_3x3_local = diag_ray_3x3  # Create local reference
        
        try:
            R_wc_g = cupy_manager.to_gpu(pose[:3,:3])
            t_wc_g = cupy_manager.to_gpu(pose[:3,3])
            
            # Ensure R_rc has proper shape for batch operations
            if R_rc.shape[-2:] != (3, 3):
                raise ValueError(f"R_rc shape invalid: {R_rc.shape}")
            
            M = batch_matrix_multiply(R_wc_g, R_rc)
            
            # Ensure diag_ray_3x3 can be used for batch operations
            if len(diag_ray_3x3_local.shape) == 3 and diag_ray_3x3_local.shape[1:] == (3, 3):
                # Element-wise multiplication of batch matrices
                Md = M * diag_ray_3x3_local
            else:
                raise ValueError(f"diag_ray_3x3 shape invalid: {diag_ray_3x3_local.shape}")
                
            Sig_world = batch_matrix_multiply(Md, xp.swapaxes(M, 1, 2))
            
            # Use batch matrix operations for world transformation
            # Transform points: pts_world = (R @ pts_cam.T).T + t
            pts_world = (pose_gpu[:3,:3] @ pts_cam.T).T + pose_gpu[:3,3]
            
        except Exception as e:
            print(f" [BUBBLE DEBUG] Raw CUDA batch operation failed: {e}")
            # Fallback to regular CuPy operations
            R_wc = pose_gpu[:3,:3]
            t_wc = pose_gpu[:3,3]
            M = xp.einsum('ij,njk->nik', R_wc, R_rc)
            
            # Use the local diag_ray_3x3 reference
            Md = M * diag_ray_3x3_local  # Use 3x3 diagonal matrices
            Sig_world = Md @ xp.swapaxes(M, 1, 2)
            pts_world = (R_wc @ pts_cam.T).T + t_wc
        
        # Handle image colors with CuPy
        N_pts_g = pts_world.shape[0]
        if image_gpu is not None:
            img_g = image_gpu
            if img_g.dtype != xp.float32:
                img_g = img_g.astype(xp.float32)
            h_img, w_img = img_g.shape[:2]
            u_i_g = gpu_clip(u_f.astype(xp.int32), 0, w_img - 1)
            v_i_g = gpu_clip(v_f.astype(xp.int32), 0, h_img - 1)
            if img_g.ndim == 3:
                colors_gpu = img_g[v_i_g, u_i_g, :].astype(xp.float64) / 255.0
            else:
                c1 = (img_g[v_i_g, u_i_g].astype(xp.float64) / 255.0)[:, None]
                colors_gpu = xp.repeat(c1, 3, axis=1)
        else:
            colors_gpu = xp.full((N_pts_g, 3), 0.5, dtype=xp.float64)
        
        ones_gpu = xp.ones(N_pts_g, dtype=xp.float64)
        
        # Convert results back to CPU if needed
        pts_world_cpu = cupy_manager.to_cpu(pts_world)
        Sig_world_cpu = cupy_manager.to_cpu(Sig_world)
        ones_gpu_cpu = cupy_manager.to_cpu(ones_gpu)
        colors_gpu_cpu = cupy_manager.to_cpu(colors_gpu)
        
        return pts_world_cpu, Sig_world_cpu, ones_gpu_cpu, colors_gpu_cpu

    def _backproject_raw_cuda(self, depth, pose, image, stride, motion_scale=1.0):
        """Backproject depth using raw CUDA kernels for maximum performance."""
        # Prepare configuration for raw kernel
        config = {
            'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy,
            'baseline': self.baseline,
            'bubble_sigma_disp': self.cfg["bubble_sigma_disp"],
            'bubble_sigma_pix': self.cfg["bubble_sigma_pix"],
            'bubble_stride': stride,
            'motion_scale': motion_scale
        }
        
        # Use raw CUDA kernel
        result = cuda_kernel_manager.backproject_bubbles_raw(depth, pose, config)
        
        if result[0] is not None and len(result[0]) > 0:
            # Handle image colors with zero-copy if available
            if self.use_zero_copy and image is not None:
                # Create zero-copy shared buffer for image
                shared_image = zero_copy_manager.create_shared_buffer(image, 'backproject_image')
                # Extract colors using GPU operations
                colors = self._extract_colors_gpu(result[0], shared_image)
                result = (result[0], result[1], result[2], colors)
                self.zero_copy_transfers += 1
            
            return result
        
        return None, None, None, None

    def _extract_colors_gpu(self, points, image):
        """Extract colors using GPU operations for maximum performance."""
        if not USE_CUPY or image is None:
            return np.full((len(points), 3), 0.5)
        
        try:
            # Create lazy mirror for image
            image_mirror = lazy_mirror_manager.create_mirror(image, 'color_extraction')
            gpu_image = image_mirror.get_gpu_array()
            
            # Use GPU operations for color extraction
            points_gpu = cupy_manager.to_gpu(points)
            
            # Simple color extraction (can be enhanced with more sophisticated GPU operations)
            colors = xp.full((len(points), 3), 0.5, dtype=xp.float64)
            
            self.lazy_mirror_hits += 1
            return cupy_manager.to_cpu(colors)
            
        except Exception as e:
            logger.warning(f"GPU color extraction failed: {e}")
            return np.full((len(points), 3), 0.5)

    def fuse(self, new_mu, new_Sigma, new_weight, new_color):
        """Intelligent fusion of new Gaussian bubbles with spatial overlap detection."""
        if len(new_mu) == 0:
            return

        # Convert inputs to GPU if needed
        if self.use_gpu and not isinstance(new_mu, xp.ndarray):
            new_mu = cupy_manager.to_gpu(new_mu)
            new_Sigma = cupy_manager.to_gpu(new_Sigma)
            new_weight = cupy_manager.to_gpu(new_weight)
            new_color = cupy_manager.to_gpu(new_color)

        # Intelligent fusion: detect overlaps and merge
        if len(self.mu) == 0:
            self.mu = new_mu
            self.Sigma = new_Sigma
            self.weight = new_weight
            self.color = new_color
        else:
            # Apply intelligent spatial fusion
            fused_mu, fused_Sigma, fused_weight, fused_color = self._intelligent_fusion(
                self.mu, self.Sigma, self.weight, self.color,
                new_mu, new_Sigma, new_weight, new_color
            )
            
            self.mu = fused_mu
            self.Sigma = fused_Sigma
            self.weight = fused_weight
            self.color = fused_color

    def _intelligent_fusion(self, existing_mu, existing_Sigma, existing_weight, existing_color,
                           new_mu, new_Sigma, new_weight, new_color):
        """
        Perform intelligent spatial fusion of existing and new bubbles.
        
        Args:
            existing_mu, existing_Sigma, existing_weight, existing_color: Current bubble map
            new_mu, new_Sigma, new_weight, new_color: New bubbles to add
            
        Returns:
            Fused (mu, Sigma, weight, color)
        """
    def _intelligent_fusion(self, existing_mu, existing_Sigma, existing_weight, existing_color,
                           new_mu, new_Sigma, new_weight, new_color):
        """
        Perform intelligent spatial fusion of existing and new bubbles.
        Keeps data on GPU if possible to avoid D2H/H2D overhead.
        """
        # If not using GPU or not CuPy, fallback to CPU path
        if not self.use_gpu or not USE_CUPY:
            return self._intelligent_fusion_cpu(existing_mu, existing_Sigma, existing_weight, existing_color,
                                              new_mu, new_Sigma, new_weight, new_color)
        
        # GPU PATH: Use CuPy for concatenation and fusion
        try:
            # Ensure all are CuPy arrays (using zero-copy if they were Torch tensors)
            e_mu = cupy_manager.to_gpu(existing_mu)
            e_Sig = cupy_manager.to_gpu(existing_Sigma)
            e_w = cupy_manager.to_gpu(existing_weight)
            e_c = cupy_manager.to_gpu(existing_color)
            
            n_mu = cupy_manager.to_gpu(new_mu)
            n_Sig = cupy_manager.to_gpu(new_Sigma)
            n_w = cupy_manager.to_gpu(new_weight)
            n_c = cupy_manager.to_gpu(new_color)
            
            # Combine all bubbles on GPU
            all_mu = xp.concatenate([e_mu, n_mu], axis=0)
            all_Sigma = xp.concatenate([e_Sig, n_Sig], axis=0)
            all_weight = xp.concatenate([e_w, n_w], axis=0)
            all_color = xp.concatenate([e_c, n_c], axis=0)
            
            # Apply GPU-accelerated spatial fusion
            if len(all_mu) > 1000:
                return self._spatial_weighted_fusion_gpu(all_mu, all_Sigma, all_weight, all_color)
            else:
                return all_mu, all_Sigma, all_weight, all_color
                
        except Exception as e:
            logger.warning(f"GPU intelligent fusion failed, falling back to CPU: {e}")
            return self._intelligent_fusion_cpu(existing_mu, existing_Sigma, existing_weight, existing_color,
                                              new_mu, new_Sigma, new_weight, new_color)

    def _intelligent_fusion_cpu(self, existing_mu, existing_Sigma, existing_weight, existing_color,
                               new_mu, new_Sigma, new_weight, new_color):
        """CPU fallback for intelligent fusion."""
        # Convert everything to CPU/NumPy
        e_mu = to_numpy_safe(existing_mu)
        e_Sig = to_numpy_safe(existing_Sigma)
        e_w = to_numpy_safe(existing_weight)
        e_c = to_numpy_safe(existing_color)
        
        n_mu = to_numpy_safe(new_mu)
        n_Sig = to_numpy_safe(new_Sigma)
        n_w = to_numpy_safe(new_weight)
        n_c = to_numpy_safe(new_color)
        
        all_mu = np.concatenate([e_mu, n_mu], axis=0)
        all_Sigma = np.concatenate([e_Sig, n_Sig], axis=0)
        all_weight = np.concatenate([e_w, n_w], axis=0)
        all_color = np.concatenate([e_c, n_c], axis=0)
        
        if len(all_mu) > 1000:
            f_mu, f_Sig, f_w, f_c = self._spatial_weighted_fusion_cpu(all_mu, all_Sigma, all_weight, all_color)
        else:
            f_mu, f_Sig, f_w, f_c = all_mu, all_Sigma, all_weight, all_color
            
        # Convert back to GPU if requested
        if self.use_gpu:
            return cupy_manager.to_gpu(f_mu), cupy_manager.to_gpu(f_Sig), cupy_manager.to_gpu(f_w), cupy_manager.to_gpu(f_c)
        return f_mu, f_Sig, f_w, f_c
    
    def _spatial_weighted_fusion_gpu(self, mu, Sigma, weight, color):
        """GPU-accelerated spatial weighted fusion using vectorized CuPy operations."""
        xp = cupy_manager.get_array_module(True)
        cell_size = self.cfg.get("fusion_distance_threshold", 0.05)
        
        # 1. Assign each bubble to a voxel cell
        grid_coords = xp.floor(mu / cell_size).astype(xp.int32)
        OFFSET = 10000
        stride_y = 20001
        stride_z = 20001 * 20001
        cell_keys = ((grid_coords[:, 0] + OFFSET).astype(xp.int64) +
                     (grid_coords[:, 1] + OFFSET).astype(xp.int64) * stride_y +
                     (grid_coords[:, 2] + OFFSET).astype(xp.int64) * stride_z)
        
        # 2. Get unique cells and mapping
        unique_keys, labels = xp.unique(cell_keys, return_inverse=True)
        n_groups = len(unique_keys)
        
        # 3. Vectorized weighted accumulation
        # fused_val = sum(weight[i] * val[i]) / sum(weight[i])
        sum_w = xp.zeros(n_groups, dtype=xp.float64)
        xp.add.at(sum_w, labels, weight)
        sum_w += 1e-10 # epsilon
        
        fused_mu = xp.zeros((n_groups, 3), dtype=xp.float64)
        weighted_mu = mu * weight[:, None]
        xp.add.at(fused_mu, (labels, slice(None)), weighted_mu)
        fused_mu /= sum_w[:, None]
        
        fused_Sigma = xp.zeros((n_groups, 3, 3), dtype=xp.float64)
        weighted_Sigma = Sigma * weight[:, None, None]
        # CuPy add.at doesn't support multi-dimensional advanced indexing for Sigma as easily
        fused_Sigma_flat = xp.zeros((n_groups, 9), dtype=xp.float64)
        weighted_Sigma_flat = weighted_Sigma.reshape(-1, 9)
        xp.add.at(fused_Sigma_flat, (labels, slice(None)), weighted_Sigma_flat)
        fused_Sigma = (fused_Sigma_flat / sum_w[:, None]).reshape(-1, 3, 3)
        
        fused_weight = sum_w - 1e-10  # Define fused_weight
        fused_color = xp.zeros((n_groups, 3), dtype=xp.float64)
        weighted_color = color * weight[:, None]
        xp.add.at(fused_color, (labels, slice(None)), weighted_color)
        fused_color /= sum_w[:, None]
        return fused_mu, fused_Sigma, fused_weight, fused_color

    def _spatial_weighted_fusion_cpu(self, mu, Sigma, weight, color):
        """CPU implementation of spatial weighted fusion."""
        cell_size = self.cfg.get("fusion_distance_threshold", 0.05)
        
        # Assign each bubble to a voxel cell
        grid_coords = np.floor(mu / cell_size).astype(np.int32)
        OFFSET = 10000
        stride_y = 20001
        stride_z = 20001 * 20001
        cell_keys = ((grid_coords[:, 0] + OFFSET).astype(np.int64) +
                     (grid_coords[:, 1] + OFFSET).astype(np.int64) * stride_y +
                     (grid_coords[:, 2] + OFFSET).astype(np.int64) * stride_z)
        
        # Sort by cell key for contiguous grouping
        sort_order = np.argsort(cell_keys, kind='stable')
        sorted_keys = cell_keys[sort_order]
        sorted_mu     = mu[sort_order]
        sorted_Sigma  = Sigma[sort_order]
        sorted_weight = weight[sort_order]
        sorted_color  = color[sort_order]
        
        # Find group boundaries
        boundaries = np.concatenate(([0], np.where(np.diff(sorted_keys) != 0)[0] + 1, [len(sorted_keys)]))
        
        n_groups = len(boundaries) - 1
        fused_mu     = np.empty((n_groups, 3),    dtype=np.float64)
        fused_Sigma  = np.empty((n_groups, 3, 3), dtype=np.float64)
        fused_weight = np.empty((n_groups,),       dtype=np.float64)
        fused_color  = np.empty((n_groups, 3),    dtype=np.float64)
        
        for g in range(n_groups):
            s, e = boundaries[g], boundaries[g + 1]
            w_slice = sorted_weight[s:e]
            w_sum = w_slice.sum() + 1e-10
            w_norm = w_slice / w_sum
            
            fused_mu[g]     = (w_norm[:, None] * sorted_mu[s:e]).sum(axis=0)
            fused_Sigma[g]  = (w_norm[:, None, None] * sorted_Sigma[s:e]).sum(axis=0)
            fused_weight[g] = w_sum - 1e-10
            fused_color[g]  = (w_norm[:, None] * sorted_color[s:e]).sum(axis=0)
        
        return fused_mu, fused_Sigma, fused_weight, fused_color

        # Intelligent bubble management for room reconstruction
        max_bubbles = self.cfg.get("bubble_max_bubbles", 500000)  # Increased limit for better room coverage
        if len(self.mu) > max_bubbles * 1.5:  # Only prune when 50% over limit
            print(f" [BUBBLE] Over limit ({len(self.mu)} > {max_bubbles}), applying intelligent pruning")
            # Use intelligent pruning instead of simple capping
            scores = self._calculate_bubble_scores()
            if len(scores) > 0:  # Check if scores array is not empty
                keep_indices = np.argsort(scores)[-max_bubbles:]  # Keep highest scoring bubbles
                keep_indices = np.sort(keep_indices)
                
                self.mu = self.mu[keep_indices]
                self.Sigma = self.Sigma[keep_indices]
                self.weight = self.weight[keep_indices]
                self.color = self.color[keep_indices]
                print(f" [BUBBLE] Intelligent pruning complete: {len(self.mu)} bubbles remaining")
            else:
                print(f" [BUBBLE] Warning: Empty scores array, using simple capping")
                # Fallback to simple capping if scoring fails
                self.mu = self.mu[-max_bubbles:]
                self.Sigma = self.Sigma[-max_bubbles:]
                self.weight = self.weight[-max_bubbles:]
                self.color = self.color[-max_bubbles:]

    def prune(self):
        """Hierarchical bubble pruning with GPU acceleration and error handling."""
        if self.mu.shape[0] == 0:
            return
        
        try:
            if self.hierarchical_pruner is not None:
                # Use hierarchical pruning system
                print(f" [BUBBLE] Hierarchical pruning: {len(self.mu)} bubbles")
                
                # Apply three-level pruning
                mu_pruned, Sigma_pruned, weight_pruned, color_pruned = self.hierarchical_pruner.prune_bubbles(
                    self.mu, self.Sigma, self.weight, self.color
                )
                
                # Update bubble map
                self.mu = mu_pruned
                self.Sigma = Sigma_pruned
                self.weight = weight_pruned
                self.color = color_pruned
                
                # Print pruning statistics
                stats = self.hierarchical_pruner.get_statistics()
                print(f" [BUBBLE] Hierarchical pruning complete: {len(self.mu)} bubbles remaining")
                print(f" [BUBBLE] Pruned: {stats['local_pruned']} local, {stats['regional_merged']} regional, {stats['global_promoted']} promoted")
                
                # Update global map ages
                self.hierarchical_pruner.update_global_map_ages()
                
            else:
                # Fallback to simple pruning
                max_bubbles = self.cfg.get("bubble_max_bubbles", 500000)
                if len(self.mu) > max_bubbles:
                    print(f" [BUBBLE] Simple pruning: {len(self.mu)} -> {max_bubbles}")
                    scores = self._calculate_bubble_scores()
                    if len(scores) > 0:
                        keep_indices = np.argsort(scores)[-max_bubbles:]
                        self.mu = self.mu[keep_indices]
                        self.Sigma = self.Sigma[keep_indices]
                        self.weight = self.weight[keep_indices]
                        self.color = self.color[keep_indices]
                    else:
                        # Simple capping
                        self.mu = self.mu[-max_bubbles:]
                        self.Sigma = self.Sigma[-max_bubbles:]
                        self.weight = self.weight[-max_bubbles:]
                        self.color = self.color[-max_bubbles:]
                        
        except Exception as e:
            print(f"[PRUNE ERROR] Hierarchical pruning failed: {e}")
            # Emergency fallback - simple capping
            try:
                max_bubbles = self.cfg.get("bubble_max_bubbles", 500000)
                if len(self.mu) > max_bubbles:
                    print(f" [BUBBLE] Emergency pruning: {len(self.mu)} -> {max_bubbles}")
                    self.mu = self.mu[-max_bubbles:]
                    self.Sigma = self.Sigma[-max_bubbles:]
                    self.weight = self.weight[-max_bubbles:]
                    self.color = self.color[-max_bubbles:]
            except Exception as emergency_e:
                print(f"[PRUNE CRITICAL] Emergency pruning failed: {emergency_e}")
                # Last resort - clear everything but keep structure
                print(f" [BUBBLE] Critical: Clearing bubble map to prevent crash")
                self.mu = np.empty((0, 3), dtype=self.mu.dtype)
                self.Sigma = np.empty((0, 3, 3), dtype=self.Sigma.dtype)
                self.weight = np.empty(0, dtype=self.weight.dtype)
                self.color = np.empty((0, 3), dtype=self.color.dtype)

    def _calculate_bubble_scores(self):
        """Calculate confidence scores for bubbles using CuPy optimization."""
        if len(self.weight) == 0:
            return np.array([])
        
        # Use CuPy for calculations if available
        xp = cupy_manager.get_array_module(self.use_gpu)
        
        # Base score from weight
        scores = self.weight.copy()
        
        # Add uncertainty penalty (lower uncertainty = higher score)
        if len(self.Sigma) > 0:
            try:
                # Use batch matrix determinant calculation
                if self.use_gpu:
                    det_Sig = xp.linalg.det(self.Sigma)
                    uncertainty_penalty = -xp.log(xp.maximum(det_Sig, 1e-10))
                    scores += 0.1 * uncertainty_penalty
                else:
                    det_Sig = np.linalg.det(self.Sigma)
                    uncertainty_penalty = -np.log(np.maximum(det_Sig, 1e-10))
                    scores += 0.1 * uncertainty_penalty
            except Exception as e:
                # Fallback to simple weight scoring
                print(f" [BUBBLE] Score calculation error: {e}")
                pass
        
        # Convert back to CPU if needed
        if self.use_gpu and isinstance(scores, xp.ndarray):
            scores = cupy_manager.to_cpu(scores)
        
        return scores

    def update(self, depth: np.ndarray, pose: np.ndarray, image: np.ndarray,
               motion_scale: float = 1.0, keyframe_manager=None):
        """Update bubble map with a new depth frame and multi-view fusion."""
        mu, Sig, w, col = self.backproject_frame(
            depth, pose, image, self.cfg["bubble_stride"], motion_scale)
        if len(mu) == 0:
            return
        
        # Limit new bubbles to prevent explosion
        MAX_NEW_PER_FRAME = self.cfg.get("max_new_bubbles_per_frame", 20000)
        if mu.shape[0] > MAX_NEW_PER_FRAME:
            print(f" [BUBBLE] Limiting new bubbles: {mu.shape[0]} -> {MAX_NEW_PER_FRAME}")
            idx = np.random.choice(mu.shape[0], MAX_NEW_PER_FRAME, replace=False)
            mu = mu[idx]
            Sig = Sig[idx]
            w = w[idx]
            col = col[idx]
        
        print(f" [BUBBLE] Generated {len(mu)} new bubbles")
        
        # Apply multi-view fusion if keyframe manager is available
        if keyframe_manager is not None:
            # PURE GPU MODE: Handle GPU arrays properly
            if hasattr(mu, 'get'):
                # GPU case - convert to CPU for keyframe manager
                mu_cpu = cupy_manager.to_cpu(mu)
                Sig_cpu = cupy_manager.to_cpu(Sig) 
                w_cpu = cupy_manager.to_cpu(w)
                col_cpu = cupy_manager.to_cpu(col)
                current_bubbles = (mu_cpu, Sig_cpu, w_cpu, col_cpu)
                fused_bubbles = keyframe_manager.fuse_multi_view_bubbles(current_bubbles, pose)
                # Convert back to GPU if needed
                if USE_CUPY:
                    mu = cupy_manager.to_gpu(fused_bubbles[0])
                    Sig = cupy_manager.to_gpu(fused_bubbles[1])
                    w = cupy_manager.to_gpu(fused_bubbles[2])
                    col = cupy_manager.to_gpu(fused_bubbles[3])
                else:
                    mu, Sig, w, col = fused_bubbles
            else:
                # CPU case
                current_bubbles = (mu, Sig, w, col)
                fused_bubbles = keyframe_manager.fuse_multi_view_bubbles(current_bubbles, pose)
                mu, Sig, w, col = fused_bubbles
            mu, Sig, w, col = fused_bubbles
            print(f" [BUBBLE] Multi-view fusion: {len(current_bubbles[0])} -> {len(mu)} bubbles")
        
        with self._lock:
            self.fuse(mu, Sig, w, col)
            total_bubbles = len(self.mu)

            # Enforce hard limits for stability
            MAX_GLOBAL_BUBBLES = self.cfg.get("max_global_bubbles", 500000)
            if self.mu.shape[0] > MAX_GLOBAL_BUBBLES:
                print(f"[BUBBLE] Over limit ({self.mu.shape[0]} > {MAX_GLOBAL_BUBBLES}), pruning...")
                self.prune()
                total_bubbles = len(self.mu)

        print(f" [BUBBLE] Total bubbles after fusion: {total_bubbles}")
        
        self.frame_counter += 1
        if self.frame_counter % 50 == 0:
            with self._lock:
                total_bubbles = len(self.mu)
            print(f" [BUBBLE] Map size: {total_bubbles} bubbles")

    def reintegrate_map(self, keyframes=None):
        """
        Reintegrate map after loop closure or global optimization.
        For the bubble map, we clear active bubbles to allow the system
        to re-populate from the corrected keyframe poses.
        """
        with self._lock:
            xp = cupy_manager.get_array_module(self.use_gpu)
            self.mu = xp.empty((0, 3), dtype=xp.float64)
            self.Sigma = xp.empty((0, 3, 3), dtype=xp.float64)
            self.weight = xp.empty(0, dtype=xp.float64)
            self.color = xp.empty((0, 3), dtype=xp.float64)
            logger.info("Bubble map active buffers cleared for reintegration")

    def get_global_map_bubbles(self):
        """Get bubbles from persistent global map."""
        if self.hierarchical_pruner is not None:
            return self.hierarchical_pruner.get_global_map_bubbles()
        return None, None, None, None

    def get_full_point_cloud(self, keyframe_manager=None):
        """Return complete bubble map: active bubbles + persistent global map combined, filtered by weight."""
        all_mu_parts    = []
        all_color_parts = []
        all_weight_parts = []
        
        # Use the visualization threshold to drop low-confidence/noisy bubbles
        thr = self.cfg.get("bubble_visualization_threshold", 0.2)

        # 1. Primary: current active bubble map
        if len(self.mu) > 0:
            mu_np     = self.mu.get()     if hasattr(self.mu,     'get') else self.mu.copy()
            color_np  = self.color.get()  if hasattr(self.color,  'get') else self.color.copy()
            weight_np = self.weight.get() if hasattr(self.weight, 'get') else self.weight.copy()
            
            valid = weight_np >= thr
            if np.any(valid):
                all_mu_parts.append(mu_np[valid].astype(np.float32))
                all_color_parts.append(color_np[valid].astype(np.float32))
                all_weight_parts.append(weight_np[valid].astype(np.float32))

        # 2. Persistent global map (bubbles promoted by hierarchical pruner)
        if self.hierarchical_pruner is not None:
            try:
                g_mu, g_Sigma, g_weight, g_color = self.hierarchical_pruner.get_global_map_bubbles()
                if g_mu is not None and len(g_mu) > 0:
                    g_mu_np     = g_mu.get()     if hasattr(g_mu,     'get') else np.asarray(g_mu,     dtype=np.float32)
                    g_color_np  = g_color.get()  if hasattr(g_color,  'get') else np.asarray(g_color,  dtype=np.float32)
                    g_weight_np = g_weight.get() if hasattr(g_weight, 'get') else np.asarray(g_weight, dtype=np.float32)
                    
                    valid = g_weight_np >= thr
                    if np.any(valid):
                        all_mu_parts.append(g_mu_np[valid].astype(np.float32))
                        all_color_parts.append(g_color_np[valid].astype(np.float32))
                        all_weight_parts.append(g_weight_np[valid].astype(np.float32))
            except Exception:
                pass

        # 3. Keyframe global reconstruction (highest quality)
        if keyframe_manager is not None:
            try:
                global_mu, _, global_weight, global_color = keyframe_manager.get_global_reconstruction()
                if len(global_mu) > 0:
                    kf_mu     = global_mu.get()     if hasattr(global_mu,     'get') else np.asarray(global_mu,     dtype=np.float32)
                    kf_color  = global_color.get()  if hasattr(global_color,  'get') else np.asarray(global_color,  dtype=np.float32)
                    kf_weight = global_weight.get() if hasattr(global_weight, 'get') else np.asarray(global_weight, dtype=np.float32)
                    
                    valid = kf_weight >= thr
                    if np.any(valid):
                        all_mu_parts.append(kf_mu[valid].astype(np.float32))
                        all_color_parts.append(kf_color[valid].astype(np.float32))
                        all_weight_parts.append(kf_weight[valid].astype(np.float32))
            except Exception:
                pass

        if not all_mu_parts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        combined_mu    = np.concatenate(all_mu_parts,    axis=0)
        combined_color = np.concatenate(all_color_parts, axis=0)
        combined_weight = np.concatenate(all_weight_parts, axis=0)
        return combined_mu, combined_color, combined_weight

    # Open3D removed - use get_point_cloud_pyvista only

    def get_point_cloud_pyvista(self, max_points=80000):
        """Return (points, colors) as NumPy arrays for PyVista visualization using CuPy."""
        with self._lock:
            mu = self.mu
            weight = self.weight
            color = self.color

        if len(mu) == 0:
            return None, None
        
        # Use CuPy for calculations if available
        xp = cupy_manager.get_array_module(self.use_gpu)
        thr = self.cfg.get("bubble_visualization_threshold", 0.05)
        valid = weight >= thr
        
        if not xp.any(valid):
            return None, None
        
        mu_f = mu[valid]
        col_f = color[valid]
        
        if len(mu_f) > max_points:
            # Use CuPy for random sampling
            if self.use_gpu:
                idx = xp.random.choice(len(mu_f), max_points, replace=False)
            else:
                idx = np.random.choice(len(mu_f), max_points, replace=False)
            mu_f = mu_f[idx]
            col_f = col_f[idx]
        
        # Convert to CPU for PyVista
        if self.use_gpu:
            mu_f = cupy_manager.to_cpu(mu_f)
            col_f = cupy_manager.to_cpu(col_f)
        col_f = (col_f * 255).astype(np.uint8)
        return mu_f, col_f

    def shutdown(self):
        """Shutdown the threaded manager."""
        if self.threaded_manager:
            self.threaded_manager.shutdown()
