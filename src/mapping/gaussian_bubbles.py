"""Chunked Gaussian bubble map for large-scale SLAM with local float32 precision."""

import numpy as np
import threading
import time
import os
import pickle
from queue import Queue, Empty
from collections import deque
from scipy.spatial import cKDTree
from .spatial_hash import SpatialHashGrid
from .hierarchical_pruner import HierarchicalPruner
from ..utils.logger import get_logger
from ..utils.cupy_utils import cp, USE_CUPY, to_numpy_safe, rotate_covariance_batch
from ..utils.se3_ops import PoseTransform
from ..core.data_model import GaussianBubbleBatch
from ..core.array_backend import get_array_module, is_gpu_array

logger = get_logger(__name__)

def _truth(value):
    """Return a Python bool for NumPy/CuPy scalar reductions."""
    try:
        if hasattr(value, "item"):
            return bool(value.item())
    except Exception:
        pass
    return bool(value)

def _fuse_gaussians_cpu(base_mu, base_sigma, base_weight, base_color,
                        new_mu, new_sigma, new_weight, new_color,
                        radius=0.05, mahal_thresh=9.0,
                        max_size=5000, voxel_size=None):
    """Deterministic bounded Gaussian fusion using CPU KDTree association.

    The GPU hash kernels are fast but unstable under heavy duplication and bucket
    overflow. This stage makes reconstruction fusion-first before any chunk write.
    """
    base_mu = np.asarray(base_mu, dtype=np.float32).reshape(-1, 3)
    base_sigma = np.asarray(base_sigma, dtype=np.float32).reshape(-1, 3, 3)
    base_weight = np.asarray(base_weight, dtype=np.float32).reshape(-1)
    base_color = np.asarray(base_color, dtype=np.float32).reshape(-1, 3)

    new_mu = np.asarray(new_mu, dtype=np.float32).reshape(-1, 3)
    new_sigma = np.asarray(new_sigma, dtype=np.float32).reshape(-1, 3, 3)
    new_weight = np.asarray(new_weight, dtype=np.float32).reshape(-1)
    new_color = np.asarray(new_color, dtype=np.float32).reshape(-1, 3)

    if len(new_mu) == 0:
        return base_mu, base_sigma, base_weight, base_color

    finite = (
        np.all(np.isfinite(new_mu), axis=1) &
        np.all(np.isfinite(new_sigma.reshape(len(new_sigma), -1)), axis=1) &
        np.isfinite(new_weight) & (new_weight > 1e-4) &
        np.all(np.isfinite(new_color), axis=1)
    )
    if not np.any(finite):
        return base_mu, base_sigma, base_weight, base_color

    new_mu = new_mu[finite]
    new_sigma = new_sigma[finite]
    new_weight = new_weight[finite]
    new_color = np.clip(new_color[finite], 0.0, 1.0)

    if voxel_size is None:
        voxel_size = radius

    # Frame-local deduplication: keep the highest-confidence sample per voxel.
    if len(new_mu) > 1 and voxel_size > 0:
        vox = np.floor(new_mu / voxel_size).astype(np.int64)
        order = np.lexsort((vox[:, 2], vox[:, 1], vox[:, 0], -new_weight))
        vox_sorted = vox[order]
        keep_sorted = np.ones(len(order), dtype=bool)
        keep_sorted[1:] = np.any(vox_sorted[1:] != vox_sorted[:-1], axis=1)
        keep = order[keep_sorted]
        new_mu = new_mu[keep]
        new_sigma = new_sigma[keep]
        new_weight = new_weight[keep]
        new_color = new_color[keep]

    # Deterministic order avoids run-to-run drift from thread scheduling.
    order = np.lexsort((new_mu[:, 2], new_mu[:, 1], new_mu[:, 0]))
    new_mu = new_mu[order]
    new_sigma = new_sigma[order]
    new_weight = new_weight[order]
    new_color = new_color[order]

    mu = base_mu.copy()
    sigma = base_sigma.copy()
    weight = base_weight.copy()
    color = base_color.copy()

    eps = 1e-6
    radius = float(max(radius, 1e-4))
    mahal_thresh = float(max(mahal_thresh, 1.0))

    base_len = len(mu)
    tree = cKDTree(mu) if base_len > 0 else None
    neighbor_lists = tree.query_ball_point(new_mu, r=radius) if tree is not None else [[] for _ in range(len(new_mu))]

    for i, (p, s, w, c) in enumerate(zip(new_mu, new_sigma, new_weight, new_color)):
        target = -1
        if tree is not None:
            best_score = mahal_thresh
            for idx in neighbor_lists[i]:
                d = p - mu[idx]
                cov = sigma[idx] + s + np.eye(3, dtype=np.float32) * eps
                try:
                    score = float(d @ np.linalg.solve(cov, d))
                except np.linalg.LinAlgError:
                    score = float(d @ d) / max(float(np.trace(cov)) / 3.0, eps)
                if score < best_score:
                    best_score = score
                    target = int(idx)

        if target >= 0:
            old_w = float(weight[target])
            fused_w = min(old_w + float(w), 255.0)
            alpha_old = old_w / max(old_w + float(w), eps)
            alpha_new = float(w) / max(old_w + float(w), eps)
            old_mu = mu[target].copy()
            fused_mu = alpha_old * old_mu + alpha_new * p
            d_old = old_mu - fused_mu
            d_new = p - fused_mu
            fused_sigma = (
                alpha_old * (sigma[target] + np.outer(d_old, d_old)) +
                alpha_new * (s + np.outer(d_new, d_new))
            )
            fused_sigma = 0.5 * (fused_sigma + fused_sigma.T)
            trace = float(np.trace(fused_sigma))
            if trace > 0.75:
                fused_sigma *= 0.75 / max(trace, eps)
            mu[target] = fused_mu.astype(np.float32)
            sigma[target] = (fused_sigma + np.eye(3, dtype=np.float32) * eps).astype(np.float32)
            color[target] = (alpha_old * color[target] + alpha_new * c).astype(np.float32)
            weight[target] = fused_w
        elif len(mu) < max_size:
            mu = np.vstack([mu, p[None]])
            sigma = np.concatenate([sigma, s[None]], axis=0)
            weight = np.concatenate([weight, np.asarray([min(float(w), 255.0)], dtype=np.float32)])
            color = np.vstack([color, c[None]])

    if len(mu) > max_size:
        keep = np.argsort(weight)[-max_size:]
        keep.sort()
        mu, sigma, weight, color = mu[keep], sigma[keep], weight[keep], color[keep]

    return mu.astype(np.float32), sigma.astype(np.float32), weight.astype(np.float32), color.astype(np.float32)

class MapChunk:
    """A 4m x 4m x 4m chunk of the world storing bubbles in local coordinates."""
    def __init__(self, origin, cell_size=0.02, use_gpu=True, max_bubbles=10000):
        self.origin = origin.astype(np.float32)  # [3]
        
        # Fixed-size buffers for GPU-efficient fusion (Section E)
        # V58: Further reduced max_bubbles per chunk to 10k for extreme stability
        self.max_bubbles = max_bubbles
        xp = get_array_module(use_gpu=(use_gpu and USE_CUPY))
        
        self.mu_local = xp.zeros((max_bubbles, 3), dtype=xp.float32)
        self.Sigma = xp.zeros((max_bubbles, 3, 3), dtype=xp.float32)
        self.weight = xp.zeros((max_bubbles,), dtype=xp.float32)
        self.color = xp.zeros((max_bubbles, 3), dtype=xp.float32)
        
        if use_gpu and USE_CUPY:
            self.current_size = cp.zeros(1, dtype=cp.int32)
            # V56: Defer hash table allocation to save VRAM until needed for fusion
            self.hash_table = None 
        else:
            self.current_size = 0
            # CPU-side KDTree for safer fallback fusion (Fix 6)
            self.kdtree = None
            
        self.spatial_grid = SpatialHashGrid(cell_size=cell_size, table_size=262144, use_gpu=use_gpu)
        self.use_gpu = use_gpu and USE_CUPY
        self.cell_size = cell_size
        self.last_used_frame = 0

    def __len__(self):
        if self.use_gpu:
            return int(self.current_size[0])
        return self.current_size

    def add_bubbles(self, local_pts, weights, colors, Sigmas, stabilization_mode=False):
        """Add and fuse bubbles using GPU bucketed hash or CPU KDTree fallback (Fix 6)."""
        batch = GaussianBubbleBatch.from_arrays(local_pts, weights, colors, Sigmas, to_numpy=to_numpy_safe)
        local_pts, weights, colors, Sigmas = batch.mu, batch.weight, batch.color, batch.Sigma
        num_new = len(local_pts)
        if num_new == 0:
            return
            
        # Reconstruction-first stability: deterministic KDTree fusion instead
        # of append-heavy stabilization or fragile GPU hash insertion.
        xp = get_array_module(use_gpu=(self.use_gpu and USE_CUPY))
        curr = int(self.current_size[0]) if (self.use_gpu and USE_CUPY) else self.current_size
        mu, sig, w, col = _fuse_gaussians_cpu(
            to_numpy_safe(self.mu_local[:curr]),
            to_numpy_safe(self.Sigma[:curr]),
            to_numpy_safe(self.weight[:curr]),
            to_numpy_safe(self.color[:curr]),
            to_numpy_safe(local_pts),
            to_numpy_safe(Sigmas),
            to_numpy_safe(weights),
            to_numpy_safe(colors),
            radius=max(self.cell_size * 2.5, 0.05),
            mahal_thresh=9.0,
            max_size=self.max_bubbles,
            voxel_size=max(self.cell_size, 0.03),
        )
        new_len = len(mu)
        self.mu_local[:new_len] = xp.asarray(mu)
        self.Sigma[:new_len] = xp.asarray(sig)
        self.weight[:new_len] = xp.asarray(w)
        self.color[:new_len] = xp.asarray(col)
        if (self.use_gpu and USE_CUPY):
            self.current_size.fill(new_len)
        else:
            self.current_size = new_len

    def prune(self, min_weight, max_bubbles_target, radius_outlier_params):
        """Prune low-confidence, redundant, and degenerate bubbles."""
        curr_len = len(self)
        if curr_len == 0:
            return
            
        xp = get_array_module(use_gpu=self.use_gpu)
        mu = self.mu_local[:curr_len]
        sig = self.Sigma[:curr_len]
        w = self.weight[:curr_len]
        col = self.color[:curr_len]
        
        # 1. Quality Filters
        mask = (w > min_weight)
        
        # Degenerate removal (trace > 0.5m means extreme uncertainty)
        trace = xp.trace(sig, axis1=1, axis2=2)
        mask &= (trace < 0.5) & xp.isfinite(trace)
        
        if not _truth(xp.any(mask)):
            if self.use_gpu:
                self.current_size.fill(0)
                if self.hash_table is not None:
                    self.hash_table.fill(-1)
            else:
                self.current_size = 0
            return
            
        mu, sig, w, col = mu[mask], sig[mask], w[mask], col[mask]
        
        # 2. Limit bubbles per chunk (keep highest weighted)
        if len(mu) > max_bubbles_target:
            indices = xp.argsort(w)[-max_bubbles_target:]
            mu, sig, w, col = mu[indices], sig[indices], w[indices], col[indices]
            
        # 3. Radius Outlier Removal
        if len(mu) > radius_outlier_params['min_neighbors']:
            self.spatial_grid.build(mu)
            valid_mask = self.spatial_grid.radius_outlier_removal(
                min_neighbors=radius_outlier_params['min_neighbors'],
                radius=radius_outlier_params['radius']
            )
            mu, sig, w, col = mu[valid_mask], sig[valid_mask], w[valid_mask], col[valid_mask]

        # Update buffers
        new_len = len(mu)
        self.mu_local[:new_len] = mu
        self.Sigma[:new_len] = sig
        self.weight[:new_len] = w
        self.color[:new_len] = col
        
        if self.use_gpu:
            self.current_size.fill(new_len)
            # Rebuild hash table after pruning
            if self.hash_table is not None:
                self.hash_table.fill(-1)
        else:
            self.current_size = new_len

    def get_lod_cloud(self, factor=0.5):
        """Returns a lower-detail version of the chunk's bubbles."""
        curr_len = len(self)
        if curr_len == 0:
            return None, None, None
            
        xp = cp if self.use_gpu else np
        n_target = max(int(curr_len * factor), 1)
        
        # Subsample by weight (keep most important)
        indices = xp.argsort(self.weight[:curr_len])[-n_target:]
        
        return (
            self.mu_local[indices] + xp.asarray(self.origin),
            self.color[indices],
            self.weight[indices]
        )

class ThreadedBubbleMapManager:
    """Async reconstruction manager for the bubble map (V52).
    
    Handles depth backprojection, Gaussian fusion, and visualization updates
    in a dedicated thread to prevent blocking the SLAM tracking loop.
    """
    def __init__(self, bubble_map):
        self.map = bubble_map
        self.queue = Queue(maxsize=10)
        self.running = False
        self.thread = None
        self._lock = threading.Lock()
        
        # State tracking
        self.last_process_time = 0
        self.processed_frames = 0
        
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info("ThreadedBubbleMapManager started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def submit(self, depth, pose, image, stride=4, is_initializing=False):
        """Submit a new frame for reconstruction."""
        try:
            # Drop frame if queue is too full to avoid latency buildup
            if self.queue.full():
                try: self.queue.get_nowait()
                except Empty: pass
            
            # Use stabilization mode during initialization
            stab = is_initializing or self.map.stabilization_mode
            self.queue.put_nowait((depth, pose, image, stride, stab))
            return True
        except Exception:
            return False

    def _worker_loop(self):
        while self.running:
            try:
                # Get next frame to process
                depth, pose, image, stride, stab = self.queue.get(timeout=0.1)
                
                start_t = time.time()
                
                orig_stab = self.map.stabilization_mode
                self.map.stabilization_mode = stab
                try:
                    # 1. Backproject
                    pts, weights, colors, sigmas = self.map.backproject_frame(
                        depth, pose, image, stride, motion_scale=1.0
                    )
                finally:
                    self.map.stabilization_mode = orig_stab
                
                if len(pts) > 0:
                    # 2. Validate and add to map through the shared data model.
                    bubbles = GaussianBubbleBatch.from_arrays(
                        pts, weights, colors, sigmas, to_numpy=to_numpy_safe
                    )
                    self.map.add_bubbles(bubbles)
                    
                    # 3. Trigger visualization update periodically or if forced
                    self.map.push_to_visualizer()
                    
                self.last_process_time = time.time() - start_t
                self.processed_frames += 1
                
                # Yield slightly
                time.sleep(0.001)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"ThreadedBubbleMapManager error: {e}")
                time.sleep(0.01)

class ChunkedBubbleMap:
    """Scalable bubble map using chunks and spatial indexing."""
    def __init__(self, K, baseline, config):
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
        self.baseline = baseline
        self.cfg = config
        
        self.chunk_size = float(config.get('chunk_size', 4.0))
        self.cell_size = float(config.get('fusion_distance_threshold', 0.02))
        self.chunks = {}  # Dict[int, MapChunk]
        self.active_chunks = set()
        # V56: Reduced default active radius to 2 to save VRAM (from 3)
        self.active_radius = config.get('active_radius_chunks', 2) 
        
        self.use_gpu = config.get('bubble_cuda', True) and USE_CUPY
        self.frame_counter = 0
        self.last_pose = None
        
        # Throttling and Visualization
        self.max_new_bubbles = config.get('max_new_bubbles_per_frame', 20000)
        self.bubble_stride = config.get('bubble_stride', 4)
        self._viz_queue = Queue(maxsize=1)
        self.viz_max_pts = config.get('max_visualisation_points', 80000)
        
        # ── V41: HIERARCHICAL PRUNER ──
        self.pruner = HierarchicalPruner(config)
        
        self._lock = threading.Lock()
        
        # ── V39: ASYNC VIZ WORKER ──
        self._viz_thread = None
        self._viz_running = False
        self._viz_update_needed = threading.Event()
        
        # ── V40: LOCAL MONOLITHIC BUFFER (Stability) ──
        # V58: Reduced local buffer to 5k to prevent VRAM exhaustion during commits
        self.local_buffer_size = config.get('local_bubble_buffer_size', 5000)
        xp = get_array_module(use_gpu=self.use_gpu)
        self.local_mu = xp.zeros((self.local_buffer_size, 3), dtype=xp.float32)
        self.local_Sigma = xp.zeros((self.local_buffer_size, 3, 3), dtype=xp.float32)
        self.local_weight = xp.zeros((self.local_buffer_size,), dtype=xp.float32)
        self.local_color = xp.zeros((self.local_buffer_size, 3), dtype=xp.float32)
        self.local_count = xp.zeros(1, dtype=xp.int32) if self.use_gpu else 0
        
        # Local spatial hash for stable fusion
        self.local_spatial_grid = SpatialHashGrid(cell_size=0.05, use_gpu=self.use_gpu)
        if self.use_gpu:
            self.local_hash_table = cp.full(262144 * 4, -1, dtype=cp.int32)
        
        # Caching for properties
        self._cached_mu = None
        self._cached_weight = None
        self._cached_color = None
        self._cached_Sigma = None
        self._cache_valid = False
        
        # Disk Offloading
        self.enable_offloading = config.get('enable_chunk_offloading', False)
        self.max_resident_chunks = config.get('max_resident_chunks', 100)
        self.offload_dir = config.get('chunk_offload_dir', 'map_chunks')
        if self.enable_offloading and not os.path.exists(self.offload_dir):
            os.makedirs(self.offload_dir)
            
        # V51: Stabilization Mode
        self.stabilization_mode = config.get('stabilization_mode', True)
        if self.stabilization_mode:
            logger.info("ChunkedBubbleMap operating in STABILIZATION MODE (fusion-first)")
            
        # ── V52: THREADED RECONSTRUCTION MANAGER ──
        self.threaded_manager = ThreadedBubbleMapManager(self)
        self.threaded_manager.start()
            
        self._start_viz_worker()
        logger.info(f"ChunkedBubbleMap initialized (chunk_size={self.chunk_size}m, local_buffer={self.local_buffer_size})")

    def _start_viz_worker(self):
        self._viz_running = True
        self._viz_thread = threading.Thread(target=self._viz_worker_loop, daemon=True)
        self._viz_thread.start()

    def _viz_worker_loop(self):
        """Background thread that builds the point cloud for visualization."""
        while self._viz_running:
            if self._viz_update_needed.wait(timeout=0.1):
                self._viz_update_needed.clear()
                pts, cols, scales = self.build_visualisation_cloud(max_pts=self.viz_max_pts)
                if pts is not None:
                    try:
                        # Non-blocking push to the viz queue
                        while not self._viz_queue.empty():
                            try: self._viz_queue.get_nowait()
                            except Empty: break
                        self._viz_queue.put_nowait((pts, cols, scales))
                    except Exception:
                        pass
            time.sleep(0.01)

    def _offload_chunk(self, key):
        """Offload chunk to disk and remove from VRAM."""
        if key not in self.chunks:
            return
            
        chunk = self.chunks[key]
        path = os.path.join(self.offload_dir, f"chunk_{key}.npz")
        
        # Build CPU-side arrays for saving
        n = len(chunk)
        if n == 0:
            del self.chunks[key]
            return

        # Save as compressed numpy
        try:
            np.savez_compressed(
                path,
                mu_local=to_numpy_safe(chunk.mu_local[:n]),
                Sigma=to_numpy_safe(chunk.Sigma[:n]),
                weight=to_numpy_safe(chunk.weight[:n]),
                color=to_numpy_safe(chunk.color[:n]),
                origin=chunk.origin
            )
            logger.info(f" [OFFLOAD] Chunk {key} saved to disk ({n} pts)")
        except Exception as e:
            logger.error(f" [OFFLOAD] Failed to save chunk {key}: {e}")
            
        del self.chunks[key]

    def _reload_chunk(self, key):
        """Reload chunk from disk if it exists."""
        if not self.enable_offloading:
            return None
        path = os.path.join(self.offload_dir, f"chunk_{key}.npz")
        if not os.path.exists(path):
            return None
            
        try:
            data = np.load(path)
            origin = data['origin']
            chunk = MapChunk(origin, cell_size=self.cell_size, use_gpu=self.use_gpu)
            
            xp = cp if self.use_gpu else np
            mu_cpu = data['mu_local']
            n = len(mu_cpu)
            
            # Re-upload to GPU if needed
            chunk.mu_local[:n] = xp.asarray(mu_cpu)
            chunk.Sigma[:n] = xp.asarray(data['Sigma'])
            chunk.weight[:n] = xp.asarray(data['weight'])
            chunk.color[:n] = xp.asarray(data['color'])
            
            if self.use_gpu:
                chunk.current_size.fill(n)
                if chunk.hash_table is not None:
                    chunk.hash_table.fill(-1)
            else:
                chunk.current_size = n
                
            logger.info(f" [RELOAD] Chunk {key} restored from disk ({n} pts)")
            return chunk
        except Exception as e:
            logger.error(f" [RELOAD] Failed to load chunk {key}: {e}")
            return None

    def _manage_memory(self):
        """Offload least recently used inactive chunks if limit reached and free VRAM."""
        # 1. Periodic VRAM defragmentation (Crucial for 6GB GPUs)
        if self.use_gpu and self.frame_counter % 30 == 0:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception: pass
            
        if not self.enable_offloading or len(self.chunks) <= self.max_resident_chunks:
            return
            
        # Candidates for offloading: not active and not recently used
        inactive = [k for k in self.chunks.keys() if k not in self.active_chunks]
        if not inactive:
            return
            
        # Sort by last_used_frame
        inactive.sort(key=lambda k: self.chunks[k].last_used_frame)
        
        num_to_offload = len(self.chunks) - self.max_resident_chunks
        for i in range(min(num_to_offload, len(inactive))):
            self._offload_chunk(inactive[i])

    def _invalidate_cache(self):
        self._cache_valid = False
        self._cached_mu = None
        self._cached_weight = None
        self._cached_color = None
        self._cached_Sigma = None

    @property
    def mu(self):
        """Property for backward compatibility - returns concatenated mu from local buffer and chunks."""
        if self._cache_valid and self._cached_mu is not None:
            return self._cached_mu
            
        xp = cp if self.use_gpu else np
        all_mu = []
        
        with self._lock:
            # 1. Local buffer
            n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
            if n_local > 0:
                all_mu.append(self.local_mu[:n_local])
                
            # 2. Chunks
            for chunk in self.chunks.values():
                n = len(chunk)
                if n > 0:
                    all_mu.append(chunk.mu_local[:n] + xp.asarray(chunk.origin))
        
        if not all_mu:
            self._cached_mu = xp.empty((0, 3), dtype=xp.float32)
        else:
            self._cached_mu = xp.concatenate(all_mu, axis=0)
            
        return self._cached_mu

    @property
    def weight(self):
        """Property for backward compatibility."""
        if self._cache_valid and self._cached_weight is not None:
            return self._cached_weight
            
        xp = cp if self.use_gpu else np
        all_w = []
        
        with self._lock:
            # 1. Local buffer
            n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
            if n_local > 0:
                all_w.append(self.local_weight[:n_local])
                
            # 2. Chunks
            for chunk in self.chunks.values():
                n = len(chunk)
                if n > 0:
                    all_w.append(chunk.weight[:n])
                
        if not all_w:
            self._cached_weight = xp.empty((0,), dtype=xp.float32)
        else:
            self._cached_weight = xp.concatenate(all_w, axis=0)
        return self._cached_weight

    @property
    def color(self):
        """Property for backward compatibility."""
        if self._cache_valid and self._cached_color is not None:
            return self._cached_color
            
        xp = cp if self.use_gpu else np
        all_c = []
        
        with self._lock:
            # 1. Local buffer
            n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
            if n_local > 0:
                all_c.append(self.local_color[:n_local])
                
            # 2. Chunks
            for chunk in self.chunks.values():
                n = len(chunk)
                if n > 0:
                    all_c.append(chunk.color[:n])
                
        if not all_c:
            self._cached_color = xp.empty((0, 3), dtype=xp.float32)
        else:
            self._cached_color = xp.concatenate(all_c, axis=0)
        return self._cached_color

    @property
    def Sigma(self):
        """Property for backward compatibility."""
        if self._cache_valid and self._cached_Sigma is not None:
            return self._cached_Sigma
            
        xp = cp if self.use_gpu else np
        all_s = []
        
        with self._lock:
            # 1. Local buffer
            n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
            if n_local > 0:
                all_s.append(self.local_Sigma[:n_local])
                
            # 2. Chunks
            for chunk in self.chunks.values():
                n = len(chunk)
                if n > 0:
                    all_s.append(chunk.Sigma[:n])
                
        if not all_s:
            self._cached_Sigma = xp.empty((0, 3, 3), dtype=xp.float32)
        else:
            self._cached_Sigma = xp.concatenate(all_s, axis=0)
        
        return self._cached_Sigma

    def validate_cache(self):
        """Manually mark cache as valid if all properties have been computed."""
        self._cache_valid = True

    def clear(self):
        """Clear all chunks and reset cache."""
        with self._lock:
            self.chunks.clear()
            self.active_chunks.clear()
            if self.use_gpu:
                self.local_count.fill(0)
                if self.local_hash_table is not None:
                    self.local_hash_table.fill(-1)
            else:
                self.local_count = 0
            self._invalidate_cache()
            
    def load_bubbles(self, mu, weight, color, Sigma):
        """Load bubbles from external data into the chunked structure."""
        self.clear()
        self.add_bubbles(GaussianBubbleBatch.from_arrays(mu, weight, color, Sigma, to_numpy=to_numpy_safe))

    def __len__(self):
        """Total number of bubbles in both local buffer and chunks (Safe V49)."""
        try:
            with self._lock:
                n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
                n_chunks = sum(len(chunk) for chunk in self.chunks.values())
                return max(0, int(n_local + n_chunks))
        except Exception:
            return 0

    def get_full_point_cloud(self):
        """Returns (mu, color, weight) for all bubbles."""
        return self.mu, self.color, self.weight

    def _world_to_chunk_key(self, pts):
        """Convert world points to 64-bit chunk keys (V59 Fix)."""
        xp = cp if (self.use_gpu and is_gpu_array(pts)) else np
        pts = xp.asarray(pts)
        # V59: Use int64 explicitly to prevent bit-shift overflow in CuPy/NumPy
        coords = xp.floor(pts / self.chunk_size).astype(xp.int64)
        OFFSET = 1000000
        return (coords[:, 0] + OFFSET) | ((coords[:, 1] + OFFSET) << 21) | ((coords[:, 2] + OFFSET) << 42)

    def _key_to_origin(self, key):
        """Convert integer chunk key back to world origin."""
        OFFSET = 1000000
        x = (key & 0x1FFFFF) - OFFSET
        y = ((key >> 21) & 0x1FFFFF) - OFFSET
        z = ((key >> 42) & 0x1FFFFF) - OFFSET
        return np.array([x, y, z], dtype=np.float32) * self.chunk_size

    def should_update_bubble(self, current_pose):
        """Gating: Only update if camera moved significantly (Fix 1)."""
        if self.last_pose is None:
            self.last_pose = current_pose.copy()
            return True
            
        dist = np.linalg.norm(current_pose[:3, 3] - self.last_pose[:3, 3])
        rot = np.arccos(np.clip((np.trace(current_pose[:3, :3].T @ self.last_pose[:3, :3]) - 1) / 2, -1, 1))
        
        # Thresholds: 1cm or 0.5 degree (Relaxed for better density)
        if dist > 0.01 or rot > np.radians(0.5):
            self.last_pose = current_pose.copy()
            return True
        return False

    def add_bubbles(self, world_pts, weights=None, colors=None, Sigmas=None):
        """Add bubbles to the local monolithic buffer first for stable fusion."""
        if isinstance(world_pts, GaussianBubbleBatch):
            batch = world_pts
        else:
            if weights is None or colors is None or Sigmas is None:
                raise TypeError("add_bubbles expects a GaussianBubbleBatch or mu, weight, color, Sigma arrays")
            batch = GaussianBubbleBatch.from_arrays(world_pts, weights, colors, Sigmas, to_numpy=to_numpy_safe)

        world_pts = batch.mu
        weights = batch.weight
        colors = batch.color
        Sigmas = batch.Sigma
        if len(world_pts) == 0:
            return
            
        with self._lock:
            xp = get_array_module(use_gpu=self.use_gpu)
            world_pts = xp.asarray(world_pts, dtype=xp.float32)
            weights = xp.asarray(weights, dtype=xp.float32)
            colors = xp.asarray(colors, dtype=xp.float32)
            Sigmas = xp.asarray(Sigmas, dtype=xp.float32)
            
            # 1. Strict Filtering
            valid_mask = xp.all(xp.isfinite(world_pts), axis=1)
            valid_mask &= xp.isfinite(weights) & (weights > 1e-4)
            if self.last_pose is not None:
                cam_pos = xp.asarray(self.last_pose[:3, 3])
                valid_mask &= (xp.sum((world_pts - cam_pos)**2, axis=1) < 2500.0)
            
            if not _truth(xp.any(valid_mask)):
                return
                
            world_pts = world_pts[valid_mask]
            weights = weights[valid_mask]
            colors = colors[valid_mask]
            Sigmas = Sigmas[valid_mask]

            # 2. Fuse into Local Monolithic Buffer before any chunk insertion.
            # Keep only a deterministic, confidence-ranked subset per frame.
            num_new = len(world_pts)
            max_new = int(self.cfg.get('max_new_bubbles_per_frame', self.max_new_bubbles))
            max_new = max(256, min(max_new, self.local_buffer_size))
            if num_new > max_new:
                stride = max(1, num_new // max_new)
                idx = xp.arange(0, num_new, stride, dtype=xp.int32)[:max_new]
                world_pts = world_pts[idx]
                weights = weights[idx]
                colors = colors[idx]
                Sigmas = Sigmas[idx]
                num_new = len(world_pts)

            n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
            
            if n_local > 0 and n_local + num_new > self.local_buffer_size * 0.95:
                # Commit local bubbles to chunked map if almost full
                self._commit_local_to_chunks()
                # Clear local hash table for reuse
                if self.use_gpu:
                    if self.local_hash_table is not None:
                        self.local_hash_table.fill(-1)
                    self.local_count.fill(0)
                else:
                    self.local_count = 0
                n_local = 0

            mu, sig, w, col = _fuse_gaussians_cpu(
                to_numpy_safe(self.local_mu[:n_local]),
                to_numpy_safe(self.local_Sigma[:n_local]),
                to_numpy_safe(self.local_weight[:n_local]),
                to_numpy_safe(self.local_color[:n_local]),
                to_numpy_safe(world_pts),
                to_numpy_safe(Sigmas),
                to_numpy_safe(weights),
                to_numpy_safe(colors),
                radius=float(self.cfg.get('local_fusion_radius', max(self.cell_size * 3.0, 0.06))),
                mahal_thresh=float(self.cfg.get('bubble_mahal_thresh', 9.0)),
                max_size=self.local_buffer_size,
                voxel_size=float(self.cfg.get('local_voxel_dedup_size', max(self.cell_size * 2.0, 0.04))),
            )

            new_len = len(mu)
            self.local_mu[:new_len] = xp.asarray(mu)
            self.local_Sigma[:new_len] = xp.asarray(sig)
            self.local_weight[:new_len] = xp.asarray(w)
            self.local_color[:new_len] = xp.asarray(col)
            if self.use_gpu:
                self.local_count.fill(new_len)
            else:
                self.local_count = new_len
            
            # V54: Invalidate cache after any insertion
            self._cache_valid = False
            self._cached_mu = None

    def _commit_local_to_chunks(self):
        """Bake the stable local bubbles into the chunked world system (V55)."""
        n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
        if n_local == 0:
            return
            
        logger.info(f" [MAP] Committing {n_local} local bubbles to chunked map")
        
        xp = cp if self.use_gpu else np
        mu_cpu = to_numpy_safe(self.local_mu[:n_local])
        sig_cpu = to_numpy_safe(self.local_Sigma[:n_local])
        w_cpu = to_numpy_safe(self.local_weight[:n_local])
        col_cpu = to_numpy_safe(self.local_color[:n_local])
        
        # Partition by chunk - use deterministic sorting for aggregation
        keys_cpu = to_numpy_safe(self._world_to_chunk_key(mu_cpu))
        unique_keys_cpu = np.unique(keys_cpu)
        unique_keys_cpu.sort()
        
        for key in unique_keys_cpu:
            key_int = int(key)
            mask_cpu = (keys_cpu == key_int)
            if not np.any(mask_cpu):
                continue
            
            if key_int not in self.chunks:
                reloaded = self._reload_chunk(key_int)
                if reloaded is not None:
                    self.chunks[key_int] = reloaded
                else:
                    origin = self._key_to_origin(key_int)
                    self.chunks[key_int] = MapChunk(
                        origin,
                        cell_size=self.cell_size,
                        use_gpu=self.use_gpu,
                        max_bubbles=int(self.cfg.get('max_bubbles_per_chunk', 10000)),
                    )
            
            chunk = self.chunks[key_int]
            chunk.last_used_frame = self.frame_counter
            
            # Subtract origin to get local coordinates for the chunk
            chunk_origin = np.asarray(chunk.origin, dtype=np.float32)
            
            # Commit with validation
            prev_len = len(chunk)
            idxs = np.flatnonzero(mask_cpu)
            batch_size = int(self.cfg.get('chunk_commit_batch_size', 1500))
            batch_size = max(128, batch_size)
            for start in range(0, len(idxs), batch_size):
                sel = idxs[start:start + batch_size]
                chunk.add_bubbles(
                    mu_cpu[sel] - chunk_origin,
                    w_cpu[sel],
                    col_cpu[sel],
                    sig_cpu[sel],
                )
            
            # Basic validation: ensure chunk size didn't shrink unexpectedly
            if len(chunk) < prev_len:
                logger.warning(f" [MAP] Chunk {key_int} size shrunk during commit: {prev_len} -> {len(chunk)}")
        
        if self.use_gpu:
            self.local_count.fill(0)
            if self.local_hash_table is not None:
                self.local_hash_table.fill(-1)
        else:
            self.local_count = 0

        self._invalidate_cache()
        self._manage_memory()

    def update_active_set(self, camera_pos):
        """Track chunks within active_radius around camera (V56 Memory-Safe)."""
        xp = cp if self.use_gpu else np
        cam_coord = np.floor(camera_pos / self.chunk_size).astype(np.int32)
        cam_coord = np.clip(cam_coord, -500000, 500000)
        
        self.active_chunks.clear()
        r = self.active_radius
        OFFSET = 1000000
        
        newly_created = 0
        MAX_NEW_CHUNKS_PER_FRAME = 20 # Limit to prevent VRAM spikes
        
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y, z = int(cam_coord[0] + dx), int(cam_coord[1] + dy), int(cam_coord[2] + dz)
                    key = int((x + OFFSET) | ((y + OFFSET) << 21) | ((z + OFFSET) << 42))
                    
                    if key not in self.chunks:
                        reloaded = self._reload_chunk(key)
                        if reloaded is not None:
                            self.chunks[key] = reloaded
                            
                    # Do not allocate empty active chunks. Chunks are created
                    # only when fused local Gaussians are committed into them.

                    if key in self.chunks:
                        self.active_chunks.add(key)
                        self.chunks[key].last_used_frame = self.frame_counter

    def prune_active(self):
        """Periodic pruning of active chunks (Fix 4)."""
        min_weight = self.cfg.get('min_weight_prune', 0.05)
        max_bubbles = self.cfg.get('max_bubbles_per_chunk', 25000)
        radius_params = {
            'min_neighbors': 3,
            'radius': 0.1
        }
        
        with self._lock:
            for key in self.active_chunks:
                self.chunks[key].prune(min_weight, max_bubbles, radius_params)
            self._invalidate_cache()

    def reintegrate_map(self, keyframes):
        """Global map correction: clears and re-integrates all keyframes (V49)."""
        keyframes = list(keyframes)
        logger.info(f" [MAP] Reintegrating {len(keyframes)} keyframes into chunked map...")
        
        with self._lock:
            # 1. Clear current state
            self.chunks.clear()
            self.active_chunks.clear()
            if self.use_gpu:
                self.local_count.fill(0)
                if self.local_hash_table is not None:
                    self.local_hash_table.fill(-1)
            else:
                self.local_count = 0
            self._invalidate_cache()
        
        added_keyframes = 0
        for kf in keyframes:
            # Re-backproject outside the map lock. add_bubbles() acquires the
            # lock internally, so holding it here would deadlock.
            pts_world, weights, colors, Sigmas = self.backproject_keyframe(kf)
            if len(pts_world) == 0:
                continue
            self.add_bubbles(pts_world, weights, colors, Sigmas)
            added_keyframes += 1
        
        self._invalidate_cache()
        self.push_to_visualizer(force=True)
        logger.info(f" [MAP] Reintegration complete ({added_keyframes}/{len(keyframes)} keyframes)")

    def backproject_keyframe(self, kf):
        """Helper to backproject a single keyframe into world coordinates."""
        # This logic is mostly a copy of the one in slam_system.py but simplified
        xp = cp if self.use_gpu else np
        
        depth = xp.asarray(kf.depth, dtype=xp.float32)
        h, w = depth.shape
        stride = self.bubble_stride
        
        # Grid of coordinates
        u = xp.arange(0, w, stride)
        v = xp.arange(0, h, stride)
        vv, uu = xp.meshgrid(v, u, indexing='ij')
        u_flat, v_flat = uu.ravel(), vv.ravel()
        
        z = depth[v_flat.astype(xp.int32), u_flat.astype(xp.int32)]
        valid = (z > 0.1) & (z < 6.0) & xp.isfinite(z)
        
        if not _truth(xp.any(valid)):
            return xp.empty((0,3)), xp.empty(0), xp.empty((0,3)), xp.empty((0,3,3))
            
        u_f, v_f, z_f = u_flat[valid], v_flat[valid], z[valid]
        
        # Cam coords
        x = (u_f - self.cx) * z_f / self.fx
        y = (v_f - self.cy) * z_f / self.fy
        pts_cam = xp.stack([x, y, z_f], axis=1)
        
        # World coords
        R_wc = xp.asarray(kf.pose[:3, :3], dtype=xp.float32)
        t_wc = xp.asarray(kf.pose[:3, 3], dtype=xp.float32)
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        
        # Weights and Colors
        weights = xp.full(len(pts_world), 2.0, dtype=xp.float32)
        
        if kf.image is not None:
            img = xp.asarray(kf.image, dtype=xp.float32)
            colors = img[v_f.astype(xp.int32), u_f.astype(xp.int32)] / 255.0
            if colors.ndim == 1:
                colors = xp.stack([colors, colors, colors], axis=1)
        else:
            colors = xp.full((len(pts_world), 3), 0.5, dtype=xp.float32)
            
        # Sigmas (simplified for reintegration)
        Sigmas = xp.tile(xp.eye(3, dtype=xp.float32) * 0.01, (len(pts_world), 1, 1))
        
        return pts_world, weights, colors, Sigmas

    def get_stabilization_cloud(self, max_pts=20000):
        """Direct extraction from local monolithic buffer for immediate visibility (V54)."""
        xp = cp if self.use_gpu else np
        with self._lock:
            n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
            if n_local == 0:
                return None, None, None
            
            # Subsample if too many
            if n_local > max_pts:
                indices = xp.random.choice(n_local, max_pts, replace=False)
            else:
                indices = xp.arange(n_local)
                
            pts = to_numpy_safe(self.local_mu[indices])
            cols = to_numpy_safe(self.local_color[indices])
            
            sig = self.local_Sigma[indices]
            trace = xp.trace(sig, axis1=1, axis2=2)
            scales = to_numpy_safe(xp.sqrt(xp.clip(trace, 1e-6, 1.0)))
            
            return pts, cols, scales

    def build_visualisation_cloud(self, max_pts=60000):
        """Decoupled visualization: builds a subsampled point cloud with adaptive scales."""
        all_mu = []
        all_colors = []
        all_scales = []
        
        threshold = self.cfg.get('bubble_visualization_threshold', 0.1)
        
        with self._lock:
            # 1. Include Local Monolithic Buffer (High priority for stability)
            n_local = int(self.local_count[0]) if self.use_gpu else self.local_count
            if n_local > 0:
                all_mu.append(to_numpy_safe(self.local_mu[:n_local]))
                all_colors.append(to_numpy_safe(self.local_color[:n_local]))
                # Scale from covariance trace
                xp = cp if self.use_gpu else np
                sig = self.local_Sigma[:n_local]
                trace = xp.trace(sig, axis1=1, axis2=2)
                all_scales.append(to_numpy_safe(xp.sqrt(xp.clip(trace, 1e-6, 1.0))))

            # 2. Include resident chunks with LOD (Level of Detail)
            for key, chunk in self.chunks.items():
                n = len(chunk)
                if n == 0: continue
                    
                xp = cp if self.use_gpu else np
                
                # Active chunks get high detail, inactive get LOD
                is_active = key in self.active_chunks
                lod_factor = 1.0 if is_active else 0.3
                
                if lod_factor < 1.0:
                    mu_world, colors, weights_chunk = chunk.get_lod_cloud(lod_factor)
                    if mu_world is None: continue
                    mask = weights_chunk > threshold
                    if not _truth(xp.any(mask)): continue
                    mu_world = mu_world[mask]
                    colors = colors[mask]
                    scale_count = len(mu_world)
                else:
                    weights_chunk = chunk.weight[:n]
                    mask = weights_chunk > threshold
                    if not _truth(xp.any(mask)): continue
                    mu_world = chunk.mu_local[:n][mask] + xp.asarray(chunk.origin)
                    colors = chunk.color[:n][mask]
                    scale_count = len(mu_world)
                
                all_mu.append(to_numpy_safe(mu_world))
                all_colors.append(to_numpy_safe(colors))
                
                # Re-calculate trace for adaptive scaling on filtered set
                sig = chunk.Sigma[:n] # Simplified: use full chunk sig for trace if available
                # In LOD mode, we would need to map indices. For now, use a constant trace 
                # for distant chunks to save GPU time.
                if is_active:
                    trace = xp.trace(chunk.Sigma[:n][mask], axis1=1, axis2=2)
                    all_scales.append(to_numpy_safe(xp.sqrt(xp.clip(trace, 1e-6, 1.0))))
                else:
                    all_scales.append(np.full(scale_count, 0.1, dtype=np.float32))
        
        if not all_mu:
            return None, None, None
            
        pts = np.concatenate(all_mu, axis=0)
        cols = np.concatenate(all_colors, axis=0)
        scales = np.concatenate(all_scales, axis=0)
        
        # Filter non-finite and insanely large values
        valid = np.all(np.isfinite(pts), axis=1) & np.isfinite(scales)
        if np.any(valid):
            valid &= np.all(np.abs(pts) < 5000.0, axis=1) # 5km limit
            
        if not np.all(valid):
            pts = pts[valid]
            cols = cols[valid]
            scales = scales[valid]
            
        if len(pts) == 0:
            return None, None, None

        # Random LOD subsampling if too many points
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
            cols = cols[idx]
            scales = scales[idx]
            
        return pts, cols, scales

    def push_to_visualizer(self, force=False):
        """Trigger the background viz worker."""
        self._viz_update_needed.set()

    # REMOVED OLD push_to_visualizer here - it was below build_visualisation_cloud in previous searchreplace

    def shutdown(self):
        """Cleanup resources."""
        self.threaded_manager.stop()
        self._viz_running = False
        self._viz_update_needed.set()
        if self._viz_thread:
            self._viz_thread.join(timeout=1.0)
        # Optional: clear memory
        self.chunks.clear()
        self.active_chunks.clear()
        if self.use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def backproject_frame(self, depth, pose, image, stride, motion_scale=1.0):
        """Adapted backprojection that returns world-space bubbles with edge filtering."""
        xp = cp if self.use_gpu else np
        
        d_gpu = xp.asarray(depth, dtype=xp.float32)
        pose_gpu = xp.asarray(pose, dtype=xp.float32)
        
        h, w = d_gpu.shape
        
        # ── V53: REDUCED FILTERING IN STABILIZATION MODE ──
        # Relax edge and weight filtering to ensure visibility during bootstrapping
        edge_thresh = self.cfg.get("bubble_depth_edge_thresh", 0.1)
        if self.stabilization_mode:
            edge_thresh = 0.25 # More relaxed
            
        if edge_thresh > 0:
            # Simple Sobel-like edge detection on depth
            d_padded = xp.pad(d_gpu, 1, mode='edge')
            dx = xp.abs(d_padded[1:-1, 2:] - d_padded[1:-1, :-2])
            dy = xp.abs(d_padded[2:, 1:-1] - d_padded[:-2, 1:-1])
            edges = (dx > edge_thresh * d_gpu) | (dy > edge_thresh * d_gpu)
        else:
            edges = xp.zeros_like(d_gpu, dtype=bool)

        u_gpu = xp.arange(0, w, stride, dtype=xp.float32)
        v_gpu = xp.arange(0, h, stride, dtype=xp.float32)
        v_grid, u_grid = xp.meshgrid(v_gpu, u_gpu, indexing='ij')
        u_flat, v_flat = u_grid.ravel(), v_grid.ravel()
        
        v_idx = v_flat.astype(xp.int32)
        u_idx = u_flat.astype(xp.int32)
        
        z_gpu = d_gpu[v_idx, u_idx]
        is_edge = edges[v_idx, u_idx]
        
        max_d = float(self.cfg.get("bubble_max_depth", 8.0))
        min_d = float(self.cfg.get("bubble_min_depth", 0.1))
        
        # NaN rejection and range gating
        valid = (z_gpu > min_d) & (z_gpu < max_d) & xp.isfinite(z_gpu)
        
        # Only apply edge filtering if not in aggressive stabilization mode
        if not self.stabilization_mode:
            valid &= (~is_edge)
        
        if not _truth(xp.any(valid)):
            return xp.empty((0,3)), xp.empty(0), xp.empty((0,3)), xp.empty((0,3,3))
            
        u_f, v_f, d_f = u_flat[valid], v_flat[valid], z_gpu[valid]
        
        x = (u_f - self.cx) * d_f / self.fx
        y = (v_f - self.cy) * d_f / self.fy
        pts_cam = xp.stack([x, y, d_f], axis=1)
        
        # Sigmas (scaled by depth)
        sigma_par = (d_f**2 / (self.fx * self.baseline)) * self.cfg.get("bubble_sigma_disp", 0.4) * motion_scale
        sigma_per = (d_f / self.fx) * self.cfg.get("bubble_sigma_pix", 0.2) * motion_scale
        sigma_par = xp.minimum(sigma_par, 0.15)
        
        zeros = xp.zeros_like(sigma_per)
        diag_ray_3x3 = xp.stack([
            xp.stack([sigma_per**2, zeros, zeros], axis=1),
            xp.stack([zeros, sigma_per**2, zeros], axis=1), 
            xp.stack([zeros, zeros, sigma_par**2], axis=1)
        ], axis=1)
        
        # Transform to world
        # V59: Ensure pose is correctly applied for T_wc
        R_wc = pose_gpu[:3,:3]
        t_wc = pose_gpu[:3,3]
        
        # pts_cam is [N, 3] in OpenCV convention (X-right, Y-down, Z-forward)
        # pts_world = R_wc * pts_cam + t_wc
        pts_world = xp.matmul(R_wc, pts_cam.T).T + t_wc
        
        # Rotate covariance: Sig_world = R_wc @ Sig_cam @ R_wc.T
        Sig_world = rotate_covariance_batch(R_wc, diag_ray_3x3)
        
        # Colors
        if image is not None:
            img_gpu = xp.asarray(image, dtype=xp.float32)
            u_i = xp.clip(u_f.astype(xp.int32), 0, w - 1)
            v_i = xp.clip(v_f.astype(xp.int32), 0, h - 1)
            colors = img_gpu[v_i, u_i] / 255.0
            if colors.ndim == 1:
                colors = xp.stack([colors, colors, colors], axis=1)
            # Add subtle vibrance boost
            colors = xp.clip(colors * 1.1, 0.0, 1.0)
        else:
            # Color by depth if no image
            norm_d = xp.clip((d_f - min_d) / (max_d - min_d), 0, 1)
            colors = xp.stack([norm_d, 1.0 - norm_d, xp.full_like(norm_d, 0.5)], axis=1)
            
        # Higher initial weight to help survive pruning and look "solid"
        initial_weight = 5.0 if self.stabilization_mode else 2.0
        weights = xp.full(len(pts_world), initial_weight, dtype=xp.float32)
        
        return pts_world, weights, colors, Sig_world
