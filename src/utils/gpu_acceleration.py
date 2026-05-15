"""GPU acceleration manager for SLAM system components."""

import numpy as np
import logging
from typing import Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

USE_KORNIA = False

try:
    from .cupy_utils import cp, USE_CUPY, USE_TORCH
    if USE_CUPY:
        logger.info("CuPy GPU acceleration available")
    else:
        logger.info("CuPy disabled or unavailable, using NumPy/Torch fallback")
except Exception:
    USE_CUPY = False
    USE_TORCH = False
    cp = np
    logger.info("CuPy not available, using NumPy fallback")

try:
    import torch
    if USE_TORCH and torch.cuda.is_available():
        logger.info("PyTorch GPU acceleration available")
        torch.cuda.empty_cache()  # Clear cache
    else:
        logger.info("PyTorch GPU not available, using CPU")
except Exception:
    logger.info("PyTorch not available")

try:
    import kornia.geometry
    USE_KORNIA = True
    logger.info("Kornia GPU geometry operations available")
except Exception:
    logger.info("Kornia not available")

class GPUAccelerationManager:
    """Manages GPU acceleration across SLAM components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.acceleration_mode = config.get('acceleration_mode', 'full_gpu')
        self.gpu_memory_pool = None
        self._setup_gpu_memory()
        
    def _setup_gpu_memory(self):
        """Setup GPU memory management with aggressive optimization settings."""
        if USE_CUPY and self.acceleration_mode != 'cpu_only':
            try:
                # Get optimization settings from config
                # V59: Conservative memory limit (2.5GB) for 6GB cards
                memory_limit = self.config.get('gpu_memory_pool_limit', 2500 * 1024 * 1024)
                
                # V59: Cap memory limit at 2500MB for 6GB cards to allow Torch/Display room
                if memory_limit > 2500 * 1024 * 1024:
                    memory_limit = 2500 * 1024 * 1024
                    logger.info(f" [VRAM] Capping CuPy memory pool at 2500MB for stability")
                
                pinned_limit = self.config.get('gpu_pinned_memory_limit', 536870912)  # 512MB
                
                # Setup optimized memory pools
                self.gpu_memory_pool = cp.get_default_memory_pool()
                self.gpu_pinned_pool = cp.get_default_pinned_memory_pool()
                
                # Set memory limits for better performance (only GPU pool supports set_limit)
                self.gpu_memory_pool.set_limit(size=memory_limit)
                # Note: PinnedMemoryPool doesn't support set_limit() in current CuPy version
                
                # Aggressive GPU memory pre-allocation (adjusted for available memory)
                if self.config.get('gpu_force_memory_usage', False): # V57: Default to False
                    # Pre-allocate moderate GPU arrays to force memory usage without OOM
                    large_arrays = []
                    try:
                        # V56: More conservative pre-allocation for 6GB GPUs
                        # Pre-allocate 2 arrays of ~50MB each
                        for i in range(2):
                            arr = cp.zeros((2000, 3000), dtype=cp.float32)  # 2K x 3K x 4 bytes = ~24MB each
                            large_arrays.append(arr)
                        
                        logger.info(f"GPU memory pre-allocated: {len(large_arrays)} arrays (~24MB each)")
                    except Exception as mem_e:
                        logger.warning(f"GPU memory pre-allocation failed: {mem_e}")
                        # Fallback to smaller allocation
                        try:
                            fallback_arr = cp.zeros((10000, 3), dtype=cp.float32)  # ~120MB
                            temp = cp.sum(fallback_arr * 2.0)
                            logger.info("GPU memory fallback allocation successful")
                        except Exception as fallback_e:
                            logger.warning(f"GPU memory fallback also failed: {fallback_e}")
                
                # Pre-allocate some memory to reduce initialization overhead
                if self.config.get('gpu_prefer_large_kernels', True):
                    # Warm up GPU with realistic workload
                    warmup_array = cp.zeros((10000, 3), dtype=cp.float32)
                    del warmup_array
                
                logger.info(f"GPU memory pools initialized (limit: {memory_limit//1024**2}MB)")
                
            except Exception as e:
                logger.warning(f"GPU memory pool setup failed: {e}")
                
    def get_acceleration_config(self) -> Dict[str, bool]:
        """Get acceleration configuration for components."""
        mode = self.acceleration_mode
        
        if mode == 'full_gpu':
            return {
                'use_cupy': USE_CUPY,
                'use_torch': USE_TORCH,
                'use_kornia': USE_KORNIA,
                'gpu_bubbles': USE_CUPY,
                'gpu_tsdf': USE_CUPY,
                'gpu_ba': USE_TORCH,
                'gpu_pgo': USE_CUPY,
                'gpu_pnp': USE_TORCH and USE_KORNIA,
                'gpu_tracking': USE_TORCH,
                'gpu_visualization': USE_CUPY
            }
        elif mode == 'mixed_gpu_cpu':
            return {
                'use_cupy': USE_CUPY,
                'use_torch': USE_TORCH,
                'use_kornia': USE_KORNIA,
                'gpu_bubbles': USE_CUPY,  # GPU for bubbles
                'gpu_tsdf': False,        # CPU for TSDF
                'gpu_ba': USE_TORCH,      # GPU for BA
                'gpu_pgo': False,         # CPU for PGO
                'gpu_pnp': USE_TORCH and USE_KORNIA,
                'gpu_tracking': False,    # CPU for tracking
                'gpu_visualization': USE_CUPY
            }
        else:  # cpu_only
            return {
                'use_cupy': False,
                'use_torch': False,
                'use_kornia': False,
                'gpu_bubbles': False,
                'gpu_tsdf': False,
                'gpu_ba': False,
                'gpu_pgo': False,
                'gpu_pnp': False,
                'gpu_tracking': False,
                'gpu_visualization': False
            }
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        if USE_CUPY and self.gpu_memory_pool:
            try:
                # Free unused memory
                self.gpu_memory_pool.free_all_blocks()
                if self.gpu_pinned_pool:
                    self.gpu_pinned_pool.free_all_blocks()
                logger.debug("GPU memory optimized")
            except Exception as e:
                logger.warning(f"GPU memory optimization failed: {e}")
                
        if USE_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("PyTorch GPU cache cleared")
            except Exception as e:
                logger.warning(f"PyTorch cache clear failed: {e}")
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        info = {}
        
        if USE_CUPY and self.gpu_memory_pool:
            try:
                info['cupy_used'] = self.gpu_memory_pool.used_bytes() / (1024**2)  # MB
                info['cupy_total'] = self.gpu_memory_pool.total_bytes() / (1024**2)  # MB
            except Exception:
                pass
                
        if USE_TORCH and torch.cuda.is_available():
            try:
                info['torch_allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
                info['torch_reserved'] = torch.cuda.memory_reserved() / (1024**2)  # MB
            except Exception:
                pass
                
        return info
    
    def benchmark_acceleration(self) -> Dict[str, float]:
        """Benchmark GPU vs CPU performance."""
        results = {}
        
        # Matrix multiplication benchmark
        size = 1000
        a_cpu = np.random.random((size, size)).astype(np.float32)
        b_cpu = np.random.random((size, size)).astype(np.float32)
        
        # CPU benchmark
        import time
        start = time.time()
        c_cpu = np.dot(a_cpu, b_cpu)
        cpu_time = time.time() - start
        results['cpu_matmul'] = cpu_time
        
        # GPU benchmark
        if USE_CUPY:
            a_gpu = cp.array(a_cpu)
            b_gpu = cp.array(b_cpu)
            start = time.time()
            c_gpu = cp.dot(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start
            results['gpu_matmul'] = gpu_time
            results['speedup'] = cpu_time / gpu_time
        else:
            results['gpu_matmul'] = None
            results['speedup'] = None
            
        return results
    
    def shutdown(self):
        """Cleanup GPU resources."""
        self.optimize_gpu_memory()
        logger.info("GPU acceleration manager shutdown complete")
