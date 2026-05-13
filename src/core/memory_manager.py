"""Memory manager for VRAM/RAM budget tracking and eviction policies."""

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_CUDA_AVAILABLE = False

import numpy as np


class MemoryManager:
    """Manages VRAM/RAM budget tracking and eviction policies with CUDA stream support."""
    
    def __init__(self, config: dict, mode: str = 'cpu_only'):
        """
        Initialize memory manager.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        mode : str
            Acceleration mode ('cpu_only', 'full_gpu', 'gpu_light')
        """
        self.config = config
        self.mode = mode
        
        # Memory limits
        if mode == 'full_gpu' and CUPY_AVAILABLE:
            self.vram_limit = config.get('gpu_memory_pool_limit', 2147483648)  # 2GB default
            self.pinned_limit = config.get('gpu_pinned_memory_limit', 536870912)  # 512MB default
            self._setup_gpu_memory_pool()
            
            # CUDA stream for async operations
            self.stream = cp.cuda.Stream()
            logger.info("CUDA stream created for async operations")
        else:
            self.vram_limit = None
            self.pinned_limit = None
            self.stream = None
        
        # RAM limit (for CPU mode)
        self.ram_limit = config.get('ram_limit', 8 * 1024**3)  # 8GB default
        
        logger.info(f"MemoryManager initialized: mode={mode}, vram_limit={self.vram_limit}, ram_limit={self.ram_limit}")
    
    def _setup_gpu_memory_pool(self):
        """Setup CuPy memory pool for GPU mode."""
        if not CUPY_AVAILABLE:
            return
        
        try:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(self.vram_limit)
            
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.set_limit(self.pinned_limit)
            
            logger.info(f"GPU memory pool configured: limit={self.vram_limit / (1024**3):.2f} GB")
        except Exception as e:
            logger.warning(f"Failed to configure GPU memory pool: {e}")
    
    def get_stream(self):
        """
        Get CUDA stream for async operations.
        
        Returns
        -------
        cp.cuda.Stream or None
            CUDA stream if available
        """
        return self.stream
    
    def synchronize_stream(self):
        """Synchronize CUDA stream if available."""
        if self.stream is not None:
            self.stream.synchronize()
    
    def get_vram_usage(self) -> float:
        """
        Get current VRAM usage in bytes.
        
        Returns
        -------
        float
            VRAM usage in bytes, 0 if unavailable
        """
        if TORCH_CUDA_AVAILABLE:
            try:
                return torch.cuda.memory_allocated()
            except Exception:
                pass
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                return mempool.used_bytes()
            except Exception:
                pass
        return 0.0
    
    def get_vram_free(self) -> float:
        """
        Get free VRAM in bytes.
        
        Returns
        -------
        float
            Free VRAM in bytes, 0 if unavailable
        """
        if TORCH_CUDA_AVAILABLE:
            try:
                return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            except Exception:
                pass
        return 0.0
    
    def check_vram_pressure(self, threshold: float = 0.9) -> bool:
        """
        Check if VRAM pressure is high.
        
        Parameters
        ----------
        threshold : float
            Pressure threshold (0-1)
        
        Returns
        -------
        bool
            True if VRAM pressure exceeds threshold
        """
        if self.vram_limit is None:
            return False
        
        usage = self.get_vram_usage()
        pressure = usage / self.vram_limit if self.vram_limit > 0 else 0.0
        return pressure > threshold
    
    def should_evict(self, current_usage: float, new_allocation: float) -> bool:
        """
        Determine if eviction is needed for new allocation.
        
        Parameters
        ----------
        current_usage : float
            Current memory usage in bytes
        new_allocation : float
            New allocation size in bytes
        
        Returns
        -------
        bool
            True if eviction should occur
        """
        if self.vram_limit is None:
            return False
        
        projected = current_usage + new_allocation
        return projected > self.vram_limit
    
    def free_all_gpu_memory(self):
        """Free all unused GPU memory."""
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                logger.info("Freed all unused GPU memory blocks")
            except Exception as e:
                logger.warning(f"Failed to free GPU memory: {e}")
        
        if TORCH_CUDA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch CUDA cache")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")
