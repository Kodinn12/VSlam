"""Zero-copy memory transfer utilities for GPU pipelines."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


class ZeroCopyTransfer:
    """Zero-copy memory transfer for efficient CPU-GPU data movement."""
    
    def __init__(self, use_pinned_memory: bool = True):
        """
        Initialize zero-copy transfer manager.
        
        Parameters
        ----------
        use_pinned_memory : bool
            Whether to use pinned memory for faster transfers
        """
        self.use_pinned_memory = use_pinned_memory and USE_CUPY
        if self.use_pinned_memory:
            logger.info("Zero-copy transfers enabled with pinned memory")
        else:
            logger.info("Zero-copy transfers disabled (using standard transfers)")
    
    def async_transfer_to_gpu(self, array: np.ndarray, stream=None) -> np.ndarray:
        """
        Async transfer array to GPU with zero-copy if possible.
        
        Parameters
        ----------
        array : np.ndarray
            Input array on CPU
        stream : cp.cuda.Stream, optional
            CUDA stream for async transfer
        
        Returns
        -------
        np.ndarray
            Array on GPU
        """
        if not USE_CUPY:
            return array
        
        if stream is not None:
            # Async transfer with stream
            return cp.asarray(array, stream=stream)
        else:
            # Standard transfer
            return cp.asarray(array)
    
    def async_transfer_to_cpu(self, array: np.ndarray, stream=None) -> np.ndarray:
        """
        Async transfer array from GPU with zero-copy if possible.
        
        Parameters
        ----------
        array : np.ndarray
            Input array on GPU
        stream : cp.cuda.Stream, optional
            CUDA stream for async transfer
        
        Returns
        -------
        np.ndarray
            Array on CPU
        """
        if not USE_CUPY:
            return np.asarray(array)
        
        if stream is not None:
            # Async transfer with stream
            return cp.asnumpy(array, stream=stream)
        else:
            # Standard transfer
            return cp.asnumpy(array)
    
    def allocate_pinned_buffer(self, shape: tuple, dtype=np.float32) -> np.ndarray:
        """
        Allocate pinned memory buffer for faster transfers.
        
        Parameters
        ----------
        shape : tuple
            Buffer shape
        dtype : type
            Data type
        
        Returns
        -------
        np.ndarray
            Pinned memory buffer on CPU
        """
        if not self.use_pinned_memory:
            return np.empty(shape, dtype=dtype)
        
        try:
            # CuPy pinned memory allocation
            mempool = cp.get_default_pinned_memory_pool()
            return cp.ndarray(shape, dtype=dtype, mempool=mempool)
        except Exception as e:
            logger.warning(f"Failed to allocate pinned memory: {e}, falling back to standard allocation")
            return np.empty(shape, dtype=dtype)
