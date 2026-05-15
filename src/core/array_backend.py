"""Unified array backend abstraction for CPU/GPU compatibility."""

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

import numpy as np


def get_array_module(use_gpu: bool = False, arr=None):
    """Return the array module to use for a requested mode or existing array."""
    if arr is not None and CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def is_gpu_array(arr) -> bool:
    """Return True when arr is a CuPy array."""
    return CUPY_AVAILABLE and isinstance(arr, cp.ndarray)


def to_numpy(arr):
    """Convert NumPy/CuPy-like input to a NumPy array."""
    if is_gpu_array(arr):
        return cp.asnumpy(arr)
    return np.asarray(arr)


class ArrayBackend:
    """
    Unified array backend that provides CPU/GPU array operations.
    
    This abstracts away the differences between NumPy (CPU) and CuPy (GPU)
    operations, allowing the same code to run in both modes.
    """
    
    def __init__(self, mode: str = 'cpu_only'):
        """
        Initialize array backend.
        
        Parameters
        ----------
        mode : str
            Acceleration mode ('cpu_only', 'full_gpu', 'gpu_light')
        """
        self.mode = mode
        self.xp = self._get_backend(mode)
        logger.info(f"ArrayBackend initialized: mode={mode}, backend={'CuPy' if self.xp is cp else 'NumPy'}")
    
    def _get_backend(self, mode: str):
        """
        Get array backend module based on mode.
        
        Parameters
        ----------
        mode : str
            Acceleration mode
        
        Returns
        -------
        module
            NumPy or CuPy module
        """
        return get_array_module(use_gpu=(mode == 'full_gpu'))
    
    def to_numpy(self, arr):
        """
        Convert array to NumPy (CPU) if needed.
        
        Parameters
        ----------
        arr : array-like
            Input array (NumPy or CuPy)
        
        Returns
        -------
        np.ndarray
            NumPy array on CPU
        """
        return to_numpy(arr)
    
    def to_gpu(self, arr):
        """
        Convert array to GPU if GPU mode is active.
        
        Parameters
        ----------
        arr : array-like
            Input array (NumPy or CuPy)
        
        Returns
        -------
        array
            Array on GPU if CuPy available and mode is GPU, else NumPy
        """
        if self.mode == 'full_gpu' and CUPY_AVAILABLE:
            return cp.asarray(arr)
        return np.asarray(arr)
    
    def is_gpu_array(self, arr) -> bool:
        """
        Check if array is on GPU.
        
        Parameters
        ----------
        arr : array-like
            Input array
        
        Returns
        -------
        bool
            True if array is CuPy array on GPU
        """
        return is_gpu_array(arr)
    
    def zeros(self, shape, dtype=np.float32):
        """
        Create zeros array.
        
        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : type
            Data type
        
        Returns
        -------
        array
            Zeros array (NumPy or CuPy based on mode)
        """
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float32):
        """
        Create ones array.
        
        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : type
            Data type
        
        Returns
        -------
        array
            Ones array (NumPy or CuPy based on mode)
        """
        return self.xp.ones(shape, dtype=dtype)
    
    def empty(self, shape, dtype=np.float32):
        """
        Create uninitialized array.
        
        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : type
            Data type
        
        Returns
        -------
        array
            Empty array (NumPy or CuPy based on mode)
        """
        return self.xp.empty(shape, dtype=dtype)
