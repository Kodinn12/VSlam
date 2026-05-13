"""Mode manager for CPU/GPU acceleration mode detection and switching."""

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
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False


class ModeManager:
    """Manages CPU/GPU acceleration mode detection and switching."""
    
    VALID_MODES = ['cpu_only', 'gpu_light', 'full_gpu', 'auto']
    
    def __init__(self, config: dict):
        """
        Initialize mode manager.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.mode = self._detect_mode(config.get('acceleration_mode', 'auto'))
        self.xp = self._get_array_backend()
        logger.info(f"ModeManager initialized: mode={self.mode}, cupy={CUPY_AVAILABLE}, torch_cuda={TORCH_CUDA_AVAILABLE}")
    
    def _detect_mode(self, requested_mode: str) -> str:
        """
        Detect and validate acceleration mode.
        
        Parameters
        ----------
        requested_mode : str
            Requested acceleration mode
        
        Returns
        -------
        str
            Validated acceleration mode
        """
        if requested_mode == 'auto':
            # Auto-detect based on available hardware
            if CUPY_AVAILABLE and self._get_vram_gb() >= 6:
                return 'full_gpu'
            elif TORCH_CUDA_AVAILABLE:
                return 'gpu_light'
            else:
                return 'cpu_only'
        
        if requested_mode not in self.VALID_MODES:
            logger.warning(f"Invalid mode '{requested_mode}', defaulting to 'cpu_only'")
            return 'cpu_only'
        
        # Validate if requested mode is actually available
        if requested_mode == 'full_gpu' and not CUPY_AVAILABLE:
            logger.warning("CuPy not available, falling back to 'cpu_only'")
            return 'cpu_only'
        
        if requested_mode == 'gpu_light' and not TORCH_CUDA_AVAILABLE:
            logger.warning("PyTorch CUDA not available, falling back to 'cpu_only'")
            return 'cpu_only'
        
        return requested_mode
    
    def _get_vram_gb(self) -> float:
        """
        Get available VRAM in GB.
        
        Returns
        -------
        float
        VRAM in GB, 0 if unavailable
        """
        if TORCH_CUDA_AVAILABLE:
            try:
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                pass
        return 0.0
    
    def _get_array_backend(self):
        """
        Get array backend (NumPy or CuPy) based on mode.
        
        Returns
        -------
        module
            NumPy or CuPy module
        """
        if self.mode == 'full_gpu' and CUPY_AVAILABLE:
            return cp
        else:
            import numpy as np
            return np
    
    def get_mode(self) -> str:
        """
        Get current acceleration mode.
        
        Returns
        -------
        str
            Current acceleration mode
        """
        return self.mode
    
    def is_gpu_mode(self) -> bool:
        """
        Check if current mode uses GPU acceleration.
        
        Returns
        -------
        bool
            True if using GPU
        """
        return self.mode in ['full_gpu', 'gpu_light']
    
    def is_cupy_mode(self) -> bool:
        """
        Check if current mode uses CuPy.
        
        Returns
        -------
        bool
            True if using CuPy
        """
        return self.mode == 'full_gpu' and CUPY_AVAILABLE
    
    def switch_mode(self, new_mode: str) -> bool:
        """
        Switch to a different acceleration mode.
        
        Parameters
        ----------
        new_mode : str
            New acceleration mode
        
        Returns
        -------
        bool
            True if switch was successful
        """
        if new_mode not in self.VALID_MODES:
            logger.error(f"Invalid mode '{new_mode}'")
            return False
        
        if new_mode == self.mode:
            logger.info(f"Already using mode '{new_mode}'")
            return True
        
        old_mode = self.mode
        self.mode = self._detect_mode(new_mode)
        self.xp = self._get_array_backend()
        
        logger.info(f"Switched mode from '{old_mode}' to '{self.mode}'")
        return True
