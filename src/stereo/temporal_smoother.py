"""Temporal smoothing for depth/disparity maps."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


class TemporalSmootherCPU:
    """IIR filter for temporal depth smoothing on CPU."""
    
    def __init__(self, alpha: float = 0.3, motion_weight: float = 0.5):
        """
        Initialize temporal smoother.
        
        Parameters
        ----------
        alpha : float
            IIR filter coefficient (0-1), higher = more smoothing
        motion_weight : float
            Weight for motion-based alpha adaptation
        """
        self.alpha = alpha
        self.motion_weight = motion_weight
        self.prev_disp = None
        self.prev_image = None
    
    def smooth(self, disparity: np.ndarray, image: np.ndarray = None,
               motion_score: float = 0.0) -> np.ndarray:
        """
        Apply temporal smoothing to disparity map.
        
        Parameters
        ----------
        disparity : np.ndarray
            Current disparity map (H, W), float32
        image : np.ndarray
            Current grayscale image (H, W), uint8, optional
        motion_score : float
            Motion score (0-1), higher = faster motion
        
        Returns
        -------
        np.ndarray
            Smoothed disparity map (H, W), float32
        """
        if self.prev_disp is None:
            self.prev_disp = disparity.copy()
            if image is not None:
                self.prev_image = image.copy()
            return disparity
        
        # Motion-adaptive alpha: higher motion = lower alpha (less smoothing)
        adaptive_alpha = self.alpha * (1.0 - self.motion_weight * motion_score)
        adaptive_alpha = np.clip(adaptive_alpha, 0.0, 1.0)
        
        # IIR filter: smoothed = alpha * current + (1-alpha) * previous
        smoothed = adaptive_alpha * disparity + (1.0 - adaptive_alpha) * self.prev_disp
        
        # Update previous
        self.prev_disp = smoothed.copy()
        if image is not None:
            self.prev_image = image.copy()
        
        return smoothed


class TemporalSmootherGPU:
    """IIR filter for temporal depth smoothing on GPU."""
    
    def __init__(self, alpha: float = 0.3, motion_weight: float = 0.5):
        """
        Initialize temporal smoother.
        
        Parameters
        ----------
        alpha : float
            IIR filter coefficient (0-1), higher = more smoothing
        motion_weight : float
            Weight for motion-based alpha adaptation
        """
        self.alpha = alpha
        self.motion_weight = motion_weight
        self.prev_disp = None
        self.prev_image = None
        
        if not USE_CUPY:
            logger.warning("CuPy not available, will use CPU fallback")
            self.cpu_smoother = TemporalSmootherCPU(alpha, motion_weight)
    
    def smooth(self, disparity: np.ndarray, image: np.ndarray = None,
               motion_score: float = 0.0) -> np.ndarray:
        """
        Apply temporal smoothing to disparity map.
        
        Parameters
        ----------
        disparity : np.ndarray
            Current disparity map on GPU (H, W), float32
        image : np.ndarray
            Current grayscale image on GPU (H, W), uint8, optional
        motion_score : float
            Motion score (0-1), higher = faster motion
        
        Returns
        -------
        np.ndarray
            Smoothed disparity map on GPU (H, W), float32
        """
        if not USE_CUPY:
            logger.warning("CuPy not available, falling back to CPU")
            return self.cpu_smoother.smooth(disparity, image, motion_score)
        
        disparity_gpu = cp.asarray(disparity, dtype=cp.float32)
        
        if self.prev_disp is None:
            self.prev_disp = disparity_gpu.copy()
            if image is not None:
                self.prev_image = cp.asarray(image, dtype=cp.uint8)
            return disparity_gpu
        
        # Motion-adaptive alpha: higher motion = lower alpha (less smoothing)
        adaptive_alpha = self.alpha * (1.0 - self.motion_weight * motion_score)
        adaptive_alpha = max(0.0, min(1.0, adaptive_alpha))
        
        # IIR filter: smoothed = alpha * current + (1-alpha) * previous
        smoothed = adaptive_alpha * disparity_gpu + (1.0 - adaptive_alpha) * self.prev_disp
        
        # Update previous
        self.prev_disp = smoothed.copy()
        if image is not None:
            self.prev_image = cp.asarray(image, dtype=cp.uint8)
        
        return smoothed
