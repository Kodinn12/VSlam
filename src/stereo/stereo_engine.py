"""Unified stereo engine interface with CPU/GPU auto-selection."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False

from .stereo_pipeline_cpu import StereoPipelineCPU
from .stereo_pipeline_gpu import StereoPipelineGPU


class StereoEngine:
    """
    Unified stereo engine that auto-selects CPU or GPU SGM pipeline.
    
    This provides a single interface for depth computation regardless of
    whether the system is running in CPU-only or GPU-accelerated mode.
    """
    
    def __init__(self, config: dict, acceleration_mode: str = 'auto'):
        """
        Initialize stereo engine.
        
        Parameters
        ----------
        config : dict
            Configuration parameters
        acceleration_mode : str
            'auto' (detect), 'cpu_only', 'full_gpu'
        """
        self.config = config
        self.acceleration_mode = acceleration_mode
        self.pipeline = None
        
        # Auto-detect acceleration mode if requested
        if acceleration_mode == 'auto':
            if USE_CUPY:
                self.acceleration_mode = 'full_gpu'
                logger.info("Auto-detected GPU mode for stereo engine")
            else:
                self.acceleration_mode = 'cpu_only'
                logger.info("Auto-detected CPU mode for stereo engine")
        
        # Initialize appropriate pipeline
        if self.acceleration_mode == 'full_gpu':
            if USE_CUPY:
                self.pipeline = StereoPipelineGPU(config)
                logger.info("Stereo engine: GPU SGM pipeline")
            else:
                logger.warning("GPU mode requested but CuPy not available, falling back to CPU")
                self.pipeline = StereoPipelineCPU(config)
        else:
            self.pipeline = StereoPipelineCPU(config)
            logger.info("Stereo engine: CPU SGM pipeline")
    
    def compute_depth(self, left: np.ndarray, right: np.ndarray,
                     baseline: float = None, focal_length: float = None,
                     motion_score: float = 0.0, **kwargs) -> np.ndarray:
        """
        Compute depth map from left/right stereo pair.
        
        Parameters
        ----------
        left : np.ndarray
            Left grayscale image (H, W), uint8
        right : np.ndarray
            Right grayscale image (H, W), uint8
        baseline : float
            Stereo baseline in meters
        focal_length : float
            Focal length in pixels
        motion_score : float
            Motion score for temporal smoothing (0-1)
        
        Returns
        -------
        np.ndarray
            Depth map (H, W), float32 (meters)
        """
        if self.pipeline is None:
            raise RuntimeError("Stereo pipeline not initialized")

        if baseline is None:
            baseline = kwargs.pop("baseline_m", None)
        if focal_length is None:
            focal_length = kwargs.pop("fx", None)
        if baseline is None or focal_length is None:
            raise TypeError("compute_depth requires baseline/focal_length or baseline_m/fx")
        
        return self.pipeline.compute_depth(left, right, baseline, focal_length, motion_score)
    
    def compute_disparity(self, left: np.ndarray, right: np.ndarray,
                         motion_score: float = 0.0) -> np.ndarray:
        """
        Compute disparity map from left/right stereo pair.
        
        Parameters
        ----------
        left : np.ndarray
            Left grayscale image (H, W), uint8
        right : np.ndarray
            Right grayscale image (H, W), uint8
        motion_score : float
            Motion score for temporal smoothing (0-1)
        
        Returns
        -------
        np.ndarray
            Disparity map (H, W), float32 (pixels)
        """
        if self.pipeline is None:
            raise RuntimeError("Stereo pipeline not initialized")
        
        return self.pipeline.compute_disparity(left, right, motion_score)
