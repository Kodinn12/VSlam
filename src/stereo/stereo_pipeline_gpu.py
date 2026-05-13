"""Full GPU SGM stereo pipeline using CuPy RawKernels."""

import numpy as np
from ..utils.logger import get_logger

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False

from .census_transform import census_transform_gpu
from .matching_cost import hamming_cost_gpu
from .cost_volume import build_cost_volume_gpu, refine_cost_volume_gpu
from .path_aggregation import aggregate_paths_gpu
from .disparity_refinement import refine_disparity_cpu  # Temporarily use CPU for refinement
from .bilateral_filter import bilateral_filter_gpu
from .temporal_smoother import TemporalSmootherGPU

logger = get_logger(__name__)


class StereoPipelineGPU:
    """Full 8-path SGM stereo pipeline on GPU."""
    
    def __init__(self, config: dict):
        """
        Initialize GPU SGM pipeline.
        
        Parameters
        ----------
        config : dict
            Configuration parameters
        """
        if not USE_CUPY:
            raise RuntimeError("CuPy not available for GPU SGM pipeline")
        
        self.config = config
        
        # SGM parameters
        self.census_window = config.get('sgm_census_window', 5)
        self.max_disparity = config.get('sgm_cost_volume_depth', 96)
        self.p1_penalty = config.get('sgm_p1_penalty', 10.0)
        self.p2_penalty = config.get('sgm_p2_penalty', 120.0)
        
        # Refinement parameters
        self.lr_check = config.get('sgm_lr_check', True)
        self.lr_tolerance = config.get('sgm_lr_tolerance', 1)
        self.confidence_threshold = config.get('sgm_confidence_threshold', 0.7)
        self.bilateral_filter = config.get('sgm_bilateral_filter', True)
        self.spatial_sigma = config.get('sgm_spatial_sigma', 2.0)
        self.range_sigma = config.get('sgm_range_sigma', 10.0)
        
        # Temporal smoothing
        self.temporal_smoothing = config.get('sgm_temporal_smoothing', True)
        if self.temporal_smoothing:
            self.smoother = TemporalSmootherGPU(
                alpha=config.get('sgm_temporal_alpha', 0.3),
                motion_weight=config.get('sgm_temporal_motion_weight', 0.5)
            )
        
        logger.info(f"GPU SGM pipeline initialized: census={self.census_window}, "
                   f"max_disp={self.max_disparity}, lr_check={self.lr_check}")
    
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
        logger.debug("Computing GPU SGM disparity")
        
        # Convert to GPU arrays
        left_gpu = cp.asarray(left, dtype=cp.uint8)
        right_gpu = cp.asarray(right, dtype=cp.uint8)
        
        # 1. Census transform (GPU)
        left_census, right_census = census_transform_gpu(left_gpu, right_gpu, self.census_window)
        
        # 2. Hamming cost volume (GPU)
        cost_volume = hamming_cost_gpu(left_census, right_census, self.max_disparity)
        
        # 3. Cost volume refinement (GPU)
        cost_volume = refine_cost_volume_gpu(cost_volume)
        
        # 4. 8-path aggregation (GPU)
        cost_volume = aggregate_paths_gpu(cost_volume, self.p1_penalty, self.p2_penalty)
        
        # 5. Disparity refinement (temporarily CPU, will be GPU)
        cost_volume_cpu = cp.asnumpy(cost_volume)
        disparity = refine_disparity_cpu(
            cost_volume_cpu,
            lr_check=self.lr_check,
            lr_tolerance=self.lr_tolerance,
            confidence_threshold=self.confidence_threshold
        )
        disparity_gpu = cp.asarray(disparity, dtype=cp.float32)
        
        # 6. Bilateral filter (GPU)
        if self.bilateral_filter:
            disparity_gpu = bilateral_filter_gpu(
                disparity_gpu, left_gpu,
                spatial_sigma=self.spatial_sigma,
                range_sigma=self.range_sigma
            )
        
        # 7. Temporal smoothing (GPU)
        if self.temporal_smoothing:
            disparity_gpu = self.smoother.smooth(disparity_gpu, left_gpu, motion_score)
        
        return cp.asnumpy(disparity_gpu)
    
    def compute_depth(self, left: np.ndarray, right: np.ndarray,
                     baseline: float, focal_length: float,
                     motion_score: float = 0.0) -> np.ndarray:
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
        disparity = self.compute_disparity(left, right, motion_score)
        
        # Convert disparity to depth: depth = (baseline * focal_length) / disparity
        depth = (baseline * focal_length) / (disparity + 1e-6)
        depth[disparity <= 0] = 0.0
        
        return depth
