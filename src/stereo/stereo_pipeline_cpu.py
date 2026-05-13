"""Full CPU SGM stereo pipeline."""

import numpy as np
from ..utils.logger import get_logger

from .census_transform import census_transform_cpu
from .matching_cost import hamming_cost_cpu
from .cost_volume import build_cost_volume_cpu, refine_cost_volume_cpu
from .path_aggregation import aggregate_paths_cpu
from .disparity_refinement import refine_disparity_cpu
from .bilateral_filter import bilateral_filter_cpu
from .temporal_smoother import TemporalSmootherCPU

logger = get_logger(__name__)


class StereoPipelineCPU:
    """Full 8-path SGM stereo pipeline on CPU."""
    
    def __init__(self, config: dict):
        """
        Initialize CPU SGM pipeline.
        
        Parameters
        ----------
        config : dict
            Configuration parameters
        """
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
        self.temporal_smoothing = config.get('sgm_temporal_smoothing', False)
        if self.temporal_smoothing:
            self.smoother = TemporalSmootherCPU(
                alpha=config.get('sgm_temporal_alpha', 0.3),
                motion_weight=config.get('sgm_temporal_motion_weight', 0.5)
            )
        
        logger.info(f"CPU SGM pipeline initialized: census={self.census_window}, "
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
        logger.debug("Computing CPU SGM disparity")
        
        # 1. Census transform
        left_census, right_census = census_transform_cpu(left, right, self.census_window)
        
        # 2. Hamming cost volume
        cost_volume = hamming_cost_cpu(left_census, right_census, self.max_disparity)
        
        # 3. Cost volume refinement
        cost_volume = refine_cost_volume_cpu(cost_volume)
        
        # 4. 8-path aggregation
        cost_volume = aggregate_paths_cpu(cost_volume, self.p1_penalty, self.p2_penalty)
        
        # 5. Disparity refinement
        disparity = refine_disparity_cpu(
            cost_volume,
            lr_check=self.lr_check,
            lr_tolerance=self.lr_tolerance,
            confidence_threshold=self.confidence_threshold
        )
        
        # 6. Bilateral filter
        if self.bilateral_filter:
            disparity = bilateral_filter_cpu(
                disparity, left,
                spatial_sigma=self.spatial_sigma,
                range_sigma=self.range_sigma
            )
        
        # 7. Temporal smoothing
        if self.temporal_smoothing:
            disparity = self.smoother.smooth(disparity, left, motion_score)
        
        return disparity
    
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
