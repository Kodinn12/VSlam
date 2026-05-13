"""Multi-view keyframe manager for real SLAM reconstruction."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from ..utils.logger import get_logger
from .keyframe import Keyframe

logger = get_logger(__name__)

class KeyframeManager:
    """
    Multi-view keyframe manager for proper SLAM reconstruction.
    
    This manages a database of keyframes and enables multi-view fusion
    of Gaussian bubbles from different viewpoints.
    """
    
    def __init__(self, config: dict):
        """
        Initialize keyframe manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Keyframe storage
        self.keyframes: Dict[int, Keyframe] = {}
        self.keyframe_poses: Dict[int, np.ndarray] = {}
        self.keyframe_bubbles: Dict[int, Tuple] = {}  # (mu, Sigma, weight, color)
        
        # Multi-view fusion parameters
        self.max_keyframes = config.get("max_keyframes", 50)
        self.fusion_overlap_threshold = config.get("fusion_overlap_threshold", 0.3)
        self.fusion_distance_threshold = config.get("fusion_distance_threshold", 0.1)
        self.min_keyframe_features = config.get("min_keyframe_features", 50)
        self.min_depth_coverage = config.get("min_depth_coverage", 0.3)
        
        # Statistics
        self.total_keyframes_added = 0
        self.total_fusions = 0
        self.rejected_keyframes = 0
        
        # Temporal consistency parameters
        self.age_decay_rate = config.get("age_decay_rate", 0.01)  # Decay rate per frame
        self.max_keyframe_age = config.get("max_keyframe_age", 1000)  # Maximum age in frames
        self.keyframe_ages = {}  # Track age of each keyframe
        self.frame_counter = 0
        
        logger.info(f"KeyframeManager initialized (max_keyframes={self.max_keyframes})")
    
    def add_keyframe(self, keyframe: Keyframe, bubbles: Tuple = None) -> bool:
        """
        Add a new keyframe to the database.
        
        Args:
            keyframe: Keyframe object
            bubbles: Tuple of (mu, Sigma, weight, color) for this keyframe
            
        Returns:
            True if keyframe was added, False if rejected
        """
        # Check if we already have this keyframe
        if keyframe.id in self.keyframes:
            logger.warning(f"Keyframe {keyframe.id} already exists")
            return False
        
        # Check keyframe quality
        if not self._is_good_keyframe(keyframe):
            logger.debug(f"Keyframe {keyframe.id} rejected: poor quality")
            self.rejected_keyframes += 1
            return False
        
        # Add to database with age tracking
        self.keyframes[keyframe.id] = keyframe
        self.keyframe_poses[keyframe.id] = keyframe.pose.copy()
        if bubbles is not None:
            self.keyframe_bubbles[keyframe.id] = bubbles
        self.keyframe_ages[keyframe.id] = 0  # New keyframe starts at age 0
        
        self.total_keyframes_added += 1
        
        # Manage keyframe count
        if len(self.keyframes) > self.max_keyframes:
            self._prune_keyframes()
        
        logger.debug(f"Added keyframe {keyframe.id} (total: {len(self.keyframes)})")
        return True
    
    def _is_good_keyframe(self, keyframe: Keyframe) -> bool:
        """Check if keyframe meets quality criteria."""
        # Check if we have enough features
        if keyframe.keypoints is None or len(keyframe.keypoints) < self.min_keyframe_features:
            return False
        
        # Check depth quality
        if keyframe.depth is None:
            return False
        
        # Check depth coverage (percentage of valid depth pixels)
        valid_depth = np.sum(keyframe.depth > 0)
        total_depth = keyframe.depth.size
        depth_coverage = valid_depth / total_depth
        
        if depth_coverage < self.min_depth_coverage:
            return False
        
        # Check pose validity
        if keyframe.pose is None or not np.isfinite(keyframe.pose).all():
            return False
        
        return True
    
    def _prune_keyframes(self):
        """Remove redundant keyframes to maintain database size with age-aware pruning."""
        if len(self.keyframes) <= self.max_keyframes:
            return
        
        # Strategy: keep spatially diverse keyframes, prioritize recent ones
        keyframe_ids = list(self.keyframes.keys())
        
        # Calculate spatial distribution and age scores
        poses = np.array([self.keyframe_poses[kf_id] for kf_id in keyframe_ids])
        positions = poses[:, :3, 3]
        ages = np.array([self.keyframe_ages.get(kf_id, 0) for kf_id in keyframe_ids])
        
        # Age-based scoring: younger keyframes get higher scores
        age_scores = np.exp(-ages * self.age_decay_rate)  # Exponential decay
        
        # Keep keyframes that are well-distributed spatially and recent
        kept_ids = []
        for i, kf_id in enumerate(keyframe_ids):
            if i == 0:
                kept_ids.append(kf_id)  # Always keep first
                continue
            
            # Check distance to already kept keyframes
            current_pos = positions[i]
            min_distance = float('inf')
            for kept_id in kept_ids:
                kept_pos = self.keyframe_poses[kept_id][:3, 3]
                distance = np.linalg.norm(current_pos - kept_pos)
                min_distance = min(min_distance, distance)
            
            # Keep if far enough from existing keyframes OR if very recent
            spatial_threshold = self.fusion_distance_threshold
            age_bonus = age_scores[i] > 0.5  # Recent keyframes get bonus
            
            if min_distance > spatial_threshold or age_bonus:
                kept_ids.append(kf_id)
        
        # Remove pruned keyframes
        for kf_id in keyframe_ids:
            if kf_id not in kept_ids:
                del self.keyframes[kf_id]
                del self.keyframe_poses[kf_id]
                if kf_id in self.keyframe_bubbles:
                    del self.keyframe_bubbles[kf_id]
                if kf_id in self.keyframe_ages:
                    del self.keyframe_ages[kf_id]
        
        logger.info(f"Pruned keyframes: {len(keyframe_ids)} -> {len(kept_ids)} (age-aware)")

    def update_temporal_consistency(self):
        """Update age-based temporal consistency for all keyframes."""
        self.frame_counter += 1
        
        # Update ages and apply decay
        expired_keyframes = []
        for kf_id in list(self.keyframe_ages.keys()):
            self.keyframe_ages[kf_id] += 1
            
            # Check if keyframe is too old
            if self.keyframe_ages[kf_id] > self.max_keyframe_age:
                expired_keyframes.append(kf_id)
            else:
                # Apply age decay to bubble weights
                if kf_id in self.keyframe_bubbles:
                    mu, Sigma, weight, color = self.keyframe_bubbles[kf_id]
                    age_factor = np.exp(-self.keyframe_ages[kf_id] * self.age_decay_rate)
                    decayed_weight = weight * age_factor
                    self.keyframe_bubbles[kf_id] = (mu, Sigma, decayed_weight, color)
        
        # Remove expired keyframes
        for kf_id in expired_keyframes:
            if kf_id in self.keyframes:
                del self.keyframes[kf_id]
                del self.keyframe_poses[kf_id]
                if kf_id in self.keyframe_bubbles:
                    del self.keyframe_bubbles[kf_id]
                del self.keyframe_ages[kf_id]
        
        if expired_keyframes:
            logger.debug(f"Removed {len(expired_keyframes)} expired keyframes")
    
    def get_overlapping_keyframes(self, pose: np.ndarray, radius: float = 5.0) -> List[int]:
        """
        Get keyframes that overlap with given pose.
        
        Args:
            pose: Current camera pose (4x4 matrix)
            radius: Search radius in meters
            
        Returns:
            List of overlapping keyframe IDs
        """
        overlapping = []
        current_pos = pose[:3, 3]
        
        for kf_id, kf_pose in self.keyframe_poses.items():
            kf_pos = kf_pose[:3, 3]
            distance = np.linalg.norm(current_pos - kf_pos)
            
            if distance < radius:
                overlapping.append(kf_id)
        
        return overlapping
    
    def fuse_multi_view_bubbles(self, current_bubbles: Tuple, pose: np.ndarray) -> Tuple:
        """
        Fuse current bubbles with overlapping keyframe bubbles.
        
        Args:
            current_bubbles: (mu, Sigma, weight, color) from current frame
            pose: Current camera pose
            
        Returns:
            Fused (mu, Sigma, weight, color)
        """
        mu_current, Sigma_current, weight_current, color_current = current_bubbles
        
        if len(mu_current) == 0:
            return current_bubbles
        
        # Get overlapping keyframes
        overlapping_kfs = self.get_overlapping_keyframes(pose)
        
        if not overlapping_kfs:
            return current_bubbles
        
        # Collect bubbles from overlapping keyframes
        all_mu = [mu_current]
        all_Sigma = [Sigma_current]
        all_weight = [weight_current]
        all_color = [color_current]
        
        for kf_id in overlapping_kfs:
            if kf_id in self.keyframe_bubbles:
                kf_mu, kf_Sigma, kf_weight, kf_color = self.keyframe_bubbles[kf_id]
                if len(kf_mu) > 0:
                    all_mu.append(kf_mu)
                    all_Sigma.append(kf_Sigma)
                    all_weight.append(kf_weight)
                    all_color.append(kf_color)

        # No stored keyframe bubbles overlapped this frame. Do not collapse the
        # current frame by spatially fusing it with itself; the live bubble map
        # performs its own incremental fusion after this step.
        if len(all_mu) == 1:
            return current_bubbles
        
        # Concatenate all bubbles
        fused_mu = np.concatenate(all_mu, axis=0)
        fused_Sigma = np.concatenate(all_Sigma, axis=0)
        fused_weight = np.concatenate(all_weight, axis=0)
        fused_color = np.concatenate(all_color, axis=0)
        
        # Apply weighted fusion for overlapping bubbles
        fused_mu, fused_Sigma, fused_weight, fused_color = self._weighted_fusion(
            fused_mu, fused_Sigma, fused_weight, fused_color
        )
        
        self.total_fusions += 1
        logger.debug(f"Multi-view fusion: {len(mu_current)} -> {len(fused_mu)} bubbles")
        
        # Ensure we return appropriate types (if input was CuPy, output should be too)
        return fused_mu, fused_Sigma, fused_weight, fused_color
    
    def _weighted_fusion(self, mu: np.ndarray, Sigma: np.ndarray, 
                        weight: np.ndarray, color: np.ndarray) -> Tuple:
        """
        Apply weighted fusion to overlapping bubbles.
        Uses GPU-accelerated path if available, otherwise falls back to CPU.
        """
        if len(mu) < 10:  # Skip fusion for tiny sets
            return mu, Sigma, weight, color
            
        from ..utils.cupy_utils import cupy_manager, USE_CUPY
        if USE_CUPY and cupy_manager.is_available():
            try:
                return self._weighted_fusion_gpu(mu, Sigma, weight, color)
            except Exception as e:
                logger.warning(f"GPU weighted fusion failed: {e}. Falling back to CPU.")
                return self._weighted_fusion_cpu(mu, Sigma, weight, color)
        else:
            return self._weighted_fusion_cpu(mu, Sigma, weight, color)

    def _weighted_fusion_gpu(self, mu, Sigma, weight, color):
        """GPU-accelerated spatial weighted fusion using vectorized CuPy operations."""
        from ..utils.cupy_utils import cupy_manager
        xp = cupy_manager.get_array_module(True)
        
        # Ensure data is on GPU
        mu_g = cupy_manager.to_gpu(mu)
        Sig_g = cupy_manager.to_gpu(Sigma)
        w_g = cupy_manager.to_gpu(weight)
        c_g = cupy_manager.to_gpu(color)
        
        cell_size = self.config.get("fusion_distance_threshold", 0.05)
        
        # 1. Assign each bubble to a voxel cell
        grid_coords = xp.floor(mu_g / cell_size).astype(xp.int32)
        OFFSET = 10000
        stride_y = 20001
        stride_z = 20001 * 20001
        cell_keys = ((grid_coords[:, 0] + OFFSET).astype(xp.int64) +
                     (grid_coords[:, 1] + OFFSET).astype(xp.int64) * stride_y +
                     (grid_coords[:, 2] + OFFSET).astype(xp.int64) * stride_z)
        
        # 2. Get unique cells and mapping
        unique_keys, labels = xp.unique(cell_keys, return_inverse=True)
        n_groups = len(unique_keys)
        
        # 3. Vectorized weighted accumulation
        sum_w = xp.zeros(n_groups, dtype=xp.float64)
        xp.add.at(sum_w, labels, w_g)
        sum_w += 1e-10 # epsilon
        
        fused_mu = xp.zeros((n_groups, 3), dtype=xp.float64)
        xp.add.at(fused_mu, (labels, slice(None)), mu_g * w_g[:, None])
        fused_mu /= sum_w[:, None]
        
        fused_c = xp.zeros((n_groups, 3), dtype=xp.float64)
        xp.add.at(fused_c, (labels, slice(None)), c_g * w_g[:, None])
        fused_c /= sum_w[:, None]
        
        fused_Sig = xp.zeros((n_groups, 3, 3), dtype=xp.float64)
        weighted_Sig_flat = (Sig_g * w_g[:, None, None]).reshape(-1, 9)
        fused_Sig_flat = xp.zeros((n_groups, 9), dtype=xp.float64)
        xp.add.at(fused_Sig_flat, (labels, slice(None)), weighted_Sig_flat)
        fused_Sig = (fused_Sig_flat / sum_w[:, None]).reshape(-1, 3, 3)
        
        # If input was NumPy, return NumPy. If it was CuPy, return CuPy.
        if isinstance(mu, np.ndarray):
            return cupy_manager.to_cpu(fused_mu), cupy_manager.to_cpu(fused_Sig), \
                   cupy_manager.to_cpu(sum_w - 1e-10), cupy_manager.to_cpu(fused_c)
        return fused_mu, fused_Sig, sum_w - 1e-10, fused_c

    def _weighted_fusion_cpu(self, mu: np.ndarray, Sigma: np.ndarray, 
                            weight: np.ndarray, color: np.ndarray) -> Tuple:
        """Original O(N^2) CPU fusion loop (kept as fallback for small sets)."""
        if len(mu) < 1: return mu, Sigma, weight, color
        
        # Convert to numpy if needed
        from ..utils.array_utils import to_numpy_safe
        mu = to_numpy_safe(mu)
        Sigma = to_numpy_safe(Sigma)
        weight = to_numpy_safe(weight)
        color = to_numpy_safe(color)

        fusion_threshold = self.fusion_distance_threshold
        fused_mu, fused_Sigma, fused_weight, fused_color = [], [], [], []
        used_indices = set()
        
        for i in range(len(mu)):
            if i in used_indices: continue
            distances = np.linalg.norm(mu - mu[i], axis=1)
            nearby_mask = (distances < fusion_threshold)
            nearby_indices = np.where(nearby_mask)[0]
            
            all_indices = nearby_indices.tolist()
            weights_all = weight[all_indices]
            weights_norm = weights_all / (weights_all.sum() + 1e-10)
            
            fused_mu.append(np.sum(weights_norm[:, None] * mu[all_indices], axis=0))
            fused_Sigma.append(np.sum(weights_norm[:, None, None] * Sigma[all_indices], axis=0))
            fused_weight.append(weights_all.sum())
            fused_color.append(np.sum(weights_norm[:, None] * color[all_indices], axis=0))
            used_indices.update(all_indices)
        
        return np.array(fused_mu), np.array(fused_Sigma), np.array(fused_weight), np.array(fused_color)
    
    def get_global_reconstruction(self) -> Tuple:
        """
        Get globally fused reconstruction from all keyframes.
        
        Returns:
            (mu, Sigma, weight, color) for global reconstruction
        """
        if not self.keyframe_bubbles:
            return np.empty((0, 3)), np.empty((0, 3, 3)), np.empty(0), np.empty((0, 3))
        
        # Collect all bubbles from all keyframes
        all_mu = []
        all_Sigma = []
        all_weight = []
        all_color = []
        
        for kf_id, (kf_mu, kf_Sigma, kf_weight, kf_color) in self.keyframe_bubbles.items():
            if len(kf_mu) > 0:
                all_mu.append(kf_mu)
                all_Sigma.append(kf_Sigma)
                all_weight.append(kf_weight)
                all_color.append(kf_color)
        
        if not all_mu:
            return np.empty((0, 3)), np.empty((0, 3, 3)), np.empty(0), np.empty((0, 3))
        
        # Concatenate and fuse
        global_mu = np.concatenate(all_mu, axis=0)
        global_Sigma = np.concatenate(all_Sigma, axis=0)
        global_weight = np.concatenate(all_weight, axis=0)
        global_color = np.concatenate(all_color, axis=0)
        
        # Apply global fusion
        global_mu, global_Sigma, global_weight, global_color = self._weighted_fusion(
            global_mu, global_Sigma, global_weight, global_color
        )
        
        logger.info(f"Global reconstruction: {len(global_mu)} bubbles from {len(self.keyframes)} keyframes")
        
        return global_mu, global_Sigma, global_weight, global_color
    
    def apply_loop_closure_correction(self, corrected_poses: dict):
        """
        Apply loop closure corrections to keyframes for global consistency.
        
        Args:
            corrected_poses: Dictionary mapping keyframe_id -> corrected_pose
        """
        corrected_count = 0
        for kf_id, corrected_pose in corrected_poses.items():
            if kf_id in self.keyframes:
                old_pose = self.keyframe_poses[kf_id].copy()
                self.keyframe_poses[kf_id] = corrected_pose.copy()
                self.keyframes[kf_id].pose = corrected_pose.copy()
                corrected_count += 1
                
                # Update bubble positions if this keyframe has bubbles
                if kf_id in self.keyframe_bubbles:
                    mu, Sigma, weight, color = self.keyframe_bubbles[kf_id]
                    # Transform bubbles to corrected world coordinates
                    T_correction = corrected_pose @ np.linalg.inv(old_pose)
                    corrected_mu = self._transform_bubbles(mu, T_correction)
                    self.keyframe_bubbles[kf_id] = (corrected_mu, Sigma, weight, color)
        
        logger.info(f"Applied loop closure corrections to {corrected_count} keyframes")
        return corrected_count

    def _transform_bubbles(self, mu: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Transform bubble positions using given transformation matrix.
        
        Args:
            mu: (N, 3) bubble positions
            transform: (4, 4) transformation matrix
            
        Returns:
            Transformed bubble positions
        """
        # Convert to homogeneous coordinates
        ones = np.ones((len(mu), 1))
        mu_homo = np.hstack([mu, ones])
        
        # Apply transformation
        mu_transformed_homo = (transform @ mu_homo.T).T
        
        # Convert back to 3D
        return mu_transformed_homo[:, :3]

    def get_loop_closure_candidates(self, current_pose: np.ndarray, 
                                   min_distance: float = 2.0, 
                                   max_distance: float = 10.0) -> List[int]:
        """
        Get keyframe candidates for loop closure detection.
        
        Args:
            current_pose: Current camera pose
            min_distance: Minimum distance to avoid recent keyframes
            max_distance: Maximum distance for loop closure
            
        Returns:
            List of candidate keyframe IDs
        """
        candidates = []
        current_pos = current_pose[:3, 3]
        
        for kf_id, kf_pose in self.keyframe_poses.items():
            kf_pos = kf_pose[:3, 3]
            distance = np.linalg.norm(current_pos - kf_pos)
            
            # Check if within distance range and not too recent
            age = self.keyframe_ages.get(kf_id, 0)
            if (min_distance < distance < max_distance and 
                age > 20):  # Only consider keyframes older than 20 frames
                candidates.append(kf_id)
        
        return candidates

    def get_statistics(self) -> dict:
        """Get keyframe manager statistics."""
        return {
            'total_keyframes': len(self.keyframes),
            'total_keyframes_added': self.total_keyframes_added,
            'total_fusions': self.total_fusions,
            'rejected_keyframes': self.rejected_keyframes,
            'keyframes_with_bubbles': len(self.keyframe_bubbles),
            'average_bubbles_per_keyframe': np.mean([len(b[0]) for b in self.keyframe_bubbles.values()]) if self.keyframe_bubbles else 0,
            'average_keyframe_age': np.mean(list(self.keyframe_ages.values())) if self.keyframe_ages else 0,
            'oldest_keyframe_age': max(self.keyframe_ages.values()) if self.keyframe_ages else 0
        }
    
    def clear(self):
        """Clear all keyframes."""
        self.keyframes.clear()
        self.keyframe_poses.clear()
        self.keyframe_bubbles.clear()
        self.total_keyframes_added = 0
        self.total_fusions = 0
        self.rejected_keyframes = 0
        logger.info("KeyframeManager cleared")
