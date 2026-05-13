"""Persistent Global Gaussian Map for high-quality bubble storage."""

import numpy as np
from typing import Tuple, Optional, List
from ..utils.logger import get_logger
from ..utils.cupy_utils import cupy_manager

logger = get_logger(__name__)

class PersistentGlobalMap:
    """
    Persistent Global Gaussian Map maintains high-quality bubbles across the entire SLAM session.
    
    This map stores the most important bubbles that represent the global structure
    of the environment, separate from the transient local bubble map used for
    real-time processing.
    """
    
    def __init__(self, max_size: int = 100000, config: dict = None):
        """
        Initialize persistent global map.
        
        Args:
            max_size: Maximum number of bubbles to store
            config: Configuration dictionary
        """
        self.max_size = max_size
        self.config = config or {}
        
        # Global bubble storage
        self.mu = np.empty((0, 3), dtype=np.float64)          # Positions
        self.Sigma = np.empty((0, 3, 3), dtype=np.float64)      # Covariances
        self.weight = np.empty(0, dtype=np.float64)            # Weights
        self.color = np.empty((0, 3), dtype=np.float64)        # Colors
        self.age = np.empty(0, dtype=np.int32)                # Age counter
        self.importance = np.empty(0, dtype=np.float64)       # Importance scores
        
        self.chunk_size = self.config.get("chunk_size", 1.0)
        self.chunk_id = np.empty((0, 3), dtype=np.int32)      # 3D Chunk indices
        
        # Statistics
        self.total_inserted = 0
        self.total_pruned = 0
        self.merge_count = 0
        
        logger.info(f"Persistent Global Map initialized (max_size={max_size})")
    
    def add_bubbles(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                   color: np.ndarray, importance: np.ndarray = None):
        """
        Add new bubbles to the persistent global map.
        
        Args:
            mu: (N, 3) bubble positions
            Sigma: (N, 3, 3) bubble covariances
            weight: (N,) bubble weights
            color: (N, 3) bubble colors
            importance: (N,) importance scores (optional)
        """
        if len(mu) == 0:
            return
        
        # Calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(mu, Sigma, weight)
        
        # Merge with existing bubbles if needed
        if len(self.mu) > 0:
            mu, Sigma, weight, color, importance = self._merge_with_existing(
                mu, Sigma, weight, color, importance
            )
        
        # Add new bubbles
        self.mu = np.concatenate([self.mu, mu], axis=0)
        self.Sigma = np.concatenate([self.Sigma, Sigma], axis=0)
        self.weight = np.concatenate([self.weight, weight], axis=0)
        self.color = np.concatenate([self.color, color], axis=0)
        self.age = np.concatenate([self.age, np.zeros(len(mu), dtype=np.int32)], axis=0)
        self.importance = np.concatenate([self.importance, importance], axis=0)
        
        new_chunk_ids = np.floor(mu / self.chunk_size).astype(np.int32)
        self.chunk_id = np.concatenate([self.chunk_id, new_chunk_ids], axis=0)
        
        self.total_inserted += len(mu)
        
        # Enforce size limit
        if len(self.mu) > self.max_size:
            self._prune_to_limit()
        
        logger.debug(f"Global map: {len(self.mu)} bubbles (added {len(mu)})")
    
    def _calculate_importance(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Calculate importance scores for bubbles."""
        importance = weight.copy()
        
        # Add uncertainty penalty (lower uncertainty = higher importance)
        det_Sigma = np.linalg.det(Sigma)
        uncertainty_penalty = -np.log(np.maximum(det_Sigma, 1e-10))
        importance += 0.1 * uncertainty_penalty
        
        # Normalize to [0, 1]
        if len(importance) > 0:
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
        
        return importance
    
    def _merge_with_existing(self, mu_new: np.ndarray, Sigma_new: np.ndarray, 
                           weight_new: np.ndarray, color_new: np.ndarray, 
                           importance_new: np.ndarray) -> Tuple:
        """Merge new bubbles with existing ones based on proximity."""
        if len(self.mu) == 0:
            return mu_new, Sigma_new, weight_new, color_new, importance_new
        
        merge_distance = self.config.get("regional_merge_distance", 0.05)
        
        # Find nearby existing bubbles for each new bubble
        merged_mu = []
        merged_Sigma = []
        merged_weight = []
        merged_color = []
        merged_importance = []
        skip_indices = set()
        
        for i, pos_new in enumerate(mu_new):
            # Find nearby existing bubbles
            distances = np.linalg.norm(self.mu - pos_new, axis=1)
            nearby_mask = distances < merge_distance
            
            if np.any(nearby_mask):
                # Merge with nearby bubbles
                nearby_indices = np.where(nearby_mask)[0]
                
                # Weighted average of nearby bubbles
                weights_nearby = self.weight[nearby_indices]
                weights_nearby = weights_nearby / (weights_nearby.sum() + 1e-10)
                
                # Merge position
                merged_pos = np.sum(weights_nearby[:, None] * self.mu[nearby_indices], axis=0)
                merged_pos = (weight_new[i] * pos_new + np.sum(weights_nearby) * merged_pos) / (weight_new[i] + np.sum(weights_nearby))
                
                # Merge covariance (simplified)
                merged_sigma = Sigma_new[i] * 0.5 + np.mean(self.Sigma[nearby_indices], axis=0) * 0.5
                
                # Merge weight
                merged_w = weight_new[i] + np.sum(self.weight[nearby_indices])
                
                # Merge color
                merged_col = (color_new[i] * weight_new[i] + 
                             np.sum(self.color[nearby_indices] * self.weight[nearby_indices, None], axis=0)) / merged_w
                
                # Merge importance
                merged_imp = max(importance_new[i], np.max(self.importance[nearby_indices]))
                
                # Mark nearby bubbles for removal
                skip_indices.update(nearby_indices.tolist())
                
                merged_mu.append(merged_pos)
                merged_Sigma.append(merged_sigma)
                merged_weight.append(merged_w)
                merged_color.append(merged_col)
                merged_importance.append(merged_imp)
                
                self.merge_count += 1
            else:
                # No nearby bubbles, keep as is
                merged_mu.append(mu_new[i])
                merged_Sigma.append(Sigma_new[i])
                merged_weight.append(weight_new[i])
                merged_color.append(color_new[i])
                merged_importance.append(importance_new[i])
        
        # Convert back to arrays
        if merged_mu:
            result_mu = np.array(merged_mu)
            result_Sigma = np.array(merged_Sigma)
            result_weight = np.array(merged_weight)
            result_color = np.array(merged_color)
            result_importance = np.array(merged_importance)
            
            # Remove merged existing bubbles
            keep_mask = np.ones(len(self.mu), dtype=bool)
            keep_mask[list(skip_indices)] = False
            
            self.mu = self.mu[keep_mask]
            self.Sigma = self.Sigma[keep_mask]
            self.weight = self.weight[keep_mask]
            self.color = self.color[keep_mask]
            self.age = self.age[keep_mask]
            self.importance = self.importance[keep_mask]
            self.chunk_id = self.chunk_id[keep_mask]
            
            return result_mu, result_Sigma, result_weight, result_color, result_importance
        else:
            return mu_new, Sigma_new, weight_new, color_new, importance_new
    
    def _prune_to_limit(self):
        """Prune bubbles to maintain size limit based on importance."""
        if len(self.mu) <= self.max_size:
            return
        
        # Sort by importance (ascending - keep most important)
        importance_threshold = np.percentile(self.importance, 
                                          (1 - self.max_size / len(self.mu)) * 100)
        
        keep_mask = self.importance >= importance_threshold
        if np.sum(keep_mask) > self.max_size:
            # If still too many, keep top N by importance
            top_indices = np.argsort(self.importance)[-self.max_size:]
            keep_mask = np.zeros(len(self.mu), dtype=bool)
            keep_mask[top_indices] = True
        
        pruned_count = len(self.mu) - np.sum(keep_mask)
        self.mu = self.mu[keep_mask]
        self.Sigma = self.Sigma[keep_mask]
        self.weight = self.weight[keep_mask]
        self.color = self.color[keep_mask]
        self.age = self.age[keep_mask]
        self.importance = self.importance[keep_mask]
        self.chunk_id = self.chunk_id[keep_mask]
        
        self.total_pruned += pruned_count
        logger.debug(f"Global map pruned {pruned_count} bubbles")
    
    def get_high_quality_bubbles(self, threshold: float = 0.5) -> Tuple:
        """Get high-quality bubbles above importance threshold."""
        if len(self.mu) == 0:
            return self.mu, self.Sigma, self.weight, self.color
        
        mask = self.importance >= threshold
        return (self.mu[mask], self.Sigma[mask], self.weight[mask], self.color[mask])
    
    def get_bubbles_by_frustum(self, pose: np.ndarray, K: np.ndarray, width: int, height: int, active_set_selector) -> Tuple:
        """Get bubbles that are within the camera frustum."""
        if len(self.mu) == 0:
            return self.mu, self.Sigma, self.weight, self.color
        
        mask = active_set_selector.frustum_culling(self.mu, pose, K, width, height)
        return self.mu[mask], self.Sigma[mask], self.weight[mask], self.color[mask], mask
    
    def update_ages(self):
        """Update age counter for all bubbles."""
        self.age += 1
    
    def get_statistics(self) -> dict:
        """Get persistent map statistics."""
        return {
            'total_bubbles': len(self.mu),
            'total_inserted': self.total_inserted,
            'total_pruned': self.total_pruned,
            'merge_count': self.merge_count,
            'avg_importance': np.mean(self.importance) if len(self.importance) > 0 else 0.0,
            'avg_age': np.mean(self.age) if len(self.age) > 0 else 0.0
        }
    
    def clear(self):
        """Clear the persistent map."""
        self.mu = np.empty((0, 3), dtype=np.float64)
        self.Sigma = np.empty((0, 3, 3), dtype=np.float64)
        self.weight = np.empty(0, dtype=np.float64)
        self.color = np.empty((0, 3), dtype=np.float64)
        self.age = np.empty(0, dtype=np.int32)
        self.importance = np.empty(0, dtype=np.float64)
        self.chunk_id = np.empty((0, 3), dtype=np.int32)
        
        self.total_inserted = 0
        self.total_pruned = 0
        self.merge_count = 0
        
        logger.info("Persistent Global Map cleared")
