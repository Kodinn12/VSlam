"""Hierarchical Bubble Pruning System for efficient memory management."""

import numpy as np
from typing import Tuple, Optional
from ..utils.logger import get_logger
from ..utils.cupy_utils import cupy_manager
from .persistent_global_map import PersistentGlobalMap

logger = get_logger(__name__)

class HierarchicalPruner:
    """
    Three-level hierarchical bubble pruning system:
    
    Level 1 (Local): Remove noise and low-quality bubbles
    Level 2 (Regional): Merge redundant bubbles in spatial proximity  
    Level 3 (Global): Select important bubbles for persistent map
    """
    
    def __init__(self, config: dict):
        """
        Initialize hierarchical pruner.
        
        Args:
            config: Configuration dictionary with pruning parameters
        """
        self.config = config
        self.cfg    = config   # alias used by Level-0 sanity pruning
        
        # Pruning thresholds
        self.local_threshold = config.get("local_pruning_threshold", 0.1)
        self.regional_merge_distance = config.get("regional_merge_distance", 0.05)
        self.global_importance_threshold = config.get("global_importance_threshold", 0.5)
        
        # Size limits
        self.hard_limit = config.get("hard_bubble_limit", 500000)
        self.batch_size = config.get("pruning_batch_size", 10000)
        
        # Persistent global map
        self.global_map = PersistentGlobalMap(
            max_size=config.get("persistent_global_map_size", 100000),
            config=config
        )
        
        # Statistics
        self.pruning_stats = {
            'local_pruned': 0,
            'regional_merged': 0,
            'global_promoted': 0,
            'total_processed': 0
        }
        
        logger.info("Hierarchical Pruner initialized")
    
    def prune_bubbles(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                     color: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply three-level hierarchical pruning to bubbles.
        
        Args:
            mu: (N, 3) bubble positions
            Sigma: (N, 3, 3) bubble covariances
            weight: (N,) bubble weights
            color: (N, 3) bubble colors
            
        Returns:
            Tuple of pruned (mu, Sigma, weight, color)
        """
        if len(mu) == 0:
            return mu, Sigma, weight, color
        
        original_count = len(mu)
        self.pruning_stats['total_processed'] += original_count

        # ── Level 0: Sanity pruning ─────────────────────────────────────────
        # Remove truly invalid bubbles BEFORE any merging or promotion.
        #   a) NaN / Inf positions
        #   b) Degenerate covariance (trace too large -> extreme uncertainty)
        #   c) Extreme spatial outliers (> 4 std-deviations from centroid)
        valid = np.all(np.isfinite(mu), axis=1)                   # no NaN/Inf
        valid &= np.all(np.isfinite(weight))                       # no NaN weight

        if np.any(valid):
            # Degenerate covariance: trace > threshold means bubble is extremely uncertain
            max_trace = self.cfg.get('max_sigma_trace', 2.0)
            traces = np.trace(Sigma, axis1=1, axis2=2)             # (N,)
            valid &= (traces < max_trace) & np.isfinite(traces)

        if np.any(valid) and np.sum(valid) > 10:
            # Spatial outlier removal: drop bubbles > 4 std from the centroid
            mu_valid = mu[valid]
            centroid = np.median(mu_valid, axis=0)
            std      = np.std(mu_valid, axis=0).clip(0.01)         # avoid div-by-zero
            dist     = np.max(np.abs((mu - centroid) / std), axis=1)
            valid   &= dist < 4.0

        if not np.all(valid):
            sanity_removed = np.sum(~valid)
            logger.debug(f"[Pruner] Level-0 sanity removed {sanity_removed} invalid/outlier bubbles")
            mu, Sigma, weight, color = mu[valid], Sigma[valid], weight[valid], color[valid]

        if len(mu) == 0:
            return mu, Sigma, weight, color
        # ────────────────────────────────────────────────────────────────────
        
        # Level 1: Local pruning (noise removal)
        mu_local, Sigma_local, weight_local, color_local = self._local_pruning(mu, Sigma, weight, color)
        
        # Level 2: Regional pruning (merge redundancy)
        mu_regional, Sigma_regional, weight_regional, color_regional = self._regional_pruning(
            mu_local, Sigma_local, weight_local, color_local
        )
        
        # Level 3: Global pruning (importance selection + persistent map)
        mu_global, Sigma_global, weight_global, color_global = self._global_pruning(
            mu_regional, Sigma_regional, weight_regional, color_regional
        )
        
        # Enforce hard limit
        if len(mu_global) > self.hard_limit:
            mu_global, Sigma_global, weight_global, color_global = self._enforce_hard_limit(
                mu_global, Sigma_global, weight_global, color_global
            )
        
        final_count = len(mu_global)
        logger.debug(f"Hierarchical pruning: {original_count} -> {final_count} bubbles")
        
        return mu_global, Sigma_global, weight_global, color_global
    
    def _local_pruning(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                      color: np.ndarray) -> Tuple:
        """Level 1: Remove noise and low-quality bubbles."""
        if len(mu) == 0:
            return mu, Sigma, weight, color
        
        # Calculate quality scores
        scores = self._calculate_quality_scores(mu, Sigma, weight)
        
        # Remove low-quality bubbles
        keep_mask = scores >= self.local_threshold
        pruned_count = len(mu) - np.sum(keep_mask)
        self.pruning_stats['local_pruned'] += pruned_count
        
        return (mu[keep_mask], Sigma[keep_mask], weight[keep_mask], color[keep_mask])
    
    def _regional_pruning(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                         color: np.ndarray) -> Tuple:
        """Level 2: Merge redundant bubbles in spatial proximity."""
        if len(mu) == 0:
            return mu, Sigma, weight, color
        
        # Use GPU for spatial clustering if available
        use_gpu = cupy_manager.is_available()
        
        if use_gpu and len(mu) > self.batch_size:
            mu_merged, Sigma_merged, weight_merged, color_merged = self._regional_pruning_gpu(
                mu, Sigma, weight, color
            )
        else:
            mu_merged, Sigma_merged, weight_merged, color_merged = self._regional_pruning_cpu(
                mu, Sigma, weight, color
            )
        
        merged_count = len(mu) - len(mu_merged)
        self.pruning_stats['regional_merged'] += merged_count
        
        return mu_merged, Sigma_merged, weight_merged, color_merged
    
    def _regional_pruning_cpu(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                             color: np.ndarray) -> Tuple:
        """CPU-based regional pruning using spatial clustering."""
        merged_mu = []
        merged_Sigma = []
        merged_weight = []
        merged_color = []
        processed = set()
        
        for i in range(len(mu)):
            if i in processed:
                continue
            
            # Find nearby bubbles
            distances = np.linalg.norm(mu - mu[i], axis=1)
            nearby_mask = (distances < self.regional_merge_distance) & (np.arange(len(mu)) != i)
            nearby_indices = np.where(nearby_mask)[0]
            
            if len(nearby_indices) > 0:
                # Merge with nearby bubbles
                all_indices = [i] + nearby_indices.tolist()
                weights_all = weight[all_indices]
                weights_norm = weights_all / (weights_all.sum() + 1e-10)
                
                # Weighted average
                merged_pos = np.sum(weights_norm[:, None] * mu[all_indices], axis=0)
                merged_sigma = np.sum(weights_norm[:, None, None] * Sigma[all_indices], axis=0)
                merged_w = weights_all.sum()
                merged_col = np.sum(weights_norm[:, None] * color[all_indices], axis=0)
                
                merged_mu.append(merged_pos)
                merged_Sigma.append(merged_sigma)
                merged_weight.append(merged_w)
                merged_color.append(merged_col)
                
                processed.update(all_indices)
            else:
                # No nearby bubbles, keep as is
                merged_mu.append(mu[i])
                merged_Sigma.append(Sigma[i])
                merged_weight.append(weight[i])
                merged_color.append(color[i])
                processed.add(i)
        
        return (
            np.array(merged_mu),
            np.array(merged_Sigma),
            np.array(merged_weight),
            np.array(merged_color)
        )
    
    def _regional_pruning_gpu(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                             color: np.ndarray) -> Tuple:
        """GPU-based regional pruning for large datasets with shape validation."""
        try:
            # Use cupy_manager to get appropriate GPU backend
            xp = cupy_manager.get_array_module(use_gpu=True)
            
            # Validate input shapes
            if len(mu.shape) != 2 or mu.shape[1] != 3:
                raise ValueError(f"Invalid mu shape: {mu.shape}, expected (N, 3)")
            if len(Sigma.shape) != 3 or Sigma.shape[1:] != (3, 3):
                raise ValueError(f"Invalid Sigma shape: {Sigma.shape}, expected (N, 3, 3)")
            if len(weight.shape) != 1:
                raise ValueError(f"Invalid weight shape: {weight.shape}, expected (N,)")
            if len(color.shape) != 2 or color.shape[1] != 3:
                raise ValueError(f"Invalid color shape: {color.shape}, expected (N, 3)")
            
            # Check consistent batch sizes
            N = mu.shape[0]
            if Sigma.shape[0] != N or weight.shape[0] != N or color.shape[0] != N:
                raise ValueError(f"Inconsistent batch sizes: mu={N}, Sigma={Sigma.shape[0]}, weight={weight.shape[0]}, color={color.shape[0]}")
            
            # Transfer to GPU
            mu_gpu = xp.asarray(mu)
            Sigma_gpu = xp.asarray(Sigma)
            weight_gpu = xp.asarray(weight)
            color_gpu = xp.asarray(color)
            
            # Simple distance-based clustering on GPU
            grid_size = self.regional_merge_distance
            grid_coords = (mu_gpu / grid_size).astype(xp.int32)
            
            # Find unique grid cells
            unique_cells, inverse_indices = xp.unique(grid_coords, axis=0, return_inverse=True)
            
            merged_mu = []
            merged_Sigma = []
            merged_weight = []
            merged_color = []
            
            # Note: For-loop over cells on GPU is slow, but better than CPU for large N
            # For extremely large N, this should be vectorized further
            for cell_id in range(len(unique_cells)):
                mask = inverse_indices == cell_id
                mask_count = xp.sum(mask)
                
                if mask_count == 1:
                    # Single bubble in cell, keep as is
                    idx = xp.where(mask)[0][0]
                    merged_mu.append(mu_gpu[idx])
                    merged_Sigma.append(Sigma_gpu[idx])
                    merged_weight.append(weight_gpu[idx])
                    merged_color.append(color_gpu[idx])
                elif mask_count > 1:
                    # Multiple bubbles, merge them
                    indices = xp.where(mask)[0]
                    weights_all = weight_gpu[indices]
                    weights_norm = weights_all / (weights_all.sum() + 1e-10)
                    
                    # Proper broadcasting for weighted averages
                    merged_pos = xp.sum(weights_norm[:, None] * mu_gpu[indices], axis=0)
                    merged_sigma = xp.sum(weights_norm[:, None, None] * Sigma_gpu[indices], axis=0)
                    merged_w = weights_all.sum()
                    merged_col = xp.sum(weights_norm[:, None] * color_gpu[indices], axis=0)
                    
                    merged_mu.append(merged_pos)
                    merged_Sigma.append(merged_sigma)
                    merged_weight.append(merged_w)
                    merged_color.append(merged_col)
            
            # Transfer back to CPU with proper error handling
            if len(merged_mu) == 0:
                return np.empty((0, 3)), np.empty((0, 3, 3)), np.empty(0), np.empty((0, 3))
            
            merged_mu = cupy_manager.to_cpu(xp.stack(merged_mu))
            merged_Sigma = cupy_manager.to_cpu(xp.stack(merged_Sigma))
            merged_weight = cupy_manager.to_cpu(xp.stack(merged_weight))
            merged_color = cupy_manager.to_cpu(xp.stack(merged_color))
            
            return merged_mu, merged_Sigma, merged_weight, merged_color
            
        except Exception as e:
            print(f"[GPU PRUNING ERROR] {e}")
            # Fallback to CPU
            return self._regional_pruning_cpu(mu, Sigma, weight, color)
    
    def _global_pruning(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                        color: np.ndarray) -> Tuple:
        """Level 3: Promote important bubbles to persistent map, but keep ALL bubbles active.
        
        Previously this removed promoted bubbles from the returned set, causing ~50% of
        bubbles to vanish from the visualization each prune cycle. Now we keep all bubbles
        active while also storing important ones in the persistent global map.
        """
        if len(mu) == 0:
            return mu, Sigma, weight, color
        
        # Calculate importance scores
        importance = self._calculate_importance_scores(mu, Sigma, weight)
        
        # Select bubbles for persistent global map (promote important ones)
        global_mask = importance >= self.global_importance_threshold
        global_mu = mu[global_mask]
        global_Sigma = Sigma[global_mask]
        global_weight = weight[global_mask]
        global_color = color[global_mask]
        global_importance = importance[global_mask]
        
        # Add to persistent global map for storage/loop-closure
        if len(global_mu) > 0:
            self.global_map.add_bubbles(global_mu, global_Sigma, global_weight, global_color, global_importance)
        
        promoted_count = len(global_mu)
        self.pruning_stats['global_promoted'] += promoted_count
        
        # Return ALL bubbles (not just the non-promoted ones)
        # Removing promoted bubbles was causing massive density loss in visualization
        return mu, Sigma, weight, color
    
    def _calculate_quality_scores(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Calculate quality scores for local pruning."""
        scores = weight.copy()
        
        # Add uncertainty penalty
        det_Sigma = np.linalg.det(Sigma)
        uncertainty_penalty = -np.log(np.maximum(det_Sigma, 1e-10))
        scores += 0.1 * uncertainty_penalty
        
        # Add distance penalty (bubbles too far from origin might be outliers)
        distances = np.linalg.norm(mu, axis=1)
        distance_penalty = np.exp(-distances / 10.0)  # Favor bubbles closer to origin
        scores *= distance_penalty
        
        # Normalize to [0, 1]
        if len(scores) > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        return scores
    
    def _calculate_importance_scores(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Calculate semantic importance scores for intelligent pruning."""
        importance = weight.copy()
        
        # Add uncertainty penalty (lower uncertainty = higher importance)
        det_Sigma = np.linalg.det(Sigma)
        uncertainty_penalty = -np.log(np.maximum(det_Sigma, 1e-10))
        importance += 0.2 * uncertainty_penalty
        
        # Add spatial diversity bonus (bubbles in sparse regions are more important)
        if len(mu) > 1:
            distances = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            diversity_bonus = min_distances / (np.max(min_distances) + 1e-10)
            importance += 0.3 * diversity_bonus
        
        # Add structural importance (edges, corners, planar features)
        structural_score = self._calculate_structural_importance(mu, Sigma)
        importance += 0.25 * structural_score
        
        # Add temporal stability score
        temporal_score = self._calculate_temporal_importance(len(mu))
        importance += 0.15 * temporal_score
        
        # Normalize to [0, 1]
        if len(importance) > 0:
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
        
        return importance

    def _calculate_structural_importance(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Calculate structural importance based on geometric features."""
        structural_score = np.zeros(len(mu))
        
        if len(mu) < 50:
            return structural_score
        
        try:
            # Local density analysis
            sample_size = min(100, len(mu))
            sample_indices = np.random.choice(len(mu), sample_size, replace=False)
            sample_mu = mu[sample_indices]
            
            for i in range(len(mu)):
                distances = np.linalg.norm(sample_mu - mu[i], axis=1)
                mean_dist = distances.mean()
                # Edge regions have higher mean distances
                structural_score[i] = mean_dist / (np.max(distances) + 1e-10)
            
            # Planarity analysis using covariance eigenvalues
            planar_score = np.zeros(len(mu))
            for i in range(min(len(mu), 1000)):  # Limit for performance
                eigenvals = np.linalg.eigvals(Sigma[i])
                eigenvals = np.sort(eigenvals)[::-1]
                if eigenvals[2] > 1e-10:
                    planarity = 1.0 - (eigenvals[2] / eigenvals[0])
                    planar_score[i] = planarity
            
            # Combine structural factors
            structural_score = 0.6 * structural_score + 0.4 * planar_score
            
        except Exception as e:
            print(f"[STRUCTURAL] Error: {e}")
            structural_score = np.zeros(len(mu))
        
        return structural_score

    def _calculate_temporal_importance(self, num_bubbles: int) -> np.ndarray:
        """Calculate temporal importance based on stability."""
        if num_bubbles == 0:
            return np.array([])
        
        # Newer bubbles get higher temporal importance
        temporal_score = np.linspace(0.3, 1.0, num_bubbles)
        return temporal_score
    
    def _enforce_hard_limit(self, mu: np.ndarray, Sigma: np.ndarray, weight: np.ndarray, 
                           color: np.ndarray) -> Tuple:
        """Enforce hard bubble limit by keeping most important bubbles."""
        if len(mu) <= self.hard_limit:
            return mu, Sigma, weight, color
        
        # Calculate importance scores
        importance = self._calculate_importance_scores(mu, Sigma, weight)
        
        # Keep top N bubbles by importance
        top_indices = np.argsort(importance)[-self.hard_limit:]
        keep_mask = np.zeros(len(mu), dtype=bool)
        keep_mask[top_indices] = True
        
        return (mu[keep_mask], Sigma[keep_mask], weight[keep_mask], color[keep_mask])
    
    def get_global_map_bubbles(self) -> Tuple:
        """Get bubbles from persistent global map."""
        return self.global_map.get_high_quality_bubbles()
    
    def get_statistics(self) -> dict:
        """Get pruning statistics."""
        stats = self.pruning_stats.copy()
        stats['global_map_stats'] = self.global_map.get_statistics()
        return stats
    
    def update_global_map_ages(self):
        """Update ages in persistent global map."""
        self.global_map.update_ages()
