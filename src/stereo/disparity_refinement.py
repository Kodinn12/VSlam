"""Disparity refinement: subpixel, LR-check, confidence, hole-fill."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


def winner_takes_all_cpu(cost_volume: np.ndarray) -> np.ndarray:
    """
    Compute disparity by winner-takes-all on CPU.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Aggregated cost volume (H, W, D), uint16
    
    Returns
    -------
    np.ndarray
        Disparity map (H, W), float32 (pixels)
    """
    disparity = np.argmin(cost_volume, axis=2).astype(np.float32)
    return disparity


def subpixel_refinement_cpu(cost_volume: np.ndarray, disparity: np.ndarray) -> np.ndarray:
    """
    Subpixel refinement using quadratic interpolation on CPU.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Aggregated cost volume (H, W, D), uint16
    disparity : np.ndarray
        Integer disparity map (H, W), float32
    
    Returns
    -------
    np.ndarray
        Subpixel refined disparity map (H, W), float32
    """
    H, W, D = cost_volume.shape
    disparity_sub = disparity.copy()
    
    for i in range(H):
        for j in range(W):
            d = int(disparity[i, j])
            if d > 0 and d < D - 1:
                # Quadratic fit: d_sub = d - (C(d-1) - C(d+1)) / (2*(C(d-1) - 2*C(d) + C(d+1)))
                c_minus = cost_volume[i, j, d-1]
                c_center = cost_volume[i, j, d]
                c_plus = cost_volume[i, j, d+1]
                
                denominator = 2.0 * (c_minus - 2.0 * c_center + c_plus)
                if abs(denominator) > 1e-6:
                    delta = (c_minus - c_plus) / denominator
                    disparity_sub[i, j] = d + delta
    
    return disparity_sub


def left_right_check_cpu(left_disp: np.ndarray, right_disp: np.ndarray, 
                         tolerance: int = 1) -> np.ndarray:
    """
    Left-right consistency check on CPU.
    
    Parameters
    ----------
    left_disp : np.ndarray
        Left disparity map (H, W), float32
    right_disp : np.ndarray
        Right disparity map (H, W), float32
    tolerance : int
        Disparity tolerance for LR check
    
    Returns
    -------
    np.ndarray
        Validity mask (H, W), bool (True = valid)
    """
    H, W = left_disp.shape
    validity = np.ones((H, W), dtype=bool)
    
    for i in range(H):
        for j in range(W):
            d_left = left_disp[i, j]
            j_right = int(j - d_left)
            
            if 0 <= j_right < W:
                d_right = right_disp[i, j_right]
                if abs(d_left - d_right) > tolerance:
                    validity[i, j] = False
            else:
                validity[i, j] = False
    
    return validity


def confidence_filter_cpu(cost_volume: np.ndarray, disparity: np.ndarray,
                          peak_ratio_threshold: float = 0.7) -> np.ndarray:
    """
    Confidence filtering based on peak ratio on CPU.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Aggregated cost volume (H, W, D), uint16
    disparity : np.ndarray
        Disparity map (H, W), float32
    peak_ratio_threshold : float
        Minimum peak ratio for valid pixels
    
    Returns
    -------
    np.ndarray
        Confidence mask (H, W), bool
    """
    H, W = disparity.shape
    confidence = np.ones((H, W), dtype=bool)
    
    for i in range(H):
        for j in range(W):
            d = int(disparity[i, j])
            min_cost = cost_volume[i, j, d]
            second_min = np.min(np.delete(cost_volume[i, j, :], d))
            
            if min_cost > 0:
                peak_ratio = 1.0 - min_cost / (second_min + 1e-6)
                if peak_ratio < peak_ratio_threshold:
                    confidence[i, j] = False
    
    return confidence


def hole_filling_cpu(disparity: np.ndarray, validity_mask: np.ndarray) -> np.ndarray:
    """
    Hole filling on CPU (horizontal scan + vertical propagation).
    
    Parameters
    ----------
    disparity : np.ndarray
        Disparity map (H, W), float32
    validity_mask : np.ndarray
        Validity mask (H, W), bool
    
    Returns
    -------
    np.ndarray
        Filled disparity map (H, W), float32
    """
    H, W = disparity.shape
    filled = disparity.copy()
    
    # Horizontal scan fill
    for i in range(H):
        last_valid = 0.0
        for j in range(W):
            if validity_mask[i, j]:
                last_valid = disparity[i, j]
            else:
                filled[i, j] = last_valid
    
    # Vertical propagation
    for j in range(W):
        last_valid = 0.0
        for i in range(H):
            if validity_mask[i, j]:
                last_valid = filled[i, j]
            else:
                filled[i, j] = last_valid
    
    return filled


def refine_disparity_cpu(cost_volume: np.ndarray, lr_check: bool = True,
                         lr_tolerance: int = 1, confidence_threshold: float = 0.7) -> np.ndarray:
    """
    Full disparity refinement pipeline on CPU.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Aggregated cost volume (H, W, D), uint16
    lr_check : bool
        Enable left-right consistency check
    lr_tolerance : int
        LR check tolerance
    confidence_threshold : float
        Confidence threshold for filtering
    
    Returns
    -------
    np.ndarray
        Refined disparity map (H, W), float32
    """
    # 1. Winner-takes-all
    disparity = winner_takes_all_cpu(cost_volume)
    
    # 2. Subpixel refinement
    disparity = subpixel_refinement_cpu(cost_volume, disparity)
    
    # 3. Left-right check
    if lr_check:
        right_disp = winner_takes_all_cpu(cost_volume)
        validity = left_right_check_cpu(disparity, right_disp, lr_tolerance)
    else:
        validity = np.ones_like(disparity, dtype=bool)
    
    # 4. Confidence filtering
    confidence = confidence_filter_cpu(cost_volume, disparity, confidence_threshold)
    
    # Combined validity
    combined_validity = validity & confidence
    
    # 5. Hole filling
    disparity = hole_filling_cpu(disparity, combined_validity)
    
    return disparity
