"""Cost volume construction and refinement for stereo SGM."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


def build_cost_volume_cpu(hamming_cost: np.ndarray, gradient_cost: np.ndarray = None,
                          gradient_weight: float = 0.1) -> np.ndarray:
    """
    Build combined cost volume from Hamming and gradient costs on CPU.
    
    Parameters
    ----------
    hamming_cost : np.ndarray
        Hamming cost volume (H, W, D), uint16
    gradient_cost : np.ndarray
        Gradient cost volume (H, W, D), optional
    gradient_weight : float
        Weight for gradient cost in combination
    
    Returns
    -------
    np.ndarray
        Combined cost volume (H, W, D), uint16
    """
    if gradient_cost is None:
        return hamming_cost.astype(np.uint16)
    
    # Combine Hamming and gradient costs
    combined = hamming_cost.astype(np.float32) + gradient_weight * gradient_cost
    return np.clip(combined, 0, 65535).astype(np.uint16)


def refine_cost_volume_cpu(cost_volume: np.ndarray) -> np.ndarray:
    """
    Refine cost volume on CPU (intensity-adaptive penalties, etc.).
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Cost volume (H, W, D), uint16
    
    Returns
    -------
    np.ndarray
        Refined cost volume (H, W, D), uint16
    """
    # For now, just return the cost volume as-is
    # TODO: Add intensity-adaptive penalty refinement
    return cost_volume


def build_cost_volume_gpu(hamming_cost: np.ndarray, gradient_cost: np.ndarray = None,
                          gradient_weight: float = 0.1) -> np.ndarray:
    """
    Build combined cost volume from Hamming and gradient costs on GPU.
    
    Parameters
    ----------
    hamming_cost : np.ndarray
        Hamming cost volume on GPU (H, W, D), uint16
    gradient_cost : np.ndarray
        Gradient cost volume on GPU (H, W, D), optional
    gradient_weight : float
        Weight for gradient cost in combination
    
    Returns
    -------
    np.ndarray
        Combined cost volume on GPU (H, W, D), uint16
    """
    if not USE_CUPY:
        logger.warning("CuPy not available, falling back to CPU")
        return build_cost_volume_cpu(hamming_cost, gradient_cost, gradient_weight)
    
    if gradient_cost is None:
        return hamming_cost.astype(cp.uint16)
    
    # Combine Hamming and gradient costs on GPU
    combined = hamming_cost.astype(cp.float32) + gradient_weight * gradient_cost
    return cp.clip(combined, 0, 65535).astype(cp.uint16)


def refine_cost_volume_gpu(cost_volume: np.ndarray) -> np.ndarray:
    """
    Refine cost volume on GPU.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Cost volume on GPU (H, W, D), uint16
    
    Returns
    -------
    np.ndarray
        Refined cost volume on GPU (H, W, D), uint16
    """
    if not USE_CUPY:
        logger.warning("CuPy not available, falling back to CPU")
        return refine_cost_volume_cpu(cost_volume)
    
    # For now, just return the cost volume as-is
    # TODO: Add intensity-adaptive penalty refinement with GPU kernel
    return cost_volume
