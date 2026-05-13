"""Gradient-adaptive penalties for SGM cost volume refinement."""

import numpy as np
import cv2
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


def compute_gradient_cost_cpu(left: np.ndarray, right: np.ndarray, 
                            max_disparity: int) -> np.ndarray:
    """
    Compute gradient cost volume on CPU.
    
    Parameters
    ----------
    left : np.ndarray
        Left grayscale image (H, W), uint8
    right : np.ndarray
        Right grayscale image (H, W), uint8
    max_disparity : int
        Maximum disparity to search
    
    Returns
    -------
    np.ndarray
        Gradient cost volume (H, W, max_disparity), uint16
    """
    H, W = left.shape
    
    # Compute image gradients (Sobel)
    grad_x_left = cv2.Sobel(left, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_left = cv2.Sobel(left, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude_left = np.sqrt(grad_x_left**2 + grad_y_left**2)
    
    grad_x_right = cv2.Sobel(right, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_right = cv2.Sobel(right, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude_right = np.sqrt(grad_x_right**2 + grad_y_right**2)
    
    # Initialize gradient cost volume
    gradient_cost = np.zeros((H, W, max_disparity), dtype=np.uint16)
    
    # For each pixel and each disparity, compute gradient cost
    for i in range(H):
        for j in range(W):
            for d in range(max_disparity):
                j_right = j - d
                if j_right < 0:
                    gradient_cost[i, j, d] = 65535  # Max cost for out-of-bounds
                else:
                    # Absolute difference of gradient magnitudes
                    grad_diff = abs(grad_magnitude_left[i, j] - grad_magnitude_right[i, j_right])
                    gradient_cost[i, j, d] = np.clip(grad_diff * 10, 0, 65535).astype(np.uint16)
    
    return gradient_cost


def compute_adaptive_penalties_cpu(image: np.ndarray, p1_base: float = 10.0, 
                                    p2_base: float = 120.0) -> tuple:
    """
    Compute intensity-adaptive penalties P1 and P2 on CPU.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), uint8
    p1_base : float
        Base small penalty
    p2_base : float
        Base large penalty
    
    Returns
    -------
    tuple
        (p1_map, p2_map) as float32 arrays (H, W)
    """
    H, W = image.shape
    
    # Compute image gradient magnitude
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude to [0, 1]
    grad_norm = grad_magnitude / (np.max(grad_magnitude) + 1e-6)
    
    # Adaptive penalties: higher gradient = higher penalty
    p1_map = p1_base * (1.0 + grad_norm)
    p2_map = p2_base * (1.0 + 2.0 * grad_norm)
    
    return p1_map.astype(np.float32), p2_map.astype(np.float32)


def compute_gradient_cost_gpu(left: np.ndarray, right: np.ndarray,
                            max_disparity: int) -> np.ndarray:
    """
    Compute gradient cost volume on GPU.
    
    Parameters
    ----------
    left : np.ndarray
        Left grayscale image on GPU (H, W), uint8
    right : np.ndarray
        Right grayscale image on GPU (H, W), uint8
    max_disparity : int
        Maximum disparity to search
    
    Returns
    -------
    np.ndarray
        Gradient cost volume on GPU (H, W, max_disparity), uint16
    """
    if not USE_CUPY:
        logger.warning("CuPy not available, falling back to CPU")
        return compute_gradient_cost_cpu(left, right, max_disparity)
    
    # Convert to GPU arrays
    left_gpu = cp.asarray(left, dtype=cp.uint8)
    right_gpu = cp.asarray(right, dtype=cp.uint8)
    
    H, W = left_gpu.shape
    
    # Compute gradients on GPU using CuPy
    # Simple gradient computation
    grad_x_left = cp.zeros_like(left_gpu, dtype=cp.float32)
    grad_y_left = cp.zeros_like(left_gpu, dtype=cp.float32)
    grad_x_left[:, 1:-1] = left_gpu[:, 2:] - left_gpu[:, :-2]
    grad_y_left[1:-1, :] = left_gpu[2:, :] - left_gpu[:-2, :]
    grad_magnitude_left = cp.sqrt(grad_x_left**2 + grad_y_left**2)
    
    grad_x_right = cp.zeros_like(right_gpu, dtype=cp.float32)
    grad_y_right = cp.zeros_like(right_gpu, dtype=cp.float32)
    grad_x_right[:, 1:-1] = right_gpu[:, 2:] - right_gpu[:, :-2]
    grad_y_right[1:-1, :] = right_gpu[2:, :] - right_gpu[:-2, :]
    grad_magnitude_right = cp.sqrt(grad_x_right**2 + grad_y_right**2)
    
    # Initialize gradient cost volume
    gradient_cost = cp.zeros((H, W, max_disparity), dtype=cp.uint16)
    
    # CuPy RawKernel for gradient cost computation
    gradient_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void gradient_cost_kernel(
        const float* __restrict__ grad_left,
        const float* __restrict__ grad_right,
        unsigned short* __restrict__ gradient_cost,
        const int H, const int W, const int max_disparity
    ) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int d = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (i >= H || j >= W || d >= max_disparity) return;
        
        const int j_right = j - d;
        
        if (j_right < 0) {
            gradient_cost[i * W * max_disparity + j * max_disparity + d] = 65535;
        } else {
            const float grad_diff = fabsf(grad_left[i * W + j] - grad_right[i * W + j_right]);
            gradient_cost[i * W * max_disparity + j * max_disparity + d] = (unsigned short)fminf(grad_diff * 10.0f, 65535.0f);
        }
    }
    ''', 'gradient_cost_kernel')
    
    block_size = (8, 8, 8)
    grid_size = ((W + block_size[0] - 1) // block_size[0],
                (H + block_size[1] - 1) // block_size[1],
                (max_disparity + block_size[2] - 1) // block_size[2])
    
    gradient_kernel(
        grid_size, block_size,
        (grad_magnitude_left, grad_magnitude_right, gradient_cost, H, W, max_disparity)
    )
    
    return gradient_cost
