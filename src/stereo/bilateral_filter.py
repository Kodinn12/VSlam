"""Edge-preserving bilateral filter for disparity refinement."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


def bilateral_filter_cpu(disparity: np.ndarray, image: np.ndarray = None,
                        spatial_sigma: float = 2.0, range_sigma: float = 10.0) -> np.ndarray:
    """
    Edge-preserving bilateral filter on CPU.
    
    Parameters
    ----------
    disparity : np.ndarray
        Disparity map (H, W), float32
    image : np.ndarray
        Grayscale image for edge guidance (H, W), uint8, optional
    spatial_sigma : float
        Spatial Gaussian sigma
    range_sigma : float
        Range Gaussian sigma
    
    Returns
    -------
    np.ndarray
        Filtered disparity map (H, W), float32
    """
    H, W = disparity.shape
    filtered = disparity.copy()
    
    # If no image provided, use disparity for range filter
    if image is None:
        image = disparity.astype(np.uint8)
    
    # Gaussian spatial weights (5x5 window)
    window_size = 5
    half = window_size // 2
    
    # Precompute spatial Gaussian weights
    spatial_weights = np.zeros((window_size, window_size), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            spatial_weights[i, j] = np.exp(-((i - half)**2 + (j - half)**2) / (2 * spatial_sigma**2))
    
    # Apply bilateral filter
    for i in range(half, H - half):
        for j in range(half, W - half):
            center_disp = disparity[i, j]
            center_img = image[i, j]
            
            sum_weights = 0.0
            sum_filtered = 0.0
            
            for wi in range(-half, half + 1):
                for wj in range(-half, half + 1):
                    ni, nj = i + wi, j + wj
                    
                    # Spatial weight
                    w_spatial = spatial_weights[wi + half, wj + half]
                    
                    # Range weight based on image intensity
                    range_diff = abs(int(image[ni, nj]) - int(center_img))
                    w_range = np.exp(-(range_diff**2) / (2 * range_sigma**2))
                    
                    # Combined weight
                    w = w_spatial * w_range
                    
                    sum_weights += w
                    sum_filtered += w * disparity[ni, nj]
            
            if sum_weights > 1e-6:
                filtered[i, j] = sum_filtered / sum_weights
    
    return filtered


def bilateral_filter_gpu(disparity: np.ndarray, image: np.ndarray = None,
                        spatial_sigma: float = 2.0, range_sigma: float = 10.0) -> np.ndarray:
    """
    Edge-preserving bilateral filter on GPU.
    
    Parameters
    ----------
    disparity : np.ndarray
        Disparity map on GPU (H, W), float32
    image : np.ndarray
        Grayscale image on GPU for edge guidance (H, W), uint8, optional
    spatial_sigma : float
        Spatial Gaussian sigma
    range_sigma : float
        Range Gaussian sigma
    
    Returns
    -------
    np.ndarray
        Filtered disparity map on GPU (H, W), float32
    """
    if not USE_CUPY:
        logger.warning("CuPy not available, falling back to CPU")
        return bilateral_filter_cpu(disparity, image, spatial_sigma, range_sigma)
    
    H, W = disparity.shape
    window_size = 5
    half = window_size // 2
    
    # Convert to GPU arrays
    disparity_gpu = cp.asarray(disparity, dtype=cp.float32)
    
    # If no image provided, use disparity for range filter
    if image is None:
        image_gpu = disparity_gpu.astype(cp.uint8)
    else:
        image_gpu = cp.asarray(image, dtype=cp.uint8)
    
    # Initialize filtered array
    filtered = cp.zeros((H, W), dtype=cp.float32)
    
    # CuPy RawKernel for bilateral filter with shared memory optimization
    bilateral_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void bilateral_filter_kernel(
        const float* __restrict__ disparity,
        const unsigned char* __restrict__ image,
        float* __restrict__ filtered,
        const int H, const int W,
        const float spatial_sigma, const float range_sigma,
        const int window_size, const int half
    ) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < half || i >= H - half || j < half || j >= W - half) {
            if (i < H && j < W) filtered[i * W + j] = disparity[i * W + j];
            return;
        }
        
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        
        // Shared memory for window data (coalesced loads)
        __shared__ float s_disp[16][16];
        __shared__ unsigned char s_img[16][16];
        
        // Load window into shared memory
        for (int wi = -half; wi <= half; wi++) {
            for (int wj = -half; wj <= half; wj++) {
                const int si = ty + wi + half;
                const int sj = tx + wj + half;
                const int ni = i + wi;
                const int nj = j + wj;
                if (si < 16 && sj < 16 && ni >= 0 && ni < H && nj >= 0 && nj < W) {
                    s_disp[si][sj] = disparity[ni * W + nj];
                    s_img[si][sj] = image[ni * W + nj];
                }
            }
        }
        __syncthreads();
        
        const float center_disp = s_disp[ty + half][tx + half];
        const unsigned char center_img = s_img[ty + half][tx + half];
        
        float sum_weights = 0.0f;
        float sum_filtered = 0.0f;
        
        for (int wi = -half; wi <= half; wi++) {
            for (int wj = -half; wj <= half; wj++) {
                const int si = ty + wi + half;
                const int sj = tx + wj + half;
                
                // Spatial weight (precomputed Gaussian)
                const float dx = wi;
                const float dy = wj;
                const float w_spatial = expf(-(dx*dx + dy*dy) / (2.0f * spatial_sigma * spatial_sigma));
                
                // Range weight based on image intensity
                const int range_diff = abs((int)s_img[si][sj] - (int)center_img);
                const float w_range = expf(-(range_diff * range_diff) / (2.0f * range_sigma * range_sigma));
                
                // Combined weight
                const float w = w_spatial * w_range;
                
                sum_weights += w;
                sum_filtered += w * s_disp[si][sj];
            }
        }
        
        if (sum_weights > 1e-6f) {
            filtered[i * W + j] = sum_filtered / sum_weights;
        } else {
            filtered[i * W + j] = center_disp;
        }
    }
    ''', 'bilateral_filter_kernel')
    
    block_size = (16, 16)
    grid_size = ((W + block_size[0] - 1) // block_size[0], 
                (H + block_size[1] - 1) // block_size[1])
    
    bilateral_kernel(
        grid_size, block_size,
        (disparity_gpu, image_gpu, filtered, H, W, 
         spatial_sigma, range_sigma, window_size, half)
    )
    
    return filtered
