"""Census transform for stereo matching (CPU + GPU)."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


def census_transform_cpu(left: np.ndarray, right: np.ndarray, window_size: int = 5) -> tuple:
    """
    Compute census transform on CPU using vectorized operations.
    
    Parameters
    ----------
    left : np.ndarray
        Left grayscale image (H, W), uint8
    right : np.ndarray
        Right grayscale image (H, W), uint8
    window_size : int
        Census window size (odd number), default 5
    
    Returns
    -------
    tuple
        (left_census, right_census) as uint64 arrays (H, W)
    """
    H, W = left.shape
    half = window_size // 2
    pad = half
    
    # Pad images
    left_padded = np.pad(left, pad, mode='reflect')
    right_padded = np.pad(right, pad, mode='reflect')
    
    # Initialize census arrays
    left_census = np.zeros((H, W), dtype=np.uint64)
    right_census = np.zeros((H, W), dtype=np.uint64)
    
    # For each pixel, compare with neighbors in census window
    # NOTE: For C/C++ port, add OpenMP parallelization:
    # #pragma omp parallel for collapse(2) schedule(dynamic)
    for i in range(H):
        for j in range(W):
            # Extract window
            window_left = left_padded[i:i+window_size, j:j+window_size]
            window_right = right_padded[i:i+window_size, j:j+window_size]
            
            # Center pixel
            center_left = window_left[half, half]
            center_right = window_right[half, half]
            
            # Compute census bitstring
            census_left = 0
            census_right = 0
            bit_idx = 0
            
            for wi in range(window_size):
                for wj in range(window_size):
                    if wi == half and wj == half:
                        continue
                    # Compare with center
                    if window_left[wi, wj] >= center_left:
                        census_left |= (1 << bit_idx)
                    if window_right[wi, wj] >= center_right:
                        census_right |= (1 << bit_idx)
                    bit_idx += 1
            
            left_census[i, j] = census_left
            right_census[i, j] = census_right
    
    return left_census, right_census


def census_transform_gpu(left: np.ndarray, right: np.ndarray, window_size: int = 5) -> tuple:
    """
    Compute census transform on GPU using CuPy RawKernel.
    
    Parameters
    ----------
    left : np.ndarray
        Left grayscale image (H, W), uint8
    right : np.ndarray
        Right grayscale image (H, W), uint8
    window_size : int
        Census window size (odd number), default 5
    
    Returns
    -------
    tuple
        (left_census, right_census) as uint64 arrays on GPU (H, W)
    """
    if not USE_CUPY:
        logger.warning("CuPy not available, falling back to CPU")
        return census_transform_cpu(left, right, window_size)
    
    H, W = left.shape
    half = window_size // 2
    
    # Convert to GPU arrays
    left_gpu = cp.asarray(left, dtype=cp.uint8)
    right_gpu = cp.asarray(right, dtype=cp.uint8)
    
    # Pad images on GPU
    left_padded = cp.pad(left_gpu, half, mode='reflect')
    right_padded = cp.pad(right_gpu, half, mode='reflect')
    
    # Initialize census arrays on GPU
    left_census = cp.zeros((H, W), dtype=cp.uint64)
    right_census = cp.zeros((H, W), dtype=cp.uint64)
    
    # CuPy RawKernel for census transform with shared memory optimization
    census_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void census_transform_kernel(
        const unsigned char* __restrict__ left_padded,
        const unsigned char* __restrict__ right_padded,
        unsigned long long* __restrict__ left_census,
        unsigned long long* __restrict__ right_census,
        const int H, const int W, const int window_size, const int half
    ) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i >= H || j >= W) return;
        
        const int pi = i + half;
        const int pj = j + half;
        const int padded_W = W + 2 * half;
        
        // Shared memory for window data (coalesced loads)
        __shared__ unsigned char s_left[16][16];
        __shared__ unsigned char s_right[16][16];
        
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        
        // Load window into shared memory
        if (ty < window_size && tx < window_size) {
            const int nwi = pi + ty - half;
            const int nwj = pj + tx - half;
            s_left[ty][tx] = left_padded[nwi * padded_W + nwj];
            s_right[ty][tx] = right_padded[nwi * padded_W + nwj];
        }
        __syncthreads();
        
        const unsigned char center_left = s_left[half][half];
        const unsigned char center_right = s_right[half][half];
        
        unsigned long long census_left = 0;
        unsigned long long census_right = 0;
        int bit_idx = 0;
        
        for (int wi = 0; wi < window_size; wi++) {
            for (int wj = 0; wj < window_size; wj++) {
                if (wi == half && wj == half) continue;
                
                const unsigned char val_left = s_left[wi][wj];
                const unsigned char val_right = s_right[wi][wj];
                
                if (val_left >= center_left) {
                    census_left |= (1ULL << bit_idx);
                }
                if (val_right >= center_right) {
                    census_right |= (1ULL << bit_idx);
                }
                bit_idx++;
            }
        }
        
        left_census[i * W + j] = census_left;
        right_census[i * W + j] = census_right;
    }
    ''', 'census_transform_kernel')
    
    # Launch kernel with shared memory optimization
    block_size = (16, 16)
    grid_size = ((W + block_size[0] - 1) // block_size[0], 
                (H + block_size[1] - 1) // block_size[1])
    
    census_kernel(
        grid_size, block_size,
        (left_padded, right_padded, left_census, right_census, 
         H, W, window_size, half)
    )
    
    return left_census, right_census
