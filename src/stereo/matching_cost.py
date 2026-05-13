"""Matching cost computation for stereo (Hamming + gradient-adaptive)."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


def hamming_cost_cpu(left_census: np.ndarray, right_census: np.ndarray, 
                     max_disparity: int) -> np.ndarray:
    """
    Compute Hamming distance cost volume on CPU.
    
    Parameters
    ----------
    left_census : np.ndarray
        Left census transform (H, W), uint64
    right_census : np.ndarray
        Right census transform (H, W), uint64
    max_disparity : int
        Maximum disparity to search
    
    Returns
    -------
    np.ndarray
        Cost volume (H, W, max_disparity), uint16
    """
    H, W = left_census.shape
    cost_volume = np.zeros((H, W, max_disparity), dtype=np.uint16)
    
    # For each pixel and each disparity, compute Hamming distance
    # TODO: Optimize with AVX2 popcount for faster bit counting
    for i in range(H):
        for j in range(W):
            for d in range(max_disparity):
                j_right = j - d
                if j_right < 0:
                    cost_volume[i, j, d] = 65535  # Max cost for out-of-bounds
                else:
                    # XOR and count bits (Hamming distance)
                    xor = left_census[i, j] ^ right_census[i, j_right]
                    cost_volume[i, j, d] = bin(xor).count('1')
    
    return cost_volume


def hamming_cost_gpu(left_census: np.ndarray, right_census: np.ndarray,
                     max_disparity: int) -> np.ndarray:
    """
    Compute Hamming distance cost volume on GPU.
    
    Parameters
    ----------
    left_census : np.ndarray
        Left census transform on GPU (H, W), uint64
    right_census : np.ndarray
        Right census transform on GPU (H, W), uint64
    max_disparity : int
        Maximum disparity to search
    
    Returns
    -------
    np.ndarray
        Cost volume on GPU (H, W, max_disparity), uint16
    """
    if not USE_CUPY:
        logger.warning("CuPy not available, falling back to CPU")
        return hamming_cost_cpu(left_census, right_census, max_disparity)
    
    H, W = left_census.shape
    
    # Initialize cost volume on GPU
    cost_volume = cp.zeros((H, W, max_disparity), dtype=cp.uint16)
    
    # CuPy RawKernel for Hamming cost computation with shared memory
    hamming_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void hamming_cost_kernel(
        const unsigned long long* __restrict__ left_census,
        const unsigned long long* __restrict__ right_census,
        unsigned short* __restrict__ cost_volume,
        const int H, const int W, const int max_disparity
    ) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int d = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (i >= H || j >= W || d >= max_disparity) return;
        
        const int j_right = j - d;
        
        // Shared memory for census values (coalesced loads)
        __shared__ unsigned long long s_left[16][16];
        __shared__ unsigned long long s_right[16][16];
        
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        
        // Load census values into shared memory
        s_left[ty][tx] = left_census[i * W + j];
        if (j_right >= 0 && j_right < W) {
            s_right[ty][tx] = right_census[i * W + j_right];
        } else {
            s_right[ty][tx] = 0;
        }
        __syncthreads();
        
        if (j_right < 0) {
            cost_volume[i * W * max_disparity + j * max_disparity + d] = 65535;
        } else {
            const unsigned long long left_val = s_left[ty][tx];
            const unsigned long long right_val = s_right[ty][tx];
            
            // XOR and popcount (Hamming distance)
            const unsigned long long xor_val = left_val ^ right_val;
            
            // Built-in popcount for 64-bit
            unsigned int hamming = __popcll(xor_val);
            
            cost_volume[i * W * max_disparity + j * max_disparity + d] = hamming;
        }
    }
    ''', 'hamming_cost_kernel')
    
    # Launch kernel
    block_size = (8, 8, 8)
    grid_size = ((W + block_size[0] - 1) // block_size[0], 
                (H + block_size[1] - 1) // block_size[1],
                (max_disparity + block_size[2] - 1) // block_size[2])
    
    hamming_kernel(
        grid_size, block_size,
        (left_census, right_census, cost_volume, H, W, max_disparity)
    )
    
    return cost_volume
