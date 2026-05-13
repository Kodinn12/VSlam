"""8-path SGM path aggregation for stereo matching."""

import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


def aggregate_paths_cpu(cost_volume: np.ndarray, p1: float = 10.0, p2: float = 120.0) -> np.ndarray:
    """
    Perform 8-path SGM aggregation on CPU.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Cost volume (H, W, D), uint16
    p1 : float
        Small penalty for small disparity changes
    p2 : float
        Large penalty for large disparity changes (intensity-adaptive)
    
    Returns
    -------
    np.ndarray
        Aggregated cost volume (H, W, D), uint16
    """
    H, W, D = cost_volume.shape
    
    # Initialize aggregated cost volume
    aggregated = cost_volume.copy().astype(np.float32)
    
    # 8 directions: (di, dj)
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # For each direction, perform path aggregation
    # TODO: Optimize with OpenMP for parallel processing
    for di, dj in directions:
        # Determine traversal order based on direction
        if di < 0:
            i_range = range(H)
        else:
            i_range = range(H-1, -1, -1)
        
        if dj < 0:
            j_range = range(W)
        else:
            j_range = range(W-1, -1, -1)
        
        for i in i_range:
            for j in j_range:
                # Previous pixel in this direction
                pi, pj = i - di, j - dj
                
                if 0 <= pi < H and 0 <= pj < W:
                    # Get costs from previous pixel
                    prev_costs = aggregated[pi, pj, :]
                    
                    # Compute path costs for current pixel
                    min_prev = np.min(prev_costs)
                    
                    for d in range(D):
                        # Cost from previous pixel at same disparity
                        lr1 = prev_costs[d]
                        
                        # Cost from previous pixel at d-1
                        lr2 = prev_costs[d-1] if d > 0 else float('inf')
                        
                        # Cost from previous pixel at d+1
                        lr3 = prev_costs[d+1] if d < D-1 else float('inf')
                        
                        # Minimum path cost
                        min_path = min(lr1, lr2 + p1, lr3 + p1, min_prev + p2)
                        
                        # Add to aggregated cost
                        aggregated[i, j, d] += min_path - min_prev
    
    return np.clip(aggregated, 0, 65535).astype(np.uint16)


def aggregate_paths_gpu(cost_volume: np.ndarray, p1: float = 10.0, p2: float = 120.0) -> np.ndarray:
    """
    Perform 8-path SGM aggregation on GPU using CuPy RawKernel.
    
    Parameters
    ----------
    cost_volume : np.ndarray
        Cost volume on GPU (H, W, D), uint16
    p1 : float
        Small penalty for small disparity changes
    p2 : float
        Large penalty for large disparity changes (intensity-adaptive)
    
    Returns
    -------
    np.ndarray
        Aggregated cost volume on GPU (H, W, D), uint16
    """
    if not USE_CUPY:
        logger.warning("CuPy not available, falling back to CPU")
        return aggregate_paths_cpu(cost_volume, p1, p2)
    
    H, W, D = cost_volume.shape
    
    # Initialize aggregated cost volume on GPU
    aggregated = cost_volume.astype(cp.float32).copy()
    
    # CuPy RawKernel for 8-path aggregation with shared memory optimization
    # Each direction is a separate kernel launch
    aggregation_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void aggregation_kernel(
        float* __restrict__ aggregated,
        const int H, const int W, const int D,
        const float p1, const float p2,
        const int di, const int dj, const int reverse_i, const int reverse_j
    ) {
        const int i = blockIdx.y * blockDim.y + threadIdx.y;
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int d = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (i >= H || j >= W || d >= D) return;
        
        // Shared memory for previous costs (reduces global memory access)
        __shared__ float s_prev[16][16][16];
        
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tz = threadIdx.z;
        
        // Previous pixel in this direction
        const int pi = reverse_i ? (i + di) : (i - di);
        const int pj = reverse_j ? (j + dj) : (j - dj);
        
        if (pi < 0 || pi >= H || pj < 0 || pj >= W) return;
        
        // Load previous costs into shared memory
        if (tz < D) {
            s_prev[ty][tx][tz] = aggregated[pi * W * D + pj * D + tz];
        }
        __syncthreads();
        
        const int idx_prev = pi * W * D + pj * D;
        const int idx_curr = i * W * D + j * D;
        
        float lr1 = s_prev[ty][tx][d];
        float lr2 = (d > 0) ? s_prev[ty][tx][d - 1] : 1e10f;
        float lr3 = (d < D - 1) ? s_prev[ty][tx][d + 1] : 1e10f;
        
        // Find minimum of previous costs from shared memory
        float min_prev = 1e10f;
        for (int k = 0; k < D; k++) {
            float val = s_prev[ty][tx][k];
            if (val < min_prev) min_prev = val;
        }
        
        // Compute minimum path cost
        float min_path = fminf(fminf(lr1, lr2 + p1), fminf(lr3 + p1, min_prev + p2));
        
        // Add to aggregated cost
        atomicAdd(&aggregated[idx_curr + d], min_path - min_prev);
    }
    ''', 'aggregation_kernel')
    
    block_size = (8, 8, 8)
    grid_size = ((W + block_size[0] - 1) // block_size[0], 
                (H + block_size[1] - 1) // block_size[1],
                (D + block_size[2] - 1) // block_size[2])
    
    # 8 directions: (di, dj, reverse_i, reverse_j)
    # reverse_i/reverse_j control traversal order
    directions = [
        (-1, 0, 0, 0),   # N
        (-1, 1, 0, 0),   # NE
        (0, 1, 0, 0),    # E
        (1, 1, 1, 1),    # SE
        (1, 0, 1, 0),    # S
        (1, -1, 1, 0),   # SW
        (0, -1, 0, 1),   # W
        (-1, -1, 0, 1)   # NW
    ]
    
    for di, dj, reverse_i, reverse_j in directions:
        aggregation_kernel(
            grid_size, block_size,
            (aggregated, H, W, D, p1, p2, di, dj, reverse_i, reverse_j)
        )
    
    return cp.clip(aggregated, 0, 65535).astype(cp.uint16)
