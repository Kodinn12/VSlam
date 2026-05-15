import numpy as np
from ..utils.cupy_utils import cp, USE_CUPY, to_numpy_safe
from scipy.spatial import cKDTree
from ..utils.logger import get_logger

logger = get_logger(__name__)

# CUDA Kernels for SpatialHashGrid
SPATIAL_HASH_KERNELS = """
extern "C" {
    __device__ __forceinline__ int hash_pos(float3 pos, float cell_size, int table_size) {
        int ix = (int)floorf(pos.x / cell_size);
        int iy = (int)floorf(pos.y / cell_size);
        int iz = (int)floorf(pos.z / cell_size);
        
        // Murmur-inspired integer hash from architecture diagram
        unsigned int h = ((unsigned int)ix * 73856093U) ^ ((unsigned int)iy * 19349663U) ^ ((unsigned int)iz * 83492791U);
        h ^= h >> 16; h *= 0x85ebca6bU; h ^= h >> 13; h *= 0xc2b2ae35U;
        
        // table_size must be power-of-2 for & mask, or we use % for safety
        return h % table_size;
    }

    // V49: BUCKETED HASHING
    // Each hash slot now contains a fixed-size bucket of indices to reduce atomic contention.
    // BUCKET_SIZE=4 allows up to 4 bubbles per cell to be fused/searched without linked list traversal.
    #define BUCKET_SIZE 4

    __global__ void fuse_bubbles_kernel_bucketed(
        const float* new_pts,       // [N, 3]
        const float* new_sigmas,    // [N, 3, 3]
        const float* new_weights,   // [N]
        const float* new_colors,    // [N, 3]
        float* chunk_pts,           // [M, 3]
        float* chunk_sigmas,        // [M, 3, 3]
        float* chunk_weights,       // [M]
        float* chunk_colors,        // [M, 3]
        int* hash_table,            // [table_size * BUCKET_SIZE]
        float cell_size,
        int table_size,
        int num_new,
        int* current_chunk_size,    // scalar
        int max_chunk_size
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_new) return;

        float3 pos = make_float3(new_pts[idx * 3 + 0], new_pts[idx * 3 + 1], new_pts[idx * 3 + 2]);
        if (!isfinite(pos.x) || !isfinite(pos.y) || !isfinite(pos.z)) return;
        
        float new_w = new_weights[idx];
        if (!isfinite(new_w) || new_w <= 1e-6f) return;
        
        int slot = hash_pos(pos, cell_size, table_size);
        float new_tr = new_sigmas[idx*9+0] + new_sigmas[idx*9+4] + new_sigmas[idx*9+8];

        bool fused = false;
        // Search the bucket
        for (int i = 0; i < BUCKET_SIZE; i++) {
            int existing = hash_table[slot * BUCKET_SIZE + i];
            if (existing == -1) break;

            float3 p2 = make_float3(chunk_pts[existing * 3 + 0], chunk_pts[existing * 3 + 1], chunk_pts[existing * 3 + 2]);
            float dist2 = (pos.x-p2.x)*(pos.x-p2.x) + (pos.y-p2.y)*(pos.y-p2.y) + (pos.z-p2.z)*(pos.z-p2.z);
            
            if (dist2 < cell_size * cell_size) {
                float old_w = chunk_weights[existing];
                float fused_w = old_w + new_w;
                if (fused_w > 1e-6f) {
                    chunk_pts[existing*3+0] = (old_w * p2.x + new_w * pos.x) / fused_w;
                    chunk_pts[existing*3+1] = (old_w * p2.y + new_w * pos.y) / fused_w;
                    chunk_pts[existing*3+2] = (old_w * p2.z + new_w * pos.z) / fused_w;
                    
                    float old_tr = chunk_sigmas[existing*9+0] + chunk_sigmas[existing*9+4] + chunk_sigmas[existing*9+8];
                    if (new_tr < old_tr) {
                        for(int k=0; k<9; k++) chunk_sigmas[existing*9+k] = new_sigmas[idx*9+k];
                    }
                    chunk_weights[existing] = fused_w;
                    fused = true;
                }
                break;
            }
        }

        if (!fused) {
            if (*current_chunk_size < max_chunk_size) {
                int target = atomicAdd(current_chunk_size, 1);
                if (target < max_chunk_size) {
                    chunk_pts[target*3+0] = pos.x;
                    chunk_pts[target*3+1] = pos.y;
                    chunk_pts[target*3+2] = pos.z;
                    for(int k=0; k<9; k++) chunk_sigmas[target*9+k] = new_sigmas[idx*9+k];
                    chunk_weights[target] = new_w;
                    chunk_colors[target*3+0] = new_colors[idx*3+0];
                    chunk_colors[target*3+1] = new_colors[idx*3+1];
                    chunk_colors[target*3+2] = new_colors[idx*3+2];
                    
                    // Atomic insertion into the first available bucket slot
                    bool inserted = false;
                    for (int i = 0; i < BUCKET_SIZE; i++) {
                        int prev = atomicCAS(&hash_table[slot * BUCKET_SIZE + i], -1, target);
                        if (prev == -1) {
                            inserted = true;
                            break;
                        }
                    }
                    // If bucket full, we just don't index it (fallback to next time or different cell)
                }
            }
        }
    }

    __global__ void insert_points_kernel(
        const float* points,        // [N, 3]
        int* cell_heads,            // [table_size]
        int* point_next,            // [N]
        float cell_size,
        int table_size,
        int num_points
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;

        float3 pos = make_float3(points[idx * 3 + 0], points[idx * 3 + 1], points[idx * 3 + 2]);
        int cell = hash_pos(pos, cell_size, table_size);

        // Atomic insertion into linked list
        int old_head = atomicExch(&cell_heads[cell], idx);
        point_next[idx] = old_head;
    }

    __global__ void count_neighbors_kernel(
        const float* points,        // [N, 3]
        const int* cell_heads,      // [table_size]
        const int* point_next,      // [N]
        int* neighbor_counts,       // [N]
        float radius,
        float cell_size,
        int table_size,
        int num_points
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;

        float3 pos = make_float3(points[idx * 3 + 0], points[idx * 3 + 1], points[idx * 3 + 2]);
        float r2 = radius * radius;
        int count = 0;

        // Search 3x3x3 neighborhood
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float3 neighbor_pos = make_float3(
                        pos.x + dx * cell_size,
                        pos.y + dy * cell_size,
                        pos.z + dz * cell_size
                    );
                    int cell = hash_pos(neighbor_pos, cell_size, table_size);
                    
                    int curr = cell_heads[cell];
                    while (curr != -1) {
                        if (curr != idx) {
                            float3 p2 = make_float3(points[curr * 3 + 0], points[curr * 3 + 1], points[curr * 3 + 2]);
                            float dx_p = pos.x - p2.x;
                            float dy_p = pos.y - p2.y;
                            float dz_p = pos.z - p2.z;
                            if (dx_p*dx_p + dy_p*dy_p + dz_p*dz_p <= r2) {
                                count++;
                            }
                        }
                        curr = point_next[curr];
                    }
                }
            }
        }
        neighbor_counts[idx] = count;
    }
}
"""

class SpatialHashGrid:
    """GPU-accelerated Spatial Hash Grid for O(1) point insertion and radius search."""
    
    def __init__(self, cell_size=0.5, table_size=1048576, use_gpu=True):
        self.cell_size = float(cell_size)
        self.table_size = int(table_size)
        self.use_gpu = use_gpu and USE_CUPY
        self.bucket_size = 4  # Matches BUCKET_SIZE in CUDA
        
        if self.use_gpu:
            try:
                self.module = cp.RawModule(code=SPATIAL_HASH_KERNELS)
                self.insert_kernel = self.module.get_function("insert_points_kernel")
                self.query_kernel = self.module.get_function("count_neighbors_kernel")
                self.fuse_kernel = self.module.get_function("fuse_bubbles_kernel_bucketed")
            except Exception as e:
                logger.warning(f"Failed to compile SpatialHashGrid kernels: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        self.cell_heads = None
        self.point_next = None
        self.points_gpu = None
        self.kdtree = None

    def build(self, points):
        """Build the hash grid from a set of points."""
        num_points = len(points)
        if num_points == 0:
            return

        if self.use_gpu:
            self.points_gpu = cp.asarray(points, dtype=cp.float32)
            self.cell_heads = cp.full(self.table_size, -1, dtype=cp.int32)
            self.point_next = cp.full(num_points, -1, dtype=cp.int32)
            
            threads_per_block = 256
            blocks = (num_points + threads_per_block - 1) // threads_per_block
            
            self.insert_kernel(
                (blocks,), (threads_per_block,),
                (self.points_gpu, self.cell_heads, self.point_next,
                 self.cell_size, self.table_size, num_points)
            )
        else:
            self.kdtree = cKDTree(to_numpy_safe(points))

    def count_neighbors(self, radius):
        """Count neighbors within radius for each point in the built grid."""
        if self.points_gpu is None and self.kdtree is None:
            return np.array([], dtype=np.int32)
        
        num_points = len(self.points_gpu) if self.use_gpu else len(self.kdtree.data)
        if num_points == 0:
            return np.array([], dtype=np.int32)

        if self.use_gpu:
            neighbor_counts = cp.zeros(num_points, dtype=cp.int32)
            threads_per_block = 256
            blocks = (num_points + threads_per_block - 1) // threads_per_block
            
            self.query_kernel(
                (blocks,), (threads_per_block,),
                (self.points_gpu, self.cell_heads, self.point_next,
                 neighbor_counts, float(radius), self.cell_size, self.table_size, num_points)
            )
            return neighbor_counts
        else:
            # CPU fallback using cKDTree
            # kdtree.query_ball_point is O(log N + K) where K is number of neighbors
            # To get counts:
            indices = self.kdtree.query_ball_point(self.kdtree.data, radius)
            # Subtract 1 because query_ball_point includes the point itself
            return np.array([len(idx) - 1 for idx in indices], dtype=np.int32)

    def radius_outlier_removal(self, min_neighbors=3, radius=0.1):
        """Return a mask of points that are NOT outliers."""
        counts = self.count_neighbors(radius)
        if len(counts) == 0:
            return np.array([], dtype=bool)
        return counts >= min_neighbors
