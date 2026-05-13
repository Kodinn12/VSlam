"""CUDA RawKernels for maximum GPU performance optimization."""

import numpy as np
import threading
from .logger import get_logger
from .cupy_utils import to_numpy_safe

logger = get_logger(__name__)

# CUDA availability check
try:
    import cupy as cp
    USE_CUPY = True
    logger.info("CuPy available for RawKernels")
except ImportError:
    cp = None
    USE_CUPY = False
    logger.info("CuPy not available, RawKernels disabled")

class CUDARawKernelManager:
    """Manages CUDA RawKernels for maximum GPU performance."""
    
    def __init__(self):
        self.kernels = {}
        self.stream = None
        self.zero_copy_enabled = False
        self.lazy_mirrors = {}
        self._setup_cuda_environment()
        
    def _setup_cuda_environment(self):
        """Setup CUDA environment with optimizations."""
        if USE_CUPY:
            try:
                # Create CUDA stream for async operations
                self.stream = cp.cuda.Stream()
                
                # Check for zero-copy support
                device = cp.cuda.Device()
                device_props = device.attributes
                self.zero_copy_enabled = device_props.get('ManagedMemory', False)
                
                if self.zero_copy_enabled:
                    logger.info("Zero-copy memory enabled")
                else:
                    logger.info("Zero-copy memory not available, using standard memory")
                
                # Compile RawKernels
                self._compile_kernels()
                
            except Exception as e:
                logger.warning(f"CUDA setup failed: {e}")
    
    def _compile_kernels(self):
        """Compile CUDA RawKernels for maximum performance."""
        if not USE_CUPY:
            return
        
        # Bubble backprojection kernel
        bubble_kernel_source = '''
        extern "C" __global__
        void backproject_bubbles(
            const float* depth,
            const float* pose,
            float* points,
            float* sigmas,
            int width, int height,
            float fx, float fy, float cx, float cy,
            float baseline, float sigma_disp, float sigma_pix,
            int stride, float motion_scale,
            int batch_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size) return;
            
            // Calculate pixel coordinates
            int u = (idx % (width / stride)) * stride;
            int v = (idx / (width / stride)) * stride;
            
            // Get depth value
            float d = depth[v * width + u];
            if (d <= 0.1f || d > 8.0f) return;
            
            // Backproject to camera coordinates
            float x = (u - cx) * d / fx;
            float y = (v - cy) * d / fy;
            float z = d;
            
            // Calculate sigmas
            float sigma_par = (d * d / (fx * baseline)) * sigma_disp * motion_scale;
            float sigma_per = (d / fx) * sigma_pix * motion_scale;
            sigma_par = min(sigma_par, 0.15f);
            
            // Transform to world coordinates
            float R[9] = {pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[8], pose[9], pose[10]};
            float t[3] = {pose[3], pose[7], pose[11]};
            
            points[idx * 3 + 0] = R[0] * x + R[1] * y + R[2] * z + t[0];
            points[idx * 3 + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
            points[idx * 3 + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
            
            sigmas[idx * 3 + 0] = sigma_per * sigma_per;
            sigmas[idx * 3 + 1] = sigma_per * sigma_per;
            sigmas[idx * 3 + 2] = sigma_par * sigma_par;
        }
        '''
        
        # Bubble fusion kernel
        fusion_kernel_source = '''
        extern "C" __global__
        void fuse_bubbles(
            const float* new_points,
            const float* new_sigmas,
            const float* new_weights,
            const float* new_colors,
            float* existing_points,
            float* existing_sigmas,
            float* existing_weights,
            float* existing_colors,
            int new_count, int existing_count,
            int max_bubbles
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= new_count) return;
            
            // Simple fusion: append new bubbles
            int target_idx = existing_count + idx;
            if (target_idx >= max_bubbles) return;
            
            // Copy new bubble data
            for (int i = 0; i < 3; i++) {
                existing_points[target_idx * 3 + i] = new_points[idx * 3 + i];
                existing_colors[target_idx * 3 + i] = new_colors[idx * 3 + i];
            }
            
            for (int i = 0; i < 9; i++) {
                existing_sigmas[target_idx * 9 + i] = new_sigmas[idx * 9 + i];
            }
            
            existing_weights[target_idx] = new_weights[idx];
        }
        '''
        
        # Bubble scoring kernel
        scoring_kernel_source = '''
        extern "C" __global__
        void score_bubbles(
            const float* weights,
            const float* sigmas,
            float* scores,
            int count
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= count) return;
            
            // Base score from weight
            float score = weights[idx];
            
            // Uncertainty penalty
            float det = sigmas[idx * 9 + 0] * (sigmas[idx * 9 + 4] * sigmas[idx * 9 + 8] - 
                        sigmas[idx * 9 + 5] * sigmas[idx * 9 + 7]) -
                        sigmas[idx * 9 + 1] * (sigmas[idx * 9 + 3] * sigmas[idx * 9 + 8] - 
                        sigmas[idx * 9 + 5] * sigmas[idx * 9 + 6]) +
                        sigmas[idx * 9 + 2] * (sigmas[idx * 9 + 3] * sigmas[idx * 9 + 7] - 
                        sigmas[idx * 9 + 4] * sigmas[idx * 9 + 6]);
            
            if (det > 1e-10f) {
                score += 0.1f * -logf(det);
            }
            
            scores[idx] = score;
        }
        '''
        
        # Matrix operations kernel
        matrix_kernel_source = '''
        extern "C" __global__
        void batch_matrix_multiply_3x3(
            const float* A, const float* B, float* C, int batch_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size) return;
            
            int baseA = idx * 9;
            int baseB = idx * 9;
            int baseC = idx * 9;
            
            // Matrix multiplication C = A * B
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        sum += A[baseA + i * 3 + k] * B[baseB + k * 3 + j];
                    }
                    C[baseC + i * 3 + j] = sum;
                }
            }
        }
        '''
        
        # Compile kernels
        try:
            self.kernels['backproject'] = cp.RawKernel(bubble_kernel_source, 'backproject_bubbles')
            self.kernels['fusion'] = cp.RawKernel(fusion_kernel_source, 'fuse_bubbles')
            self.kernels['scoring'] = cp.RawKernel(scoring_kernel_source, 'score_bubbles')
            self.kernels['matmul'] = cp.RawKernel(matrix_kernel_source, 'batch_matrix_multiply_3x3')
            logger.info("CUDA RawKernels compiled successfully")
        except Exception as e:
            logger.warning(f"RawKernel compilation failed: {e}")
    
    def get_zero_copy_array(self, shape, dtype=np.float32):
        """Create zero-copy memory array."""
        if not USE_CUPY or not self.zero_copy_enabled:
            return cp.zeros(shape, dtype=dtype)
        
        try:
            # Use unified memory for zero-copy
            return cp.zeros(shape, dtype=dtype, memptr=cp.cuda.alloc_unified(shape, dtype))
        except:
            return cp.zeros(shape, dtype=dtype)
    
    def create_lazy_mirror(self, cpu_array):
        """Create lazy mirror for CPU array."""
        if not USE_CUPY:
            return cpu_array
        
        array_id = id(cpu_array)
        if array_id in self.lazy_mirrors:
            return self.lazy_mirrors[array_id]
        
        # Create lazy mirror that only copies when accessed
        class LazyMirror:
            def __init__(self, cpu_arr, manager):
                self.cpu_arr = cpu_arr
                self.manager = manager
                self.gpu_arr = None
                self.dirty = True
            
            def get_gpu_array(self):
                if self.dirty or self.gpu_arr is None:
                    self.gpu_arr = cp.asarray(self.cpu_arr)
                    self.dirty = False
                return self.gpu_arr
            
            def mark_dirty(self):
                self.dirty = True
        
        mirror = LazyMirror(cpu_array, self)
        self.lazy_mirrors[array_id] = mirror
        return mirror
    
    def backproject_bubbles_raw(self, depth, pose, config):
        """Raw CUDA backprojection for maximum performance."""
        if not USE_CUPY or 'backproject' not in self.kernels:
            return None, None, None, None
        
        h, w = depth.shape
        stride = config.get('bubble_stride', 2)
        motion_scale = config.get('motion_scale', 1.0)
        
        # Calculate batch size
        batch_size = (h // stride) * (w // stride)
        
        # Allocate GPU memory
        points_gpu = cp.zeros(batch_size * 3, dtype=np.float32)
        sigmas_gpu = cp.zeros(batch_size * 3, dtype=np.float32)
        
        # Convert inputs to GPU
        depth_gpu = cp.asarray(depth.astype(np.float32))
        pose_gpu = cp.asarray(pose.astype(np.float32).flatten())
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
        
        with self.stream:
            self.kernels['backproject'](
                (blocks_per_grid,), (threads_per_block,),
                (depth_gpu, pose_gpu, points_gpu, sigmas_gpu,
                 w, h, config['fx'], config['fy'], config['cx'], config['cy'],
                 config['baseline'], config['bubble_sigma_disp'], config['bubble_sigma_pix'],
                 stride, motion_scale, batch_size)
            )
        
        # Synchronize and get results
        self.stream.synchronize()
        
        # Count valid points
        points_reshaped = points_gpu.reshape(-1, 3)
        valid_mask = cp.any(points_reshaped != 0, axis=1)
        valid_count = int(cp.sum(valid_mask))
        
        if valid_count == 0:
            return np.empty((0, 3)), np.empty((0, 3, 3)), np.empty(0), np.empty((0, 3))
        
        # Extract valid results
        valid_points = points_reshaped[to_numpy_safe(valid_mask)]
        valid_sigmas = sigmas_gpu.reshape(-1, 3, 3)[to_numpy_safe(valid_mask)]
        valid_weights = cp.ones(valid_count, dtype=np.float32)
        valid_colors = cp.full((valid_count, 3), 0.5, dtype=np.float32)
        
        return cp.asnumpy(valid_points), cp.asnumpy(valid_sigmas), cp.asnumpy(valid_weights), cp.asnumpy(valid_colors)
    
    def fuse_bubbles_raw(self, new_points, new_sigmas, new_weights, new_colors, existing_data):
        """Raw CUDA fusion for maximum performance."""
        if not USE_CUPY or 'fusion' not in self.kernels:
            return None
        
        new_count = len(new_points)
        existing_count = len(existing_data['points'])
        max_bubbles = existing_data['max_bubbles']
        
        if new_count == 0 or existing_count + new_count > max_bubbles:
            return None
        
        # Allocate GPU memory
        total_count = existing_count + new_count
        
        # Convert inputs to GPU
        new_points_gpu = cp.asarray(new_points.astype(np.float32).flatten())
        new_sigmas_gpu = cp.asarray(new_sigmas.astype(np.float32).flatten())
        new_weights_gpu = cp.asarray(new_weights.astype(np.float32))
        new_colors_gpu = cp.asarray(new_colors.astype(np.float32).flatten())
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (new_count + threads_per_block - 1) // threads_per_block
        
        with self.stream:
            self.kernels['fusion'](
                (blocks_per_grid,), (threads_per_block,),
                (new_points_gpu, new_sigmas_gpu, new_weights_gpu, new_colors_gpu,
                 existing_data['points_gpu'], existing_data['sigmas_gpu'],
                 existing_data['weights_gpu'], existing_data['colors_gpu'],
                 new_count, existing_count, max_bubbles)
            )
        
        self.stream.synchronize()
        return True
    
    def score_bubbles_raw(self, weights, sigmas):
        """Raw CUDA scoring for maximum performance."""
        if not USE_CUPY or 'scoring' not in self.kernels:
            return None
        
        count = len(weights)
        if count == 0:
            return np.array([])
        
        # GPU zone: allocate GPU memory using CuPy
        scores_gpu = cp.zeros(count, dtype=cp.float32)
        
        # GPU zone: convert inputs to GPU using CuPy
        weights_gpu = cp.asarray(weights.astype(np.float32))
        sigmas_gpu = cp.asarray(sigmas.astype(np.float32).flatten())
        
        # GPU zone: launch CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (count + threads_per_block - 1) // threads_per_block
        
        with self.stream:
            self.kernels['scoring'](
                (blocks_per_grid,), (threads_per_block,),
                (weights_gpu, sigmas_gpu, scores_gpu, count)
            )
        
        self.stream.synchronize()
        # GPU->CPU conversion at boundary
        return cp.asnumpy(scores_gpu)
    
    def batch_matrix_multiply_raw(self, A, B):
        """Raw CUDA batch matrix multiplication for maximum performance."""
        if not USE_CUPY or 'matmul' not in self.kernels:
            return None
        
        batch_size = A.shape[0]
        if batch_size == 0:
            return np.zeros((0, 3, 3))
        
        # GPU zone: allocate GPU memory using CuPy
        C_gpu = cp.zeros(batch_size * 9, dtype=np.float32)
        
        # GPU zone: convert inputs to GPU using CuPy
        A_gpu = cp.asarray(A.astype(np.float32).flatten())
        B_gpu = cp.asarray(B.astype(np.float32).flatten())
        
        # GPU zone: launch CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
        
        with self.stream:
            self.kernels['matmul'](
                (blocks_per_grid,), (threads_per_block,),
                (A_gpu, B_gpu, C_gpu, batch_size)
            )
        
        self.stream.synchronize()
        # GPU->CPU conversion at boundary
        return cp.asnumpy(C_gpu).reshape(batch_size, 3, 3)
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if USE_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
            self.stream.synchronize()
    
    def get_performance_stats(self):
        """Get performance statistics."""
        stats = {}
        if USE_CUPY:
            try:
                meminfo = cp.cuda.Device().mem_info
                stats['gpu_memory_free'] = meminfo[0] / (1024**3)
                stats['gpu_memory_total'] = meminfo[1] / (1024**3)
                stats['gpu_memory_used'] = stats['gpu_memory_total'] - stats['gpu_memory_free']
                stats['zero_copy_enabled'] = self.zero_copy_enabled
                stats['lazy_mirrors_count'] = len(self.lazy_mirrors)
                stats['compiled_kernels'] = list(self.kernels.keys())
            except:
                pass
        return stats

# Global CUDA RawKernel manager instance
cuda_kernel_manager = CUDARawKernelManager()
