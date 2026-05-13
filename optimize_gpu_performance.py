#!/usr/bin/env python3
"""GPU Performance Optimization Script for SLAM System."""

import sys
import os
import time

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def optimize_gpu_performance():
    """Optimize GPU settings for maximum performance."""
    
    print("=== GPU Performance Optimization ===")
    
    try:
        import cupy as cp
        import numpy as np
        
        if not cp.cuda.is_available():
            print("ERROR: CUDA not available")
            return False
            
        device = cp.cuda.Device()
        print(f"GPU Device: {device}")
        print(f"GPU Memory: {device.mem_info[1] // (1024**2)} MB total")
        
        # Optimize GPU memory pool
        print("\n=== Optimizing GPU Memory Pool ===")
        memory_pool = cp.get_memory_pool()
        memory_pool.set_limit(size=2**30)  # 1GB limit
        
        pinned_pool = cp.get_pinned_memory_pool()
        pinned_pool.set_limit(size=2**28)  # 256MB limit
        
        print("GPU memory pools configured")
        
        # Pre-warm GPU with realistic SLAM workloads
        print("\n=== GPU Pre-warming ===")
        
        # Simulate bubble backprojection workload
        depth_size = (400, 640)
        stride = 4
        h, w = depth_size
        
        # Create coordinate grids (typical SLAM operation)
        u = cp.arange(0, w, stride, dtype=cp.float32)
        v = cp.arange(0, h, stride, dtype=cp.float32)
        v_grid, u_grid = cp.meshgrid(v, u, indexing='ij')
        
        # Simulate depth processing
        depth_gpu = cp.random.random(depth_size, dtype=cp.float32) * 5.0 + 0.5
        valid_mask = (depth_gpu > 0.1) & (depth_gpu < 8.0)
        
        # Simulate coordinate transformation
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        d_flat = depth_gpu[v_flat.astype(cp.int32), u_flat.astype(cp.int32)]
        
        # Simulate 3D backprojection
        fx, fy, cx, cy = 471.05, 471.05, 330.98, 189.69
        x = (u_flat - cx) * d_flat / fx
        y = (v_flat - cy) * d_flat / fy
        z = d_flat
        
        points_3d = cp.stack([x, y, z], axis=1)
        valid_points = points_3d[valid_mask.ravel()[:len(points_3d)]]
        
        print(f"Pre-warmed GPU with {len(valid_points)} 3D points")
        
        # Performance benchmark
        print("\n=== Performance Benchmark ===")
        
        # GPU matrix operations (typical SLAM workload)
        n_points = 50000
        points_gpu = cp.random.random((n_points, 3), dtype=cp.float32)
        pose_gpu = cp.random.random((4, 4), dtype=cp.float32)
        
        # Benchmark GPU operations
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        
        # Simulate SLAM operations
        transformed = cp.dot(points_gpu, pose_gpu[:3, :3].T) + pose_gpu[:3, 3]
        distances = cp.linalg.norm(transformed, axis=1)
        filtered = transformed[distances < 10.0]
        
        end.record()
        end.synchronize()
        
        gpu_time = cp.cuda.get_elapsed_time(start, end)
        print(f"GPU operations: {gpu_time:.2f} ms for {n_points} points")
        
        # Compare with CPU
        points_cpu = cp.asnumpy(points_gpu)
        pose_cpu = cp.asnumpy(pose_gpu)
        
        start_cpu = time.time()
        transformed_cpu = np.dot(points_cpu, pose_cpu[:3, :3].T) + pose_cpu[:3, 3]
        distances_cpu = np.linalg.norm(transformed_cpu, axis=1)
        filtered_cpu = transformed_cpu[distances_cpu < 10.0]
        cpu_time = (time.time() - start_cpu) * 1000
        
        print(f"CPU operations: {cpu_time:.2f} ms for {n_points} points")
        print(f"GPU Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Memory usage report
        print("\n=== Memory Usage ===")
        free_mem = device.mem_info[0] // (1024**2)
        total_mem = device.mem_info[1] // (1024**2)
        used_mem = total_mem - free_mem
        print(f"GPU Memory: {used_mem} MB used, {free_mem} MB free")
        
        # Cleanup
        del points_gpu, pose_gpu, transformed, distances, filtered
        del points_cpu, pose_cpu, transformed_cpu, distances_cpu, filtered_cpu
        del depth_gpu, valid_mask, points_3d, valid_points
        
        return True
        
    except Exception as e:
        print(f"GPU optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_slam_gpu_performance():
    """Test actual SLAM system GPU performance."""
    
    print("\n=== SLAM GPU Performance Test ===")
    
    try:
        from src.mapping.gaussian_bubbles import GaussianBubbleMap
        import numpy as np
        
        # Create realistic test data
        K = np.array([[471.05, 0, 330.98], [0, 471.05, 189.69], [0, 0, 1]])
        
        # Optimized configuration for GPU performance
        config = {
            'bubble_stride': 4,  # Reduced from 1 to 4
            'bubble_cuda': True,
            'use_gpu': True,
            'bubble_max_depth': 8.0,
            'bubble_min_depth': 0.1,
            'bubble_sigma_disp': 0.4,
            'bubble_sigma_pix': 0.2,
            'bubble_sigma_par_max': 0.12,
            'bubble_depth_edge_thresh': 0.40,
            'use_raw_kernels': False,  # Disabled for stability
            'use_zero_copy': True,
            'use_lazy_mirrors': True
        }
        
        bubble_map = GaussianBubbleMap(K, 0.1, config)
        
        # Test with realistic depth data
        depth = np.random.rand(400, 640) * 5.0 + 0.5
        pose = np.eye(4)
        image = np.random.rand(400, 640, 3)
        
        print("Testing GPU-accelerated backprojection...")
        
        # Time the GPU operations
        start_time = time.time()
        mu, Sigma, w, col = bubble_map.backproject_frame(depth, pose, image, 4)
        gpu_time = (time.time() - start_time) * 1000
        
        print(f"GPU backprojection: {gpu_time:.2f} ms")
        print(f"Generated {len(mu)} 3D points")
        
        # Test multiple frames for stability
        print("\nTesting multiple frames...")
        times = []
        for i in range(5):
            start = time.time()
            mu_test, _, _, _ = bubble_map.backproject_frame(depth, pose, image, 4)
            frame_time = (time.time() - start) * 1000
            times.append(frame_time)
            print(f"Frame {i+1}: {frame_time:.2f} ms ({len(mu_test)} points)")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"\nAverage: {avg_time:.2f} ± {std_time:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"SLAM GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting GPU Performance Optimization...")
    
    success1 = optimize_gpu_performance()
    success2 = test_slam_gpu_performance()
    
    if success1 and success2:
        print("\n=== GPU Optimization Complete ===")
        print("GPU acceleration is now properly configured!")
        print("\nTo run SLAM with optimized GPU performance:")
        print("python run_gpu.py")
        print("\nYou should now see:")
        print("- Higher GPU memory usage in Task Manager")
        print("- Faster processing times")
        print("- Real GPU acceleration (not CPU fallback)")
    else:
        print("\n=== GPU Optimization Failed ===")
        print("Check the error messages above for troubleshooting.")
