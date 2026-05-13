# Stereo Pipeline Documentation

Custom 8-path Semi-Global Matching (SGM) stereo depth estimation pipeline for OAK-D SLAM system.

## Overview

This package provides a high-performance stereo matching pipeline with both CPU and GPU implementations. It supports dual depth source selection between OAK-D hardware depth and custom SGM processing.

## Architecture

```
stereo/
├── __init__.py              # Package exports
├── stereo_engine.py         # Unified CPU/GPU interface
├── census_transform.py      # Census transform (CPU + GPU)
├── matching_cost.py         # Hamming cost computation (CPU + GPU)
├── cost_volume.py           # Cost volume construction/refinement (CPU + GPU)
├── path_aggregation.py      # 8-path SGM aggregation (CPU + GPU)
├── disparity_refinement.py  # Disparity refinement pipeline (CPU)
├── bilateral_filter.py      # Edge-preserving bilateral filter (CPU + GPU)
├── temporal_smoother.py     # Temporal depth smoothing (CPU + GPU)
├── gradient_adaptive.py     # Gradient-adaptive penalties (CPU + GPU)
└── multiscale_sgm.py        # Multi-scale pyramid SGM (CPU + GPU)
```

## Pipeline Stages

### 1. Census Transform
- **Purpose**: Robust descriptor for pixel matching
- **Method**: Compare center pixel with neighbors in 5x5 window
- **Output**: 64-bit census descriptors per pixel
- **GPU**: Shared memory optimization (16x16 blocks)

### 2. Hamming Cost Computation
- **Purpose**: Compute matching cost between census descriptors
- **Method**: XOR + popcount (bit count)
- **Output**: Cost volume (H, W, max_disparity)
- **GPU**: Shared memory for census values

### 3. Cost Volume Construction
- **Purpose**: Combine Hamming and gradient costs
- **Method**: Weighted sum of cost sources
- **Output**: Combined cost volume
- **GPU**: Element-wise operations on GPU

### 4. 8-Path SGM Aggregation
- **Purpose**: Accumulate costs along 8 directional paths
- **Method**: Dynamic programming with adaptive penalties
- **Output**: Aggregated cost volume
- **GPU**: Shared memory for cost volumes, atomicAdd

### 5. Disparity Refinement
- **Purpose**: Extract final disparity from cost volume
- **Stages**:
  - Winner-takes-all (WTA)
  - Subpixel refinement
  - Left-right consistency check
  - Confidence filtering
  - Hole filling
- **Output**: Refined disparity map

### 6. Bilateral Filter
- **Purpose**: Edge-preserving smoothing
- **Method**: Spatial + range Gaussian weights
- **GPU**: Shared memory for window data

### 7. Temporal Smoother
- **Purpose**: Reduce temporal flicker
- **Method**: IIR filter with motion-adaptive alpha
- **GPU**: Vector operations on GPU arrays

## Advanced Features

### Gradient-Adaptive Penalties
- Adapts P1/P2 penalties based on image gradients
- Higher gradients = higher penalties (preserves edges)
- CPU: OpenCV Sobel gradients
- GPU: Simple gradient computation

### Multi-Scale Pyramid
- Processes images at multiple scales
- Coarse-to-fine disparity estimation
- Improves accuracy on large disparities
- CPU: OpenCV pyrDown
- GPU: Strided access downsampling

## Configuration

### Depth Source Selection
Set in config files (`gpu_config.yaml`, `cpu_config.yaml`):
```yaml
depth_source: 'oakd_hardware'  # Options: 'oakd_hardware', 'custom_sgm'
```

### Acceleration Mode
```yaml
acceleration_mode: 'full_gpu'  # Options: 'cpu_only', 'gpu_light', 'full_gpu', 'auto'
```

### SGM Parameters
```yaml
sgm:
  max_disparity: 64
  p1: 10.0          # Small penalty
  p2: 120.0         # Large penalty
  window_size: 5    # Census window
```

## Usage

### Basic Usage
```python
from stereo.stereo_engine import StereoEngine

config = {
    "acceleration_mode": "full_gpu",
    "sgm": {
        "max_disparity": 64,
        "p1": 10.0,
        "p2": 120.0
    }
}

engine = StereoEngine(config, acceleration_mode="full_gpu")
depth = engine.compute_depth(left, right, baseline_m, fx, motion_score)
```

### Integration with SLAM System
```python
from camera.oakd_manager import OakDStereoManager
from camera.depth_source_selector import DepthSourceSelector
from stereo.stereo_engine import StereoEngine

# Initialize with dual depth source support
config = {"depth_source": "custom_sgm"}
selector = DepthSourceSelector(config)
cam = OakDStereoManager(config)

# Initialize stereo engine if using custom SGM
if selector.is_custom_sgm():
    stereo_engine = StereoEngine(config)
    
# Get frames based on mode
if selector.is_hardware_depth():
    left, depth = cam.get_frames()  # OAK-D hardware depth
else:
    left, right = cam.get_frames()  # Raw L/R
    depth = stereo_engine.compute_depth(left, right, cam.baseline_m, cam.fx)
```

## Performance

### CPU Implementation
- Uses NumPy vectorized operations
- OpenMP pragmas for C/C++ port (future optimization)
- Baseline: ~5-10 FPS at 640x400

### GPU Implementation
- CuPy RawKernels for CUDA acceleration
- Shared memory optimization for coalesced accesses
- Async transfers with CUDA streams
- Expected: 3-5× faster than CPU

## Memory Management

### GPU Memory Pool
- Configurable VRAM limit (default: 2GB)
- Pinned memory limit (default: 512MB)
- Automatic eviction on pressure

### Zero-Copy Transfers
- Pinned memory for faster CPU-GPU transfers
- Async transfers with CUDA streams
- Reduces H2D/D2H latency

## Testing

Run integration tests:
```bash
python tests/test_dual_depth_sources.py
```

## Future Work

- [ ] AVX2 SIMD optimization for CPU
- [ ] CUDA kernel fusion (combine stages)
- [ ] Dynamic disparity range adaptation
- [ ] Real-time parameter tuning
- [ ] Confidence-aware depth filtering
