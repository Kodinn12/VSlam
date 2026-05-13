#!/usr/bin/env python3
"""Run SLAM in GPU-accelerated mode."""

import sys
import os

# Professional CUDA 13.1 stability fixes
# Disable problematic CUB reductions on CUDA 13.1
os.environ["CUPY_ACCELERATORS"] = ""
# Safer JIT behavior
os.environ["CUPY_COMPILE_WITH_PTX"] = "1"
# Additional stability fixes
os.environ["CUPY_CACHE_JIT"] = "0"
os.environ["CUPY_DISABLE_JITIFY_CACHE"] = "1"

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run SLAM with GPU configuration."""
    print("=== Starting SLAM System (GPU Mode) ===")
    
    # Set config file for GPU mode
    config_file = 'config/gpu_config.yaml'
    
    if not os.path.exists(config_file):
        print(f"GPU config file not found: {config_file}")
        sys.exit(1)
    
    print(f"Using GPU config: {config_file}")
    
    # Check GPU availability
    try:
        import cupy as cp
        print("GPU: CuPy available - Full GPU acceleration enabled")
    except ImportError:
        print("GPU: CuPy not available - Falling back to CPU mode")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU: PyTorch CUDA available")
        else:
            print("GPU: PyTorch CUDA not available")
    except ImportError:
        print("GPU: PyTorch not available")
    
    try:
        import kornia
        print("GPU: Kornia geometry operations available")
    except ImportError:
        print("GPU: Kornia not available")
    
    # Override config in main module for maximum performance
    import src.main as main_module
    original_config = getattr(main_module, 'config_file', None)
    main_module.config_file = config_file
    
    # DISABLE DATA SAVING FOR MAXIMUM SPEED
    # Override save functions to do nothing
    def dummy_save(*args, **kwargs):
        pass
    
    # Override all save operations
    main_module.save_dataset = dummy_save
    main_module.save_keyframes = dummy_save
    main_module.save_voxel_grid = dummy_save
    main_module.save_bubble_map = dummy_save
    main_module.save_reconstruction_state = dummy_save
    
    # DISABLE DATASET GENERATION FOR MAXIMUM SPEED
    def dummy_generate(*args, **kwargs):
        pass
    
    # Override dataset generation
    main_module.generate_dataset = dummy_generate
    
    # DISABLE FOLDER OPERATIONS FOR MAXIMUM SPEED
    def dummy_folder(*args, **kwargs):
        pass
    
    # Override folder operations
    main_module.create_output_directory = dummy_folder
    main_module.ensure_output_directory = dummy_folder
    
    print("[PERFORMANCE] Data saving, dataset generation, and folder operations disabled for maximum speed")
    
    # Import and run SLAM
    try:
        from src.main import main as slam_main
        slam_main(["--config", config_file])
    except KeyboardInterrupt:
        print("\nSLAM stopped by user")
    except Exception as e:
        import traceback

        print(f"SLAM error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
