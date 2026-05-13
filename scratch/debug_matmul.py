import os
import sys

# Standard CUDA 13.1 path
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
bin_path = os.path.join(cuda_path, "bin")

print(f"Testing with CUDA path: {bin_path}")
if os.path.exists(bin_path):
    os.add_dll_directory(bin_path)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # Force torch to initialize CUDA context
        _ = torch.cuda.get_device_properties(0)
    import kornia
    print("Kornia imported successfully")
except Exception as e:
    print(f"Torch/Kornia import failed: {e}")

try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    
    print("Testing basic array ops...")
    x = cp.arange(10)
    print(f"Sum: {(x*x).sum()}")
    
    print("Testing cuBLAS matmul (float32) with LARGE MATRICES...")
    # Large matrices trigger cublasLt heuristics and memory-intensive paths
    size = 2048
    a_large = cp.random.standard_normal((size, size), dtype=cp.float32)
    b_large = cp.matmul(a_large, a_large)
    cp.cuda.Stream.null.synchronize()
    print(f"Large Matmul successful (shape: {b_large.shape})")
    
    print("Testing cuBLAS matmul (uint8) - triggered by crash log...")
    a_u8 = cp.random.randint(0, 255, (128, 128), dtype=cp.uint8)
    try:
        b_u8 = cp.matmul(a_u8, a_u8)
        print("Matmul uint8 successful")
    except Exception as e:
        print(f"Matmul uint8 failed (expected if not supported): {e}")

except Exception as e:
    print(f"CRASH or ERROR: {e}")
    import traceback
    traceback.print_exc()
