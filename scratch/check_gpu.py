import cupy as cp
import numpy as np
import time

def check_gpu():
    print("=== GPU Diagnostic ===")
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
        device = cp.cuda.Device(0)
        print(f"Device: {device}")
        
        a = cp.random.random((1000, 1000), dtype=cp.float32)
        start = time.time()
        b = cp.matmul(a, a)
        cp.cuda.Stream.null.synchronize()
        print(f"cuBLAS Matmul: OK ({time.time() - start:.4f}s)")
        
        return True
    except Exception as e:
        print(f"GPU Error: {e}")
        return False

if __name__ == "__main__":
    check_gpu()
