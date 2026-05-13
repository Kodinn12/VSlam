import sys
import os
import platform

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

print(f"Python: {sys.version}")

# Try to use the project's own DLL initialization logic
try:
    from utils.cupy_utils import cupy_manager
    print("CuPy Manager imported")
    print(f"CuPy Available: {cupy_manager.is_available()}")
    
    if cupy_manager.available:
        cp = cupy_manager.cp
        print(f"CuPy Version: {getattr(cp, '__version__', 'UNKNOWN')}")
        
        # Test matmul
        try:
            a = cp.array([[1.0, 0.0], [0.0, 1.0]], dtype=cp.float32)
            b = cp.array([[1.0, 0.0], [0.0, 1.0]], dtype=cp.float32)
            c = cp.matmul(a, b)
            print("Matmul Test: SUCCESS")
        except Exception as e:
            print(f"Matmul Test: FAILED - {e}")
            
except Exception as e:
    print(f"Diagnosis Failed: {e}")
    import traceback
    traceback.print_exc()
