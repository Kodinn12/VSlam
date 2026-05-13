from .cupy_utils import cupy_manager

def to_numpy_safe(arr):
    """
    Safely convert GPU array (CuPy or TorchXP) -> NumPy.
    Leaves NumPy unchanged.
    """
    return cupy_manager.to_cpu(arr)


def to_gpu_safe(arr):
    """
    Convert NumPy -> GPU array (CuPy or TorchXP) safely.
    """
    return cupy_manager.to_gpu(arr)
