"""CuPy utilities for GPU-accelerated SLAM operations."""

import numpy as np
from ..utils.logger import get_logger
import os
import sys
import platform
import re
import importlib.metadata
import subprocess

logger = get_logger(__name__)


def _cublas_healthcheck(timeout_s=8):
    """Return True only if CuPy cuBLAS matmul works in a child process."""
    if os.environ.get("SLAM_SKIP_CUBLAS_HEALTHCHECK", "0") == "1":
        return True

    code = (
        "import cupy as cp\n"
        "a = cp.eye(2, dtype=cp.float32)\n"
        "b = cp.matmul(a, a)\n"
        "cp.cuda.Stream.null.synchronize()\n"
        "print(float(b[0, 0].get()))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            logger.warning(f"CuPy cuBLAS healthcheck failed with exit code {result.returncode}")
            if result.stderr:
                logger.warning(result.stderr.strip().splitlines()[-1])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"CuPy cuBLAS healthcheck timed out after {timeout_s}s")
        return False
    except Exception as e:
        logger.warning(f"CuPy cuBLAS healthcheck could not run: {e}")
        return False


def _torch_cuda_healthcheck(timeout_s=8):
    """Return True only if PyTorch CUDA matmul/linear works in a child process."""
    if os.environ.get("SLAM_SKIP_TORCH_CUDA_HEALTHCHECK", "0") == "1":
        return True

    code = (
        "import torch\n"
        "assert torch.cuda.is_available()\n"
        "x = torch.randn(1, 64, device='cuda')\n"
        "w = torch.randn(64, 64, device='cuda')\n"
        "y = torch.nn.functional.linear(x, w)\n"
        "torch.cuda.synchronize()\n"
        "print(float(y[0, 0].detach().cpu()))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            logger.warning(f"PyTorch CUDA healthcheck failed with exit code {result.returncode}")
            if result.stderr:
                logger.warning(result.stderr.strip().splitlines()[-1])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"PyTorch CUDA healthcheck timed out after {timeout_s}s")
        return False
    except Exception as e:
        logger.warning(f"PyTorch CUDA healthcheck could not run: {e}")
        return False

# ──────────────────────────────────────────────────────────────────
# 1. IMMEDIATE BACKEND GUARDS
# ──────────────────────────────────────────────────────────────────
# Disable OpenCV OpenCL to prevent interference with CuPy/Torch CUDA.
# This must be done BEFORE any other GPU initialization.
try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# DLL Discovery for Windows
if platform.system() == "Windows":
    def _cuda_major(path):
        match = re.search(r"CUDA\\v(\d+)", path or "", re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _installed_cupy_cuda12x():
        try:
            importlib.metadata.version("cupy-cuda12x")
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    cupy_cuda12x = _installed_cupy_cuda12x()
    allow_cuda13 = os.environ.get("SLAM_ALLOW_CUDA13_FOR_CUPY", "0") == "1"

    # cupy-cuda12x must not be forced to load CUDA 13 Toolkit DLLs. On Windows,
    # os.add_dll_directory gives those DLLs high priority and can hard-crash
    # inside cuBLAS before Python can raise an exception.
    if cupy_cuda12x and not allow_cuda13:
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        filtered = [
            p for p in path_parts
            if not ("NVIDIA GPU Computing Toolkit" in p and _cuda_major(p) and _cuda_major(p) >= 13)
        ]
        if len(filtered) != len(path_parts):
            os.environ["PATH"] = os.pathsep.join(filtered)
            logger.warning("Removed CUDA 13 Toolkit paths from PATH for cupy-cuda12x compatibility")

    # Prefer a CUDA 12 Toolkit for cupy-cuda12x. If only CUDA 13 is installed,
    # do not add it; CuPy's wheel/driver runtime can still work without the
    # incompatible Toolkit DLL directory being injected.
    cuda_path = None
    candidate_envs = ["CUDA_PATH_V12_8", "CUDA_PATH_V12_7", "CUDA_PATH_V12_6",
                      "CUDA_PATH_V12_5", "CUDA_PATH_V12_4", "CUDA_PATH_V12_3",
                      "CUDA_PATH_V12_2", "CUDA_PATH_V12_1", "CUDA_PATH_V12_0"]
    for env_name in candidate_envs:
        candidate = os.environ.get(env_name)
        if candidate and os.path.exists(candidate):
            cuda_path = candidate
            break

    env_cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path is None and env_cuda_path and os.path.exists(env_cuda_path):
        major = _cuda_major(env_cuda_path)
        if not cupy_cuda12x or allow_cuda13 or major == 12:
            cuda_path = env_cuda_path
        elif major and major >= 13:
            logger.warning(f"Skipping incompatible CUDA Toolkit for cupy-cuda12x: {env_cuda_path}")

    if cuda_path is None:
        base_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(base_path):
            versions = sorted(os.listdir(base_path), reverse=True)
            for version in versions:
                candidate = os.path.join(base_path, version)
                major = _cuda_major(candidate)
                if cupy_cuda12x and not allow_cuda13 and major and major >= 13:
                    continue
                cuda_path = candidate
                break

    if cuda_path:
        # Standard CUDA bin path - this is where all DLLs reside in v11+
        bin_path = os.path.join(cuda_path, "bin")
        
        if os.path.exists(bin_path):
            try:
                # Add to DLL search path for Python 3.8+
                os.add_dll_directory(bin_path)
                logger.info(f"Added CUDA DLL directory to search path: {bin_path}")
                
                # Also ensure it's in PATH for sub-processes
                if bin_path not in os.environ["PATH"]:
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
            except Exception as e:
                logger.warning(f"Failed to add CUDA DLL directory {bin_path}: {e}")
        else:
            logger.warning(f"CUDA bin directory not found at {bin_path}")
    elif cupy_cuda12x:
        logger.warning("No compatible CUDA 12 Toolkit directory found; not adding CUDA Toolkit DLLs")

# Disable OpenCV OpenCL to prevent interference with CuPy CUDA
try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# CuPy and PyTorch availability check
# Export cp and USE_CUPY for other modules to use safely
cp = None
USE_CUPY = False
CUBLAS_HEALTHY = False

try:
    import importlib
    cp_real = importlib.import_module("cupy")
    if cp_real.cuda.runtime.getDeviceCount() > 0:
        x = cp_real.arange(10)
        if (x * x).sum() == 285:
            CUBLAS_HEALTHY = _cublas_healthcheck()
            if CUBLAS_HEALTHY:
                cp = cp_real
                USE_CUPY = True
                logger.info("CuPy GPU acceleration available and verified")
            else:
                logger.warning("Disabling CuPy because cuBLAS is unstable; using Torch/NumPy fallback")
                cp = np
                USE_CUPY = False
        else:
            logger.warning("CuPy test sum failed. Falling back to NumPy.")
            cp = np
    else:
        logger.warning("No CUDA devices found. Falling back to NumPy.")
        cp = np
except Exception as e:
    logger.warning(f"CuPy not found or failed: {e}. Falling back to NumPy.")
    cp = np
    USE_CUPY = False

try:
    import torch
    if torch.cuda.is_available():
        allow_torch_without_cupy_blas = os.environ.get("SLAM_ALLOW_TORCH_CUDA_WITH_BAD_CUBLAS", "0") == "1"
        if not CUBLAS_HEALTHY and not allow_torch_without_cupy_blas:
            USE_TORCH = False
            logger.warning("Disabling PyTorch CUDA because CuPy/cuBLAS healthcheck failed")
        elif _torch_cuda_healthcheck():
            USE_TORCH = True
            logger.info("PyTorch GPU foundation available")
        else:
            USE_TORCH = False
            logger.warning("Disabling PyTorch CUDA because cuBLAS is unstable")
    else:
        USE_TORCH = False
        logger.info("PyTorch GPU not available")
except ImportError:
    USE_TORCH = False
    logger.info("PyTorch not available")

# Zero-copy interop functions
def torch_to_cupy(tensor):
    """Convert a PyTorch tensor to a CuPy array without copying memory."""
    if not USE_CUPY or not USE_TORCH:
        return tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else tensor
    return cp.from_dlpack(tensor)

def cupy_to_torch(arr):
    """Convert a CuPy array to a PyTorch tensor without copying memory."""
    if not USE_CUPY or not USE_TORCH:
        return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    return torch.from_dlpack(arr.toDlpack())

def to_numpy_safe(arr):
    """Safely convert any GPU array (CuPy or Torch) to NumPy. Use sparingly!"""
    if hasattr(arr, 'get'): # CuPy
        return arr.get()
    if hasattr(arr, 'detach'): # Torch
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

class TorchXP:
    """A NumPy-like wrapper for PyTorch to act as a GPU backend fallback."""
    def __init__(self):
        import torch
        self.torch = torch
        self.device = torch.device('cuda' if USE_TORCH and torch.cuda.is_available() else 'cpu')
        
    class TorchLinalg:
        def __init__(self, torch_mod):
            self.torch = torch_mod
        def norm(self, x, ord=None, axis=None, keepdims=False):
            if isinstance(axis, tuple):
                return self.torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)
            return self.torch.norm(x, p=ord, dim=axis, keepdim=keepdims)
        def inv(self, a):
            return self.torch.linalg.inv(a)
        def det(self, a):
            return self.torch.linalg.det(a)
        def matmul(self, a, b):
            return self.torch.matmul(a, b)

    @property
    def linalg(self):
        return self.TorchLinalg(self.torch)

    def array(self, data, dtype=None):
        t_dtype = self._get_torch_dtype(dtype)
        return self.torch.as_tensor(data, dtype=t_dtype, device=self.device)
    
    def asarray(self, data):
        return self.torch.as_tensor(data, device=self.device)

    def ones(self, shape, dtype=None):
        t_dtype = self._get_torch_dtype(dtype)
        return self.torch.ones(shape, dtype=t_dtype, device=self.device)

    def zeros(self, shape, dtype=None):
        t_dtype = self._get_torch_dtype(dtype)
        return self.torch.zeros(shape, dtype=t_dtype, device=self.device)

    def ones_like(self, a):
        return self.torch.ones_like(a)

    def zeros_like(self, a):
        return self.torch.zeros_like(a)

    def full_like(self, a, fill_value):
        return self.torch.full_like(a, fill_value)

    def arange(self, *args, **kwargs):
        return self.torch.arange(*args, **kwargs, device=self.device)

    def meshgrid(self, *args, **kwargs):
        if 'indexing' not in kwargs:
            kwargs['indexing'] = 'ij'
        return self.torch.meshgrid(*args, **kwargs)

    def stack(self, arrays, axis=0):
        return self.torch.stack(arrays, dim=axis)

    def concatenate(self, arrays, axis=0):
        return self.torch.cat(arrays, dim=axis)

    def hstack(self, tensors):
        return self.torch.cat(tensors, dim=1)

    def roll(self, *args, **kwargs):
        return self.torch.roll(*args, **kwargs)

    def mean(self, a, axis=None, keepdims=False):
        return self.torch.mean(a, dim=axis, keepdim=keepdims)

    def std(self, a, axis=None):
        return self.torch.std(a, dim=axis)

    def clip(self, a, a_min, a_max):
        return self.torch.clamp(a, a_min, a_max)

    def sqrt(self, a):
        return self.torch.sqrt(a)

    def sum(self, a, axis=None):
        return self.torch.sum(a, dim=axis)

    def dot(self, a, b):
        return self.torch.matmul(a, b)

    def matmul(self, a, b):
        return self.torch.matmul(a, b)

    def einsum(self, formula, *operands):
        return self.torch.einsum(formula, *operands)

    def where(self, condition, x, y):
        return self.torch.where(condition, x, y)

    def abs(self, a):
        return self.torch.abs(a)

    def eye(self, n, dtype=None):
        t_dtype = self._get_torch_dtype(dtype)
        return self.torch.eye(n, dtype=t_dtype, device=self.device)

    def nan_to_num(self, a, nan=0.0, posinf=None, neginf=None):
        return self.torch.nan_to_num(a, nan=nan, posinf=posinf, neginf=neginf)

    def sort(self, a, axis=-1):
        return self.torch.sort(a, dim=axis)[0]

    def take(self, a, indices, axis=None):
        return self.torch.index_select(a, axis, indices)

    def unique(self, a, axis=0, return_inverse=False):
        if return_inverse:
            return self.torch.unique(a, dim=axis, sorted=True, return_inverse=True)
        return self.torch.unique(a, dim=axis, sorted=True)

    def cross(self, a, b):
        return self.torch.cross(a, b, dim=-1)

    def swapaxes(self, a, axis1, axis2):
        return self.torch.transpose(a, axis1, axis2)
    
    def transpose(self, a, *axes):
        if not axes:
            return a.T
        return a.permute(*axes)

    def reshape(self, a, shape):
        return self.torch.reshape(a, shape)

    def linspace(self, start, stop, num):
        return self.torch.linspace(start, stop, num, device=self.device)

    def log10(self, a):
        return self.torch.log10(a)

    def isfinite(self, a):
        return self.torch.isfinite(a)
    
    def any(self, a):
        return self.torch.any(a)
    
    def all(self, a):
        return self.torch.all(a)
    
    def maximum(self, a, b):
        return self.torch.maximum(a, b)
    
    def minimum(self, a, b):
        return self.torch.minimum(a, b)
    
    def exp(self, a):
        return self.torch.exp(a)
    
    def log(self, a):
        return self.torch.log(a)
    
    def sin(self, a):
        return self.torch.sin(a)
        
    def cos(self, a):
        return self.torch.cos(a)
        
    def arccos(self, a):
        return self.torch.acos(a)
        
    def trace(self, a, offset=0, axis1=0, axis2=1):
        if a.ndim >= 2:
            return self.torch.diagonal(a, offset=offset, dim1=axis1, dim2=axis2).sum(-1)
        return self.torch.trace(a)

    def asnumpy(self, a):
        if hasattr(a, 'cpu'):
            return a.cpu().numpy()
        return a

    def _get_torch_dtype(self, dtype):
        if dtype is None: return self.torch.float32
        if dtype == self.torch.float32 or dtype == 'float32' or dtype == np.float32: return self.torch.float32
        if dtype == self.torch.float64 or dtype == 'float64' or dtype == np.float64: return self.torch.float64
        if dtype == self.torch.int32 or dtype == 'int32' or dtype == np.int32: return self.torch.int32
        if dtype == self.torch.uint8 or dtype == 'uint8' or dtype == np.uint8: return self.torch.uint8
        return dtype

    @property
    def ndarray(self): return self.torch.Tensor
    @property
    def float32(self): return self.torch.float32
    @property
    def float64(self): return self.torch.float64
    @property
    def int32(self): return self.torch.int32
    @property
    def uint8(self): return self.torch.uint8

class CuPyManager:
    """Manages CuPy GPU operations and memory with safe backend."""
    
    def __init__(self):
        self.cp = None
        self.available = False
        self.stream = None
        self.memory_pool = None
        self.pinned_pool = None
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU detection and safe backend."""
        if USE_CUPY:
            self.cp = cp
            self.available = True
            
            # Test basic GPU operation
            try:
                test_array = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
                test_result = cp.sum(test_array)
                
                # Setup GPU resources if available
                self.stream = cp.cuda.Stream()
                self.memory_pool = cp.get_default_memory_pool()
                self.pinned_pool = cp.get_default_pinned_memory_pool()
                logger.info("CuPy GPU acceleration available and initialized")
            except Exception as e:
                logger.warning(f"CuPy GPU resources setup failed: {e}")
                self.available = False
        else:
            logger.warning("CuPy not available, using NumPy fallback.")
            import numpy as np
            # Shim asnumpy for NumPy
            if not hasattr(np, 'asnumpy'):
                np.asnumpy = np.asarray
            self.cp = np
            self.available = False
    
    def is_available(self):
        """Check if GPU is available."""
        return self.available
    
    def get_array_module(self, use_gpu=True):
        """Get appropriate array module (CuPy, PyTorch, or NumPy)."""
        if self.available and use_gpu:
            return self.cp
        elif USE_TORCH and use_gpu:
            if not hasattr(self, '_torch_xp'):
                self._torch_xp = TorchXP()
            return self._torch_xp
        else:
            import numpy as np
            return np
    
    def to_gpu(self, array, stream=None):
        """Convert NumPy array to CuPy array."""
        if self.available and isinstance(array, np.ndarray):
            stream = stream or self.stream
            if stream:
                with stream:
                    return self.cp.asarray(array)
            else:
                return self.cp.asarray(array)
        return array
    
    def to_cpu(self, array):
        """Convert GPU array (CuPy or PyTorch) to NumPy array."""
        if self.available and hasattr(array, 'get'):
            return array.get()
        if hasattr(array, 'cpu'):
            if hasattr(array, 'detach'):
                return array.detach().cpu().numpy()
            return array.cpu().numpy()
        return array
    
    def synchronize(self):
        """Synchronize GPU operations."""
        if self.available and self.stream:
            self.stream.synchronize()
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if self.available and self.memory_pool:
            self.memory_pool.free_all_blocks()
            if self.pinned_pool:
                self.pinned_pool.free_all_blocks()
    
    def get_memory_info(self):
        """Get GPU memory information."""
        info = {}
        if self.available:
            try:
                meminfo = self.cp.cuda.Device(0).mem_info
                info['total'] = meminfo[1] / (1024**3)  # GB
                info['free'] = meminfo[0] / (1024**3)   # GB
                info['used'] = info['total'] - info['free']
                
                if self.memory_pool:
                    info['pool_used'] = self.memory_pool.used_bytes() / (1024**2)  # MB
                    info['pool_total'] = self.memory_pool.total_bytes() / (1024**2)  # MB
            except Exception:
                pass
        return info

# Global CuPy manager instance
# Global instance for backward compatibility
# Replaced with a lazy property to prevent import-time crashes
_cupy_manager_instance = None

def get_cupy_manager():
    global _cupy_manager_instance
    if _cupy_manager_instance is None:
        logger.info("Lazy-initializing CuPyManager...")
        _cupy_manager_instance = CuPyManager()
    return _cupy_manager_instance

# Mock object that delegates to the lazy instance
class _LazyCuPyManagerProxy:
    def __getattr__(self, name):
        return getattr(get_cupy_manager(), name)

cupy_manager = _LazyCuPyManagerProxy()

def cupy_wrapper(func):
    """Decorator to automatically handle CuPy/NumPy array conversion."""
    def wrapper(*args, **kwargs):
        # Convert inputs to GPU if needed
        gpu_args = []
        for arg in args:
            if isinstance(arg, np.ndarray) and USE_CUPY:
                gpu_args.append(cupy_manager.to_gpu(arg))
            else:
                gpu_args.append(arg)
        
        # Call function
        result = func(*gpu_args, **kwargs)
        
        # Convert output to CPU if needed
        if isinstance(result, (list, tuple)):
            return tuple(cupy_manager.to_cpu(r) if hasattr(r, 'get') else r for r in result)
        elif hasattr(result, 'get'):
            return cupy_manager.to_cpu(result)
        else:
            return result
    
    return wrapper

# CuPy-optimized mathematical operations
def batch_matrix_multiply(A, B, use_gpu=True):
    """Batch matrix multiplication with GPU acceleration."""
    xp = cupy_manager.get_array_module(use_gpu)
    if len(A.shape) == 2 and len(B.shape) == 2:
        return xp.dot(A, B)
    elif len(A.shape) == 3 and len(B.shape) == 3:
        return xp.einsum('bij,bjk->bik', A, B)
    elif len(A.shape) == 2 and len(B.shape) == 3:
        # Multiply 2D matrix (3,3) with batch of 3D matrices (N,3,3)
        return xp.einsum('ij,njk->nik', A, B)
    elif len(A.shape) == 3 and len(B.shape) == 2:
        # Multiply batch of 3D matrices (N,3,3) with 2D matrix (3,3)
        return xp.einsum('nij,jk->nik', A, B)
    else:
        raise ValueError(f"Unsupported shapes: A={A.shape}, B={B.shape}")

def batch_matrix_inverse(A, use_gpu=True):
    """Batch matrix inversion with GPU acceleration."""
    xp = cupy_manager.get_array_module(use_gpu)
    if len(A.shape) == 2:
        return xp.linalg.inv(A)
    elif len(A.shape) == 3:
        # Batch inversion for 3x3 matrices
        if A.shape[-2:] == (3, 3):
            return _batch_inv3x3(A, xp)
        else:
            return xp.linalg.inv(A)
    else:
        raise ValueError(f"Unsupported shape: {A.shape}")

def _batch_inv3x3(A, xp):
    """Optimized batch inversion for 3x3 matrices."""
    # Extract elements
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a10, a11, a12 = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    a20, a21, a22 = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    
    # Calculate determinant
    det = (a00 * (a11 * a22 - a12 * a21) -
           a01 * (a10 * a22 - a12 * a20) +
           a02 * (a10 * a21 - a11 * a20))
    
    # Avoid division by zero
    det = xp.where(xp.abs(det) < 1e-10, 1e-10, det)
    
    # Calculate inverse elements
    inv_det = 1.0 / det
    
    inv = xp.empty_like(A)
    inv[..., 0, 0] = inv_det * (a11 * a22 - a12 * a21)
    inv[..., 0, 1] = inv_det * (a02 * a21 - a01 * a22)
    inv[..., 0, 2] = inv_det * (a01 * a12 - a02 * a11)
    inv[..., 1, 0] = inv_det * (a12 * a20 - a10 * a22)
    inv[..., 1, 1] = inv_det * (a00 * a22 - a02 * a20)
    inv[..., 1, 2] = inv_det * (a02 * a10 - a00 * a12)
    inv[..., 2, 0] = inv_det * (a10 * a21 - a11 * a20)
    inv[..., 2, 1] = inv_det * (a01 * a20 - a00 * a21)
    inv[..., 2, 2] = inv_det * (a00 * a11 - a01 * a10)
    
    return inv

def to_numpy_safe(arr):
    """Safely convert GPU array (CuPy or PyTorch) to NumPy."""
    return cupy_manager.to_cpu(arr)

def to_cupy_safe(arr):
    """Safely convert NumPy array to CuPy with float32 dtype."""
    if USE_CUPY:
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr.astype(np.float32))
        elif isinstance(arr, cp.ndarray):
            return arr.astype(cp.float32)
        else:
            return cp.array(arr, dtype=cp.float32)
    
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32)
    return np.array(arr, dtype=np.float32)

def assert_cpu(x):
    """Assert that x is a NumPy array (CPU zone)."""
    import numpy as np
    assert isinstance(x, np.ndarray), f"Expected NumPy array, got {type(x)}"
    return x

def assert_gpu(x):
    """Assert that x is a CuPy array (GPU zone)."""
    if USE_CUPY:
        assert isinstance(x, cp.ndarray), f"Expected CuPy array, got {type(x)}"
    else:
        assert isinstance(x, np.ndarray), f"Expected NumPy array, got {type(x)}"
    return x

def batch_transform_points(points, transforms, use_gpu=True):
    """Batch transform points using transformation matrices."""
    xp = cupy_manager.get_array_module(use_gpu)
    
    if len(points.shape) == 2:  # Single point cloud
        if len(transforms.shape) == 2:  # Single transform
            # Convert to homogeneous coordinates
            ones = xp.ones((points.shape[0], 1), dtype=points.dtype)
            points_h = xp.hstack([points, ones])
            # Transform
            transformed = points_h @ transforms.T
            return transformed[:, :3]
        else:  # Multiple transforms
            ones = xp.ones((points.shape[0], 1), dtype=points.dtype)
            points_h = xp.hstack([points, ones])
            # Batch transform
            transformed = xp.einsum('ij,nj->ni', transforms, points_h)
            return transformed[..., :3]
    else:  # Batch of point clouds
        ones = xp.ones((points.shape[0], points.shape[1], 1), dtype=points.dtype)
        points_h = xp.concatenate([points, ones], axis=-1)
        transformed = xp.einsum('bij,bnk->bik', transforms, points_h)
        return transformed[..., :3]

def batch_distance_matrix(points1, points2, use_gpu=True):
    """Compute batch distance matrix between point sets."""
    xp = cupy_manager.get_array_module(use_gpu)
    
    if len(points1.shape) == 2 and len(points2.shape) == 2:
        # Single distance matrix
        diff = points1[:, None, :] - points2[None, :, :]
        dists = xp.sqrt(xp.sum(diff ** 2, axis=-1))
        return dists
    else:
        # Batch distance matrices
        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dists = xp.sqrt(xp.sum(diff ** 2, axis=-1))
        return dists

def gpu_reduce_mean(array, axis=None, use_gpu=True):
    """GPU-accelerated mean reduction."""
    xp = cupy_manager.get_array_module(use_gpu)
    return xp.mean(array, axis=axis)

def gpu_reduce_std(array, axis=None, use_gpu=True):
    """GPU-accelerated standard deviation."""
    xp = cupy_manager.get_array_module(use_gpu)
    return xp.std(array, axis=axis)

def gpu_reduce_percentile(array, q, axis=None, use_gpu=True):
    """GPU-accelerated percentile calculation."""
    xp = cupy_manager.get_array_module(use_gpu)
    if USE_CUPY and use_gpu:
        # CuPy doesn't have percentile, so we sort and index
        sorted_array = xp.sort(array, axis=axis)
        if axis is None:
            idx = int(q * (sorted_array.size - 1) / 100)
            return float(sorted_array.ravel()[idx])
        else:
            idx = int(q * (sorted_array.shape[axis] - 1) / 100)
            return xp.take(sorted_array, idx, axis=axis)
    else:
        return np.percentile(array, q, axis=axis)

def gpu_clip(array, min_val, max_val, use_gpu=True):
    """GPU-accelerated clipping."""
    xp = cupy_manager.get_array_module(use_gpu)
    return xp.clip(array, min_val, max_val)

def gpu_norm(array, axis=None, keepdims=False, use_gpu=True):
    """GPU-accelerated norm calculation."""
    xp = cupy_manager.get_array_module(use_gpu)
    return xp.linalg.norm(array, axis=axis, keepdims=keepdims)

def gpu_normalize_vectors(vectors, use_gpu=True):
    """Normalize vectors to unit length."""
    xp = cupy_manager.get_array_module(use_gpu)
    norms = xp.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = xp.where(norms < 1e-10, 1.0, norms)
    return vectors / norms

def stable_batched_covariance_transform(R_batch, Sigma_batch, regularization=1e-6, use_gpu=True):
    """
    Stable batched covariance transformation: Sigma' = R @ Sigma @ R.T
    
    Args:
        R_batch: (N, 3, 3) rotation matrices
        Sigma_batch: (N, 3, 3) covariance matrices  
        regularization: float for numerical stability
        use_gpu: bool to use CuPy or NumPy
        
    Returns:
        (N, 3, 3) transformed covariance matrices
    """
    xp = cupy_manager.get_array_module(use_gpu)
    
    # Add regularization to prevent singular matrices
    if regularization > 0:
        reg_matrix = regularization * xp.eye(3, dtype=Sigma_batch.dtype)
        Sigma_reg = Sigma_batch + reg_matrix[None, :, :]
    else:
        Sigma_reg = Sigma_batch
    
    # Proper batched matrix multiplication: R @ Sigma @ R.T
    # First compute R @ Sigma
    RS = xp.einsum('nij,njk->nik', R_batch, Sigma_reg)
    
    # Then compute (R @ Sigma) @ R.T
    Sigma_transformed = xp.einsum('nij,nkj->nik', RS, R_batch)
    
    # NaN/Inf guards
    Sigma_transformed = xp.nan_to_num(Sigma_transformed, nan=regularization, posinf=1e6, neginf=1e-6)
    
    return Sigma_transformed

def stable_batched_covariance_transform_with_points(R_batch, points_batch, Sigma_ray_batch, regularization=1e-6, use_gpu=True):
    """
    Stable batched covariance transform for ray-based uncertainties: Sigma_world = R @ Sigma_ray @ R.T
    
    Args:
        R_batch: (N, 3, 3) rotation matrices
        points_batch: (N, 3) 3D points (for validation)
        Sigma_ray_batch: (N, 3, 3) ray covariance matrices
        regularization: float for numerical stability
        use_gpu: bool to use CuPy or NumPy
        
    Returns:
        (N, 3, 3) transformed covariance matrices
    """
    xp = cupy_manager.get_array_module(use_gpu)
    
    # Validate inputs
    if R_batch.shape[0] != points_batch.shape[0] or R_batch.shape[0] != Sigma_ray_batch.shape[0]:
        raise ValueError("Batch sizes must match")
    
    # Add regularization to prevent singular matrices
    if regularization > 0:
        reg_matrix = regularization * xp.eye(3, dtype=Sigma_ray_batch.dtype)
        Sigma_reg = Sigma_ray_batch + reg_matrix[None, :, :]
    else:
        Sigma_reg = Sigma_ray_batch
    
    # Proper batched matrix multiplication
    RS = xp.einsum('nij,njk->nik', R_batch, Sigma_reg)
    Sigma_transformed = xp.einsum('nij,nkj->nik', RS, R_batch)
    
    # Ensure positive definite and finite
    diag_elements = xp.diagonal(Sigma_transformed, axis1=1, axis2=2)
    diag_elements = xp.maximum(diag_elements, regularization)  # Ensure positive diagonal
    Sigma_transformed = xp.nan_to_num(Sigma_transformed, nan=regularization, posinf=1e6, neginf=1e-6)
    
    return Sigma_transformed

def batched_matrix_multiply_3x3(A_batch, B_batch, regularization=1e-6, use_gpu=True):
    """
    Stable batched 3x3 matrix multiplication with error handling
    
    Args:
        A_batch: (N, 3, 3) matrices
        B_batch: (N, 3, 3) matrices  
        regularization: float for numerical stability
        use_gpu: bool to use CuPy or NumPy
        
    Returns:
        (N, 3, 3) result matrices
    """
    xp = cupy_manager.get_array_module(use_gpu)
    
    # Validate shapes
    if A_batch.shape != B_batch.shape:
        raise ValueError(f"Shape mismatch: {A_batch.shape} vs {B_batch.shape}")
    
    # Batched matrix multiplication
    result = xp.einsum('nij,njk->nik', A_batch, B_batch)
    
    # NaN/Inf guards
    result = xp.nan_to_num(result, nan=regularization, posinf=1e6, neginf=1e-6)
    
    return result

def gpu_cross_product(a, b, use_gpu=True):
    """GPU-accelerated cross product."""
    xp = cupy_manager.get_array_module(use_gpu)
    return xp.cross(a, b)

def gpu_dot_product(a, b, use_gpu=True):
    """GPU-accelerated dot product."""
    xp = cupy_manager.get_array_module(use_gpu)
    return xp.dot(a, b)

def gpu_einsum(equation, *operands, use_gpu=True):
    """GPU-accelerated Einstein summation."""
    xp = cupy_manager.get_array_module(use_gpu)
    return xp.einsum(equation, *operands)

def rotate_covariance_batch(R, Sigma_batch):
    """
    Rotate batch of covariance matrices: Sigma_rot = R @ Sigma @ R.T (V59 Robust)
    
    Args:
        R: (3, 3) rotation matrix
        Sigma_batch: (N, 3, 3) batch of covariance matrices
        
    Returns:
        (N, 3, 3) rotated covariance matrices
    """
    is_gpu_array = bool(
        USE_CUPY and
        cp is not np and
        hasattr(cp, "ndarray") and
        isinstance(Sigma_batch, cp.ndarray)
    )
    xp = cp if is_gpu_array else np
    
    try:
        # Ensure R is on the same device as Sigma_batch
        if is_gpu_array:
            R_g = xp.asarray(R, dtype=Sigma_batch.dtype)
        else:
            R_g = np.asarray(R, dtype=Sigma_batch.dtype)

        Rb = R_g[None, :, :]  # (1,3,3)
        Rt = R_g.T[None, :, :]  # (1,3,3)
        
        # Proper batch matrix multiplication: R @ Sigma @ R.T
        Sigma_rot = xp.matmul(xp.matmul(Rb, Sigma_batch), Rt)
        
        # Enforce symmetry
        Sigma_rot = 0.5 * (Sigma_rot + Sigma_rot.transpose(0, 2, 1))
        
        # Add regularization for numerical stability
        eps = 1e-6
        Sigma_rot += eps * xp.eye(3)[None, :, :]
        
        return Sigma_rot
        
    except Exception as e:
        print("[CUDA FAIL] fallback CPU:", e)
        
        Sigma_cpu = Sigma_batch.get() if hasattr(Sigma_batch, "get") else Sigma_batch
        
        # CPU fallback
        Sigma_rot = np.matmul(
            np.matmul(R[None, :, :], Sigma_cpu),
            R.T[None, :, :]
        )
        
        # Enforce symmetry and regularization on CPU
        Sigma_rot = 0.5 * (Sigma_rot + Sigma_rot.transpose(0, 2, 1))
        eps = 1e-6
        Sigma_rot += eps * np.eye(3)[None, :, :]
        
        return Sigma_rot
