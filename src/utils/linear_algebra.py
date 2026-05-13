"""Batch linear algebra helpers (3x3 matrices, Mahalanobis)."""

import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

from .cupy_utils import cupy_manager
xp = cupy_manager.get_array_module()
USE_CUPY = cupy_manager.is_available()

def batch_inv3(S):
    """Batch inversion of Nx3x3 matrices (NumPy)."""
    a,b,c = S[:,0,0], S[:,0,1], S[:,0,2]
    d,e,f = S[:,1,0], S[:,1,1], S[:,1,2]
    g,h,i = S[:,2,0], S[:,2,1], S[:,2,2]
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    safe_det = np.where(np.abs(det) < 1e-12,
                        np.where(det < 0, -1e-12, 1e-12), det)
    inv = 1.0 / safe_det
    out = np.empty_like(S)
    out[:,0,0] = (e*i - f*h)*inv; out[:,0,1] = (c*h - b*i)*inv; out[:,0,2] = (b*f - c*e)*inv
    out[:,1,0] = (f*g - d*i)*inv; out[:,1,1] = (a*i - c*g)*inv; out[:,1,2] = (c*d - a*f)*inv
    out[:,2,0] = (d*h - e*g)*inv; out[:,2,1] = (b*g - a*h)*inv; out[:,2,2] = (a*e - b*d)*inv
    return out

def batch_mahal3(dmu, Sigma_sum):
    """Batch Mahalanobis distance (NumPy)."""
    Sinv = batch_inv3(Sigma_sum)
    tmp = np.einsum('ni,nij->nj', dmu, Sinv)
    return np.einsum('ni,ni->n', tmp, dmu)

def batch_inv3_gpu(S):
    """GPU batch inversion (uses RawKernel if available)."""
    if not USE_CUPY or not isinstance(S, xp.ndarray):
        return batch_inv3(np.asarray(S))
    from .se3_ops import batch_inv3_gpu as _gpu_inv3
    return _gpu_inv3(S)

def batch_mahal3_gpu(dmu, Sigma_sum):
    """GPU Mahalanobis distance."""
    if not USE_CUPY or not isinstance(Sigma_sum, xp.ndarray):
        return batch_mahal3(np.asarray(dmu), np.asarray(Sigma_sum))
    Sinv = batch_inv3_gpu(Sigma_sum)
    tmp = xp.matmul(Sinv, dmu[:, :, None])[:, :, 0]
    return xp.sum(tmp * dmu, axis=1)
