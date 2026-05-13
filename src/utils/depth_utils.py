"""GPU-accelerated bilinear depth sampling."""

import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

from .cupy_utils import cupy_manager
xp = cupy_manager.get_array_module()
USE_CUPY = cupy_manager.is_available()

_BLD_KERNEL = None

def _get_bld_kernel():
    global _BLD_KERNEL
    if _BLD_KERNEL is not None:
        return _BLD_KERNEL
    if not USE_CUPY:
        return None
    _BLD_KERNEL = xp.RawKernel(r'''
extern "C" __global__
void bilinear_depth_kernel(
    const float* __restrict__ depth,
    const float* __restrict__ u,
    const float* __restrict__ v,
    double* __restrict__ out,
    int H, int W, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float uf = u[i], vf = v[i];
    int u0 = (int)floorf(uf), u1 = u0 + 1;
    int v0 = (int)floorf(vf), v1 = v0 + 1;
    if (u0 < 0 || v0 < 0 || u1 >= W || v1 >= H) { out[i] = 0.0; return; }
    float du = uf - u0, dv = vf - v0;
    float d = depth[v0 * W + u0] * (1.f - du) * (1.f - dv)
            + depth[v0 * W + u1] *        du   * (1.f - dv)
            + depth[v1 * W + u0] * (1.f - du) *        dv
            + depth[v1 * W + u1] *        du   *        dv;
    out[i] = (double)d;
}
''', 'bilinear_depth_kernel')
    return _BLD_KERNEL

def bilinear_depth(depth, u, v):
    """Vectorized bilinear depth sampling (backend-agnostic)."""
    u0 = xp.floor(u).astype(xp.int32)
    u1 = u0 + 1
    v0 = xp.floor(v).astype(xp.int32)
    v1 = v0 + 1
    h, w = depth.shape
    du = u - u0; dv = v - v0
    in_bounds = (u0 >= 0) & (v0 >= 0) & (u1 < w) & (v1 < h)
    out = xp.zeros_like(u, dtype=xp.float64)
    u0c, v0c = xp.clip(u0, 0, w-1), xp.clip(v0, 0, h-1)
    u1c, v1c = xp.clip(u1, 0, w-1), xp.clip(v1, 0, h-1)
    
    # Use masking for safe sampling
    if xp.any(in_bounds):
        # Slice inputs to only in-bounds points for performance/safety if needed
        # but vectorized math is often fine with clips
        out = (depth[v0c, u0c] * (1-du) * (1-dv) +
               depth[v0c, u1c] * du * (1-dv) +
               depth[v1c, u0c] * (1-du) * dv +
               depth[v1c, u1c] * du * dv)
        # Apply bounds mask
        out = xp.where(in_bounds, out, xp.zeros_like(out))
    return out

def bilinear_depth_gpu(depth_gpu, u, v, return_gpu=False):
    """GPU bilinear depth sampling."""
    kern = _get_bld_kernel()
    if kern is None or not USE_CUPY or not isinstance(depth_gpu, xp.ndarray):
        # fallback to CPU
        d_cpu = depth_gpu if isinstance(depth_gpu, np.ndarray) else xp.asnumpy(depth_gpu)
        u_np = xp.asnumpy(u) if isinstance(u, xp.ndarray) else np.asarray(u)
        v_np = xp.asnumpy(v) if isinstance(v, xp.ndarray) else np.asarray(v)
        return bilinear_depth(d_cpu, u_np, v_np)
    u_g = u.astype(xp.float32) if isinstance(u, xp.ndarray) else xp.asarray(u, dtype=xp.float32)
    v_g = v.astype(xp.float32) if isinstance(v, xp.ndarray) else xp.asarray(v, dtype=xp.float32)
    N = u_g.shape[0]
    if N == 0:
        return xp.zeros(0, dtype=xp.float64) if return_gpu else np.zeros(0, dtype=np.float64)
    H, W = depth_gpu.shape
    d32 = depth_gpu.astype(xp.float32) if depth_gpu.dtype != xp.float32 else depth_gpu
    out_g = xp.empty(N, dtype=xp.float64)
    threads = 256
    blocks = (N + threads - 1) // threads
    kern((blocks,), (threads,), (d32, u_g, v_g, out_g, H, W, N))
    return out_g if return_gpu else xp.asnumpy(out_g)
