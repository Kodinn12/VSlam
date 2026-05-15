"""SE(3) exponential and logarithmic maps, batch GPU versions."""

import numpy as np
import math
from .logger import get_logger

logger = get_logger(__name__)

from .cupy_utils import cupy_manager, USE_TORCH, cp
xp = cupy_manager.get_array_module()
USE_CUPY = cupy_manager.is_available()

# ----------------------------------------------------------------------
# CUDA RawKernels (compiled once)
# ----------------------------------------------------------------------
_BATCH_INV3_KERNEL = None
_BATCH_EXP_SE3_KERNEL = None
_BATCH_LOG_SE3_KERNEL = None

if USE_CUPY:
    _INV3_SRC = r"""
extern "C" __global__
void batch_inv3_kernel(const double* __restrict__ S,
                       double* __restrict__ out, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int b = n * 9;
    double a=S[b+0], bv=S[b+1], c=S[b+2];
    double d=S[b+3], e=S[b+4], f=S[b+5];
    double g=S[b+6], h=S[b+7], i=S[b+8];
    double det = a*(e*i - f*h) - bv*(d*i - f*g) + c*(d*h - e*g);
    if (fabs(det) < 1e-12) det = (det < 0.0) ? -1e-12 : 1e-12;
    double inv = 1.0 / det;
    int o = n * 9;
    out[o+0]=(e*i-f*h)*inv; out[o+1]=(c*h-bv*i)*inv; out[o+2]=(bv*f-c*e)*inv;
    out[o+3]=(f*g-d*i)*inv; out[o+4]=(a*i-c*g)*inv; out[o+5]=(c*d-a*f)*inv;
    out[o+6]=(d*h-e*g)*inv; out[o+7]=(bv*g-a*h)*inv; out[o+8]=(a*e-bv*d)*inv;
}
"""
    _EXP_SE3_SRC = r"""
extern "C" __global__
void batch_exp_se3_kernel(const double* __restrict__ xi,
                          double* __restrict__ T_out, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int xi_b = n * 6;
    double px = xi[xi_b+0], py = xi[xi_b+1], pz = xi[xi_b+2];
    double ox = xi[xi_b+3], oy = xi[xi_b+4], oz = xi[xi_b+5];
    double th2 = ox*ox + oy*oy + oz*oz;
    double th  = sqrt(th2);
    double a, b, d;
    if (th < 1e-10) {
        a = 1.0; b = 0.5; d = 1.0/6.0;
    } else {
        double st = sin(th), ct = cos(th);
        a = st / th;
        b = (1.0 - ct) / th2;
        d = (th - st) / (th * th2);
    }
    double K[9] = {0.0, -oz,  oy,
                    oz, 0.0, -ox,
                   -oy,  ox, 0.0};
    double K2[9];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            #pragma unroll
            for (int k = 0; k < 3; k++) s += K[i*3+k] * K[k*3+j];
            K2[i*3+j] = s;
        }
    double R[9];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            R[i*3+j] = (i==j ? 1.0 : 0.0) + a*K[i*3+j] + b*K2[i*3+j];
    double V[9];
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            V[i*3+j] = (i==j ? 1.0 : 0.0) + b*K[i*3+j] + d*K2[i*3+j];
    double tx = V[0]*px + V[1]*py + V[2]*pz;
    double ty = V[3]*px + V[4]*py + V[5]*pz;
    double tz = V[6]*px + V[7]*py + V[8]*pz;
    int T_b = n * 16;
    T_out[T_b+ 0]=R[0]; T_out[T_b+ 1]=R[1]; T_out[T_b+ 2]=R[2]; T_out[T_b+ 3]=tx;
    T_out[T_b+ 4]=R[3]; T_out[T_b+ 5]=R[4]; T_out[T_b+ 6]=R[5]; T_out[T_b+ 7]=ty;
    T_out[T_b+ 8]=R[6]; T_out[T_b+ 9]=R[7]; T_out[T_b+10]=R[8]; T_out[T_b+11]=tz;
    T_out[T_b+12]=0.0;  T_out[T_b+13]=0.0;  T_out[T_b+14]=0.0;  T_out[T_b+15]=1.0;
}
"""
    _LOG_SE3_SRC = r"""
extern "C" __global__
void batch_log_se3_kernel(const double* __restrict__ T,
                          double* __restrict__ xi, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int Tb = n * 16;
    double R00=T[Tb+ 0], R01=T[Tb+ 1], R02=T[Tb+ 2], tx=T[Tb+ 3];
    double R10=T[Tb+ 4], R11=T[Tb+ 5], R12=T[Tb+ 6], ty=T[Tb+ 7];
    double R20=T[Tb+ 8], R21=T[Tb+ 9], R22=T[Tb+10], tz=T[Tb+11];
    double cos_a = (R00 + R11 + R22 - 1.0) * 0.5;
    if (cos_a >  1.0) cos_a =  1.0;
    if (cos_a < -1.0) cos_a = -1.0;
    double angle  = acos(cos_a);
    double sin_a  = sin(angle);
    int xi_b = n * 6;
    if (angle < 1e-10) {
        xi[xi_b+0]=tx; xi[xi_b+1]=ty; xi[xi_b+2]=tz;
        xi[xi_b+3]=0.0; xi[xi_b+4]=0.0; xi[xi_b+5]=0.0;
        return;
    }
    int near_pi = ((3.14159265358979323846 - angle) < 1e-4);
    double safe_sin = near_pi ? 1e-7 : sin_a;
    double haos = angle / (2.0 * safe_sin);
    double kx = (R21 - R12) * haos;
    double ky = (R02 - R20) * haos;
    double kz = (R10 - R01) * haos;
    if (near_pi) {
        xi[xi_b+0]=tx; xi[xi_b+1]=ty; xi[xi_b+2]=tz;
        xi[xi_b+3]=0.0; xi[xi_b+4]=0.0; xi[xi_b+5]=0.0;
        return;
    }
    double one_mc = 1.0 - cos_a;
    if (one_mc < 1e-12) one_mc = 1e-12;
    double half_v = angle * sin_a / (2.0 * one_mc);
    double ang2   = angle * angle;
    if (ang2 < 1e-20) ang2 = 1e-20;
    double coeff  = (1.0 - half_v) / ang2;
    double K2_00 = -(kz*kz + ky*ky);
    double K2_01 =  kx*ky;
    double K2_02 =  kx*kz;
    double K2_10 =  kx*ky;
    double K2_11 = -(kz*kz + kx*kx);
    double K2_12 =  ky*kz;
    double K2_20 =  kx*kz;
    double K2_21 =  ky*kz;
    double K2_22 = -(ky*ky + kx*kx);
    double Vi00 = 1.0           + coeff*K2_00;
    double Vi01 = 0.0 + 0.5*kz + coeff*K2_01;
    double Vi02 = 0.0 - 0.5*ky + coeff*K2_02;
    double Vi10 = 0.0 - 0.5*kz + coeff*K2_10;
    double Vi11 = 1.0           + coeff*K2_11;
    double Vi12 = 0.0 + 0.5*kx + coeff*K2_12;
    double Vi20 = 0.0 + 0.5*ky + coeff*K2_20;
    double Vi21 = 0.0 - 0.5*kx + coeff*K2_21;
    double Vi22 = 1.0           + coeff*K2_22;
    double rho_x = Vi00*tx + Vi01*ty + Vi02*tz;
    double rho_y = Vi10*tx + Vi11*ty + Vi12*tz;
    double rho_z = Vi20*tx + Vi21*ty + Vi22*tz;
    xi[xi_b+0]=rho_x; xi[xi_b+1]=rho_y; xi[xi_b+2]=rho_z;
    xi[xi_b+3]=kx;    xi[xi_b+4]=ky;    xi[xi_b+5]=kz;
}
"""
    try:
        opts = ('--fmad=true', '--prec-div=true', '--prec-sqrt=true')
        _BATCH_INV3_KERNEL = xp.RawKernel(_INV3_SRC, 'batch_inv3_kernel', options=opts)
        _BATCH_EXP_SE3_KERNEL = xp.RawKernel(_EXP_SE3_SRC, 'batch_exp_se3_kernel', options=opts)
        _BATCH_LOG_SE3_KERNEL = xp.RawKernel(_LOG_SE3_SRC, 'batch_log_se3_kernel', options=opts)
        logger.info("CUDA RawKernels compiled for SE(3) ops")
    except Exception as e:
        logger.warning(f"CUDA kernel compilation failed: {e}")
        _BATCH_INV3_KERNEL = _BATCH_EXP_SE3_KERNEL = _BATCH_LOG_SE3_KERNEL = None


def batch_inv3_gpu(S):
    """Batch invert N 3x3 matrices. Uses CuPy RawKernel for N>=64."""
    if not USE_CUPY or not isinstance(S, xp.ndarray):
        return _batch_inv3_numpy(np.asarray(S))
    N = S.shape[0]
    if N >= 64 and _BATCH_INV3_KERNEL is not None:
        try:
            S_c = xp.ascontiguousarray(S.reshape(N, 9))
            out = xp.empty((N, 9), dtype=xp.float32)
            bsz = 256
            gsz = (N + bsz - 1) // bsz
            _BATCH_INV3_KERNEL((gsz,), (bsz,), (S_c, out, np.int32(N)))
            return out.reshape(N, 3, 3)
        except Exception:
            pass
    # fallback: element-wise CuPy
    a,b,c = S[:,0,0], S[:,0,1], S[:,0,2]
    d,e,f = S[:,1,0], S[:,1,1], S[:,1,2]
    g,h,i = S[:,2,0], S[:,2,1], S[:,2,2]
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    safe_det = xp.where(xp.abs(det) < 1e-12,
                        xp.where(det < 0, -1e-12, 1e-12), det)
    inv = 1.0 / safe_det
    out = xp.empty_like(S)
    out[:,0,0] = (e*i - f*h) * inv
    out[:,0,1] = (c*h - b*i) * inv
    out[:,0,2] = (b*f - c*e) * inv
    out[:,1,0] = (f*g - d*i) * inv
    out[:,1,1] = (a*i - c*g) * inv
    out[:,1,2] = (c*d - a*f) * inv
    out[:,2,0] = (d*h - e*g) * inv
    out[:,2,1] = (b*g - a*h) * inv
    out[:,2,2] = (a*e - b*d) * inv
    return out

def _batch_inv3_numpy(S):
    a,b,c = S[:,0,0], S[:,0,1], S[:,0,2]
    d,e,f = S[:,1,0], S[:,1,1], S[:,1,2]
    g,h,i = S[:,2,0], S[:,2,1], S[:,2,2]
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    safe_det = np.where(np.abs(det) < 1e-12,
                        np.where(det < 0, -1e-12, 1e-12), det)
    inv = 1.0 / safe_det
    out = np.empty_like(S)
    out[:,0,0] = (e*i - f*h) * inv
    out[:,0,1] = (c*h - b*i) * inv
    out[:,0,2] = (b*f - c*e) * inv
    out[:,1,0] = (f*g - d*i) * inv
    out[:,1,1] = (a*i - c*g) * inv
    out[:,1,2] = (c*d - a*f) * inv
    out[:,2,0] = (d*h - e*g) * inv
    out[:,2,1] = (b*g - a*h) * inv
    out[:,2,2] = (a*e - b*d) * inv
    return out


def batch_exp_se3_gpu(xi_batch):
    """Batch exponential map for SE(3)."""
    xi = xp.asarray(xi_batch)
    N = xi.shape[0]
    if USE_CUPY and N >= 64 and _BATCH_EXP_SE3_KERNEL is not None:
        try:
            xi_c = xp.ascontiguousarray(xi.astype(xp.float64))
            T_out = xp.empty((N, 16), dtype=xp.float64)
            bsz = 256
            gsz = (N + bsz - 1) // bsz
            _BATCH_EXP_SE3_KERNEL((gsz,), (bsz,), (xi_c, T_out, np.int32(N)))
            return T_out.reshape(N, 4, 4)
        except Exception:
            pass
    rho = xi[:, :3]
    omega = xi[:, 3:]
    theta = xp.linalg.norm(omega, axis=1, keepdims=True)
    K = xp.zeros((N, 3, 3), dtype=xi.dtype)
    K[:, 0, 1] = -omega[:, 2]; K[:, 0, 2] =  omega[:, 1]
    K[:, 1, 0] =  omega[:, 2]; K[:, 1, 2] = -omega[:, 0]
    K[:, 2, 0] = -omega[:, 1]; K[:, 2, 1] =  omega[:, 0]
    K2 = xp.matmul(K, K)
    tf = theta.flatten()
    small = tf < 1e-10
    st = xp.sin(tf)
    ct = xp.cos(tf)
    tf_safe = xp.where(small, xp.ones_like(tf), tf)
    tf2_safe = tf_safe * tf_safe
    a = xp.where(small, xp.ones_like(tf), st / tf_safe)
    b = xp.where(small, xp.full_like(tf, 0.5), (1.0 - ct) / tf2_safe)
    d = xp.where(small, xp.full_like(tf, 1.0/6.0), (tf - st) / (tf_safe * tf2_safe))
    eye3 = xp.eye(3, dtype=xi.dtype)[None, :, :]
    R = eye3 + a[:, None, None] * K + b[:, None, None] * K2
    V = eye3 + b[:, None, None] * K + d[:, None, None] * K2
    t = xp.matmul(V, rho[:, :, None])[:, :, 0]
    T = xp.zeros((N, 4, 4), dtype=xi.dtype)
    T[:, 3, 3] = 1.0
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def batch_log_se3_gpu(T_batch):
    """Batch logarithmic map for SE(3)."""
    T = xp.asarray(T_batch)
    N = T.shape[0]
    if USE_CUPY and N >= 64 and _BATCH_LOG_SE3_KERNEL is not None:
        try:
            T_c = xp.ascontiguousarray(T.reshape(N, 16).astype(xp.float64))
            xi_out = xp.empty((N, 6), dtype=xp.float64)
            bsz = 256
            gsz = (N + bsz - 1) // bsz
            _BATCH_LOG_SE3_KERNEL((gsz,), (bsz,), (T_c, xi_out, np.int32(N)))
            return xi_out
        except Exception:
            pass
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    cos_a = xp.clip((xp.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = xp.arccos(cos_a)
    small = angle < 1e-10
    near_pi = (xp.pi - angle) < 1e-4
    sin_a = xp.sin(angle)
    safe_sin = xp.where(small, xp.ones_like(sin_a),
                        xp.where(near_pi, xp.full_like(sin_a, 1e-7), sin_a))
    haos = angle / (2.0 * safe_sin)
    RmRt = R - xp.swapaxes(R, 1, 2)
    omega = xp.stack([RmRt[:, 2, 1], RmRt[:, 0, 2], RmRt[:, 1, 0]], axis=1) * haos[:, None]
    angle_safe = xp.maximum(angle, 1e-10)
    omega_n = omega / angle_safe[:, None]
    Kn = xp.zeros((N, 3, 3), dtype=T.dtype)
    Kn[:, 0, 1] = -omega_n[:, 2]; Kn[:, 0, 2] =  omega_n[:, 1]
    Kn[:, 1, 0] =  omega_n[:, 2]; Kn[:, 1, 2] = -omega_n[:, 0]
    Kn[:, 2, 0] = -omega_n[:, 1]; Kn[:, 2, 1] =  omega_n[:, 0]
    Kn2 = xp.matmul(Kn, Kn)
    s1mc = xp.where(small, xp.ones_like(cos_a), 1.0 - cos_a)
    s1mc_safe = xp.maximum(s1mc, 1e-12)
    half = angle * sin_a / (2.0 * s1mc_safe)
    coeff = (1.0 / xp.maximum(angle**2, 1e-20)) * (1.0 - half)
    skw = xp.zeros((N, 3, 3), dtype=T.dtype)
    skw[:, 0, 1] = -omega[:, 2]; skw[:, 0, 2] =  omega[:, 1]
    skw[:, 1, 0] =  omega[:, 2]; skw[:, 1, 2] = -omega[:, 0]
    skw[:, 2, 0] = -omega[:, 1]; skw[:, 2, 1] =  omega[:, 0]
    eye3 = xp.eye(3, dtype=T.dtype)[None, :, :]
    V_inv = eye3 - 0.5 * skw + coeff[:, None, None] * Kn2
    rho = xp.matmul(V_inv, t[:, :, None])[:, :, 0]
    rho = xp.where(small[:, None], t, rho)
    omega = xp.where(small[:, None], xp.zeros_like(omega), omega)
    rho = xp.where(near_pi[:, None], t, rho)
    omega = xp.where(near_pi[:, None], xp.zeros_like(omega), omega)
    return xp.concatenate([rho, omega], axis=1)


def se3_inv_gpu(T_gpu):
    """Inverse of SE(3) matrix (or batch of matrices) on GPU."""
    xp_local = cp if USE_CUPY else np
    T = xp_local.asarray(T_gpu, dtype=xp_local.float64)
    
    if T.ndim == 2:
        # Single matrix (4, 4)
        R = T[:3, :3]
        t = T[:3, 3]
        Rt = R.T
        Ti = xp_local.eye(4, dtype=xp_local.float64)
        Ti[:3, :3] = Rt
        Ti[:3, 3] = -(Rt @ t)
        return Ti
    elif T.ndim == 3:
        # Batch of matrices (N, 4, 4)
        N = T.shape[0]
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        Rt = xp_local.swapaxes(R, 1, 2)
        Ti = xp_local.zeros((N, 4, 4), dtype=xp_local.float64)
        Ti[:, 3, 3] = 1.0
        Ti[:, :3, :3] = Rt
        # t is (N, 3), Rt is (N, 3, 3)
        # We need (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1) -> (N, 3)
        Ti[:, :3, 3] = -xp_local.matmul(Rt, t[:, :, None])[:, :, 0]
        return Ti
    else:
        raise ValueError(f"se3_inv_gpu: Expected 2D or 3D input, got {T.ndim}D shape {T.shape}")


def batch_matmul_gpu(A, B):
    """Batch matrix multiply (N,4,4) @ (N,4,4)."""
    if USE_CUPY:
        A = xp.asarray(A) if not isinstance(A, xp.ndarray) else A
        B = xp.asarray(B) if not isinstance(B, xp.ndarray) else B
        return xp.matmul(A, B)
    return np.einsum('nij,njk->nik', A, B)


def batch_mahal3_gpu(X, mu, Sigma):
    """Batch Mahalanobis distance for 3D vectors.
    
    Args:
        X: (N, 3) array of points
        mu: (N, 3) or (3,) array of means
        Sigma: (N, 3, 3) or (3, 3) array of covariance matrices
    
    Returns:
        (N,) array of Mahalanobis distances
    """
    X = xp.asarray(X) if USE_CUPY else np.asarray(X)
    mu = xp.asarray(mu) if USE_CUPY else np.asarray(mu)
    Sigma = xp.asarray(Sigma) if USE_CUPY else np.asarray(Sigma)
    
    # Ensure proper broadcasting
    if mu.ndim == 1:
        mu = mu[None, :]
    if Sigma.ndim == 2:
        Sigma = Sigma[None, :, :]
    
    diff = X - mu  # (N, 3)
    
    if USE_CUPY:
        # Solve Sigma^{-1} * diff for each batch
        inv_sigma = batch_inv3_gpu(Sigma)  # (N, 3, 3)
        mahal = xp.einsum('ni,nij,nj->n', diff, inv_sigma, diff)
    else:
        mahal = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            inv_sigma = np.linalg.inv(Sigma[i])
            mahal[i] = diff[i] @ inv_sigma @ diff[i]
    
    return xp.sqrt(mahal) if USE_CUPY else np.sqrt(mahal)


class PoseTransform:
    """Convenience class for SE(3) operations (CPU only)."""

    @staticmethod
    def skew(v):
        # CPU zone: use NumPy for single SE(3) operations
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]], dtype=np.float64)

    @staticmethod
    def exp_se3(xi):
        # CPU zone: use NumPy for single SE(3) exponential
        rho, omega = xi[:3], xi[3:]
        theta = np.linalg.norm(omega)
        if theta < 1e-10:
            R = np.eye(3) + PoseTransform.skew(omega)
            V = np.eye(3) + 0.5 * PoseTransform.skew(omega)
        else:
            K = PoseTransform.skew(omega / theta)
            K2 = K @ K
            R = np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*K2
            V = np.eye(3) + ((1-math.cos(theta))/theta)*K + ((theta-math.sin(theta))/theta)*K2
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = V @ rho
        return T

    @staticmethod
    def log_se3(T):
        # CPU zone: use NumPy for single SE(3) logarithm
        R, t = T[:3, :3], T[:3, 3]
        cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(cos_a)
        if angle < 1e-10:
            return np.concatenate([t, np.zeros(3)])
        K = (angle/(2.0*math.sin(angle))) * (R - R.T)
        omega = np.array([K[2,1], K[0,2], K[1,0]])
        K_n = PoseTransform.skew(omega/angle)
        K2 = K_n @ K_n
        half = angle*math.sin(angle)/(2.0*(1.0-math.cos(angle)))
        V_inv = np.eye(3) - 0.5*PoseTransform.skew(omega) + (1.0/(angle*angle))*(1.0-half)*K2
        return np.concatenate([V_inv @ t, omega])

    @staticmethod
    def inverse(T):
        Ri = T[:3, :3].T
        Ti = np.eye(4)
        Ti[:3, :3] = Ri
        Ti[:3, 3] = -Ri @ T[:3, 3]
        return Ti

    @staticmethod
    def angular_distance(R1, R2):
        return math.acos(np.clip((np.trace(R1 @ R2.T) - 1.0) / 2.0, -1.0, 1.0))

    @staticmethod
    def transform_points(T, pts):
        return (T[:3, :3] @ pts.T).T + T[:3, 3]
