"""Multi-scale SGM pyramid for improved disparity estimation."""

import numpy as np
import cv2
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False


class MultiScaleSGM:
    """Multi-scale SGM using image pyramid for improved disparity estimation."""
    
    def __init__(self, num_scales: int = 3, scale_factor: float = 0.5):
        """
        Initialize multi-scale SGM.
        
        Parameters
        ----------
        num_scales : int
            Number of pyramid scales
        scale_factor : float
            Scale factor between pyramid levels
        """
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        logger.info(f"MultiScaleSGM initialized: scales={num_scales}, factor={scale_factor}")
    
    def build_pyramid_cpu(self, image: np.ndarray) -> list:
        """
        Build image pyramid on CPU.
        
        Parameters
        ----------
        image : np.ndarray
            Input grayscale image (H, W), uint8
        
        Returns
        -------
        list
            List of pyramid images from coarse to fine
        """
        pyramid = [image]
        for _ in range(self.num_scales - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid
    
    def build_pyramid_gpu(self, image: np.ndarray) -> list:
        """
        Build image pyramid on GPU.
        
        Parameters
        ----------
        image : np.ndarray
            Input grayscale image on GPU (H, W), uint8
        
        Returns
        -------
        list
            List of pyramid images on GPU from coarse to fine
        """
        if not USE_CUPY:
            logger.warning("CuPy not available, falling back to CPU")
            return self.build_pyramid_cpu(image)
        
        pyramid = [cp.asarray(image, dtype=cp.uint8)]
        for _ in range(self.num_scales - 1):
            current = pyramid[-1]
            H, W = current.shape
            new_H, new_W = int(H * self.scale_factor), int(W * self.scale_factor)
            # Simple downsampling using strided access
            downsampled = current[::2, ::2]
            pyramid.append(downsampled)
        return pyramid
    
    def upsample_disparity_cpu(self, disparity: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Upsample disparity map to target resolution on CPU.
        
        Parameters
        ----------
        disparity : np.ndarray
        Input disparity map (H, W), float32
        target_shape : tuple
        Target shape (H, W)
        
        Returns
        -------
        np.ndarray
        Upsampled disparity map (H, W), float32
        """
        return cv2.resize(disparity, (target_shape[1], target_shape[0]), 
                        interpolation=cv2.INTER_LINEAR)
    
    def upsample_disparity_gpu(self, disparity: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Upsample disparity map to target resolution on GPU.
        
        Parameters
        ----------
        disparity : np.ndarray
        Input disparity map on GPU (H, W), float32
        target_shape : tuple
        Target shape (H, W)
        
        Returns
        -------
        np.ndarray
        Upsampled disparity map on GPU (H, W), float32
        """
        if not USE_CUPY:
            logger.warning("CuPy not available, falling back to CPU")
            return self.upsample_disparity_cpu(disparity, target_shape)
        
        H, W = target_shape
        # Simple bilinear upsampling using CuPy
        disp_gpu = cp.asarray(disparity, dtype=cp.float32)
        upsampled = cp.zeros((H, W), dtype=cp.float32)
        
        # CuPy RawKernel for bilinear upsampling
        upsample_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void upsample_kernel(
            const float* __restrict__ disparity,
            float* __restrict__ upsampled,
            const int H_in, const int W_in,
            const int H_out, const int W_out
        ) {
            const int i = blockIdx.y * blockDim.y + threadIdx.y;
            const int j = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (i >= H_out || j >= W_out) return;
            
            const float x = (j + 0.5f) * ((float)W_in / W_out) - 0.5f;
            const float y = (i + 0.5f) * ((float)H_in / H_out) - 0.5f;
            
            const int x0 = (int)floorf(x);
            const int y0 = (int)floorf(y);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            
            const float fx = x - x0;
            const float fy = y - y0;
            
            float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
            
            if (y0 >= 0 && y0 < H_in && x0 >= 0 && x0 < W_in)
                v00 = disparity[y0 * W_in + x0];
            if (y0 >= 0 && y0 < H_in && x1 >= 0 && x1 < W_in)
                v01 = disparity[y0 * W_in + x1];
            if (y1 >= 0 && y1 < H_in && x0 >= 0 && x0 < W_in)
                v10 = disparity[y1 * W_in + x0];
            if (y1 >= 0 && y1 < H_in && x1 >= 0 && x1 < W_in)
                v11 = disparity[y1 * W_in + x1];
            
            upsampled[i * W_out + j] = 
                (1.0f - fx) * (1.0f - fy) * v00 +
                fx * (1.0f - fy) * v01 +
                (1.0f - fx) * fy * v10 +
                fx * fy * v11;
        }
        ''', 'upsample_kernel')
        
        H_in, W_in = disparity.shape
        block_size = (16, 16)
        grid_size = ((W + block_size[0] - 1) // block_size[0],
                    (H + block_size[1] - 1) // block_size[1])
        
        upsample_kernel(
            grid_size, block_size,
            (disp_gpu, upsampled, H_in, W_in, H, W)
        )
        
        return upsampled
