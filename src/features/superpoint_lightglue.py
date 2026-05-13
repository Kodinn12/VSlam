"""SuperPoint + LightGlue feature extraction and matching.

This module integrates LightGlue (lightweight SIFT-like global matcher) with
SuperPoint (deep-learned detector/descriptor) to provide fast, reliable feature
tracking suitable for SLAM applications.

GPU paths support both PyTorch and CuPy arrays, with zero-copy CuPy->PyTorch
tensor interchange via __cuda_array_interface__ to eliminate PCIe transfers on
every frame.
"""

import numpy as np
import cv2
import torch
from typing import Dict, Tuple, Optional
from ..utils.cupy_utils import USE_TORCH

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False

try:
    from lightglue import SuperPoint, LightGlue
except ImportError as e:
    print(f"[WARN] LightGlue import failed: {e}")
    print("[WARN] Attempting workaround... features will be stubbed")
    # Create stub classes to allow module loading
    class SuperPoint:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("SuperPoint not available. Install: pip install git+https://github.com/cvg/LightGlue.git")
    class LightGlue:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LightGlue not available. Install: pip install git+https://github.com/cvg/LightGlue.git")


class SuperPointLightGlue:
    """SuperPoint feature extractor + LightGlue matcher for SLAM tracking.
    
    Two extraction modes:
      • Normal-motion: NMS radius 4, threshold 0.002 - aggressive NMS, fewer keypoints
      • Fast-motion: NMS radius 2, threshold 0.001 - looser NMS, denser grid
    
    The fast-motion mode is triggered by high IMU angular velocity (documented
    feature: IMU NMS-radius-reduction).
    """
    
    def __init__(self, device: str = 'cuda', max_num_keypoints: Optional[int] = None):
        """Initialize SuperPoint + LightGlue on the given device.
        
        Parameters
        ----------
        device : str
            'cuda' or 'cpu'.  Falls back to CPU if CUDA unavailable.
        max_num_keypoints : int, optional
            Maximum keypoints to extract per image.  If None, uses LightGlue defaults.
        """
        # Check CUDA availability more robustly
        try:
            if device == 'cuda' and USE_TORCH and torch.cuda.is_available():
                # Test if CUDA actually works
                test_tensor = torch.randn(1, device='cuda')
                self.device = 'cuda'
                del test_tensor
            else:
                self.device = 'cpu'
        except Exception as e:
            print(f" [WARN] CUDA initialization failed: {e}, falling back to CPU")
            self.device = 'cpu'
        
        self.max_num_keypoints = max_num_keypoints
        print(f" [INIT] Loading SuperPoint + LightGlue on {self.device} ...")

        # Normal-motion extractor (nms_radius=4, detection_threshold=0.002)
        self.extractor = SuperPoint(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=0.002,
            nms_radius=4
        ).eval().to(self.device)

        # Fast-motion extractor - tighter NMS + lower threshold so more
        # keypoints survive motion blur.  Implements the documented
        # "NMS radius reduction" IMU feature.  Descriptors are from the
        # same network so LightGlue matching between any two extractions
        # (normal↔fast or fast↔normal) is valid.
        self.extractor_fast = SuperPoint(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=0.001,   # lower threshold -> more detections
            nms_radius=2                 # tighter NMS -> denser keypoint grid
        ).eval().to(self.device)

        self.matcher = LightGlue(
            features='superpoint',
            depth_confidence=0.9,
            width_confidence=0.95,
            filter_threshold=0.1
        ).eval().to(self.device)

        self._img_tensor      = None
        self._img_tensor_fast = None   # separate buffer to avoid shape conflicts

        if self.device == 'cuda' and USE_TORCH:
            # Configure CUDA for stability
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False  # Disable for stability
            torch.backends.cudnn.deterministic = False
            
            # Warm up CUDA to ensure all contexts are initialized
            try:
                _ = torch.nn.functional.linear(
                    torch.randn(1, 64, device='cuda'),
                    torch.randn(64, 64, device='cuda')
                )
                torch.cuda.synchronize()
            except Exception as _we:
                print(f" [WARN] CUDA warmup failed: {_we}")
            
            # OPT: torch.compile fuses kernels and eliminates overhead.
            # The default 'inductor' backend requires Triton, which is NOT
            # available on Windows.  We probe for Triton first; if present
            # we use 'reduce-overhead' (inductor), otherwise 'aot_eager'
            # which gives kernel fusion without Triton.
            _torch_major = int(torch.__version__.split('.')[0])
            if _torch_major >= 2:
                try:
                    import triton  # noqa: F401
                    _compile_backend = 'inductor'
                    _compile_mode    = 'reduce-overhead'
                except ImportError:
                    _compile_backend = 'aot_eager'  # Triton-free, Windows-safe
                    _compile_mode    = None
                try:
                    _ckw = dict(backend=_compile_backend)
                    if _compile_mode:
                        _ckw['mode'] = _compile_mode
                    # Temporarily disable torch.compile to avoid CUDA issues
                    # self.extractor      = torch.compile(self.extractor,      **_ckw)
                    # self.extractor_fast = torch.compile(self.extractor_fast, **_ckw)
                    # self.matcher        = torch.compile(self.matcher,        **_ckw)
                    print(f" [OPTIM] torch.compile disabled to avoid CUDA issues")
                except Exception as _ce:
                    print(f" [OPTIM] torch.compile skipped: {_ce}")
        print(f" [✓] Models loaded.  Max keypoints: {max_num_keypoints or 'Unlimited'}"
              f"  (normal NMS=4, fast-motion NMS=2)")

    @torch.no_grad()
    def extract(self, gray, fast_motion: bool = False) -> Dict:
        """Extract SuperPoint features.

        Parameters
        ----------
        gray : np.ndarray (uint8 HxW CPU)  OR  cp.ndarray (uint8/float32 HxW GPU)
            When a CuPy GPU array is supplied the image stays on-device:
            uint8->float32 conversion and normalisation happen on the GPU,
            and torch.as_tensor(__cuda_array_interface__) gives a zero-copy
            PyTorch view - no PCIe H2D transfer at all.

        fast_motion : bool
            When True, uses the fast-motion extractor (nms_radius=2,
            detection_threshold=0.001) which produces a denser grid of
            keypoints that better survives motion blur.  Implements the
            documented IMU NMS-radius-reduction feature.

        Returns
        -------
        dict
            'keypoints': (N, 2) torch.float32 on device
            'descriptors': (N, 256) torch.float32 on device
            'keypoint_scores': (N,) torch.float32 on device
            'image_size': (2,) torch.int64 on device

        Notes
        -----
        V43: CuPy zero-copy path - eliminates H2D of gray image (~144 KB,
        ~0.1-0.3 ms) on every frame.  Benchmark:
          old: torch.from_numpy(cpu_arr).to('cuda')  -> PCIe H2D each call
          new: torch.as_tensor(cp_arr)               -> zero-copy, GPU-side
        """
        # ── Determine shape regardless of input type ─────────────────────
        h, w = gray.shape[0], gray.shape[1]

        # ── V43: CuPy GPU input - zero-copy path ─────────────────────────
        if USE_CUPY and isinstance(gray, cp.ndarray):
            # GPU-side uint8->float32 + normalise; both ops are cheap device kernels.
            # torch.as_tensor() creates a ZERO-COPY view via __cuda_array_interface__
            # into the CuPy buffer - no PCIe transfer, no allocation.
            if gray.dtype != cp.float32:
                gray_f32 = gray.astype(cp.float32)
                if gray.dtype == cp.uint8:
                    gray_f32 = gray_f32 * (1.0 / 255.0)
            else:
                gray_f32 = gray
            # Ensure device compatibility - if self.device is CPU, we need to convert CuPy to CPU first
            if self.device == 'cpu':
                # Convert CuPy array to CPU numpy, then to PyTorch CPU tensor
                img_view = torch.as_tensor(cp.asnumpy(gray_f32), device=self.device)   # (H, W)
            else:
                # Use zero-copy CuPy -> PyTorch
                img_view = torch.as_tensor(gray_f32, device=self.device)   # (H, W)
            
            # Allocate persistent (1,1,H,W) buffer once and copy from view
            buf = self._img_tensor_fast if fast_motion else self._img_tensor
            if buf is None or buf.shape[-2:] != (h, w):
                buf = torch.empty(
                    (1, 1, h, w), dtype=torch.float32, device=self.device)
                if fast_motion:
                    self._img_tensor_fast = buf
                else:
                    self._img_tensor = buf
            buf[0, 0].copy_(img_view, non_blocking=True)  # GPU->GPU or CPU->CPU
            extractor = self.extractor_fast if fast_motion else self.extractor
            feats = extractor({'image': buf})
            return {
                'keypoints':       feats['keypoints'][0],
                'descriptors':     feats['descriptors'][0],
                'keypoint_scores': feats['keypoint_scores'][0],
                'image_size':      torch.tensor([h, w], device=self.device)
            }

        # ── CPU numpy path (fallback when CuPy unavailable or CPU array) ─
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        if fast_motion:
            # Use the dedicated fast-motion extractor
            if (self._img_tensor_fast is None or
                    self._img_tensor_fast.shape[-2:] != gray.shape):
                self._img_tensor_fast = torch.empty(
                    (1, 1, gray.shape[0], gray.shape[1]),
                    dtype=torch.float32, device=self.device)
            self._img_tensor_fast[0, 0].copy_(
                torch.from_numpy(gray).to(self.device, non_blocking=True) / 255.0)
            feats = self.extractor_fast({'image': self._img_tensor_fast})
        else:
            if self._img_tensor is None or self._img_tensor.shape[-2:] != gray.shape:
                self._img_tensor = torch.empty(
                    (1, 1, gray.shape[0], gray.shape[1]),
                    dtype=torch.float32, device=self.device)
            self._img_tensor[0, 0].copy_(
                torch.from_numpy(gray).to(self.device, non_blocking=True) / 255.0)
            feats = self.extractor({'image': self._img_tensor})
        return {
            'keypoints':       feats['keypoints'][0],
            'descriptors':     feats['descriptors'][0],
            'keypoint_scores': feats['keypoint_scores'][0],
            'image_size':      torch.tensor(
                [gray.shape[0], gray.shape[1]], device=self.device)
        }

    @torch.no_grad()
    def match(self, feats0: Dict, feats1: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match features between two frames using LightGlue.
        
        Parameters
        ----------
        feats0 : dict
            Features from first frame (output of extract())
        feats1 : dict
            Features from second frame (output of extract())
        
        Returns
        -------
        matches : (M,) torch.int64
            Indices of matched keypoints in feats0 (or -1 for no match)
        scores : (M,) torch.float32
            Confidence scores for each match
        """
        data = {
            'image0': {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                       for k, v in feats0.items()},
            'image1': {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                       for k, v in feats1.items()}
        }
        
        # Ensure CUDA context is alive before matcher call
        if self.device == 'cuda':
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        
        try:
            pred = self.matcher(data)
        except RuntimeError as e:
            if 'CUBLAS' in str(e) or 'CUDA' in str(e):
                # Recover from CUDA context loss by reinitializing
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    _ = torch.randn(1, 1, device='cuda')
                    torch.cuda.synchronize()
                # Retry once
                pred = self.matcher(data)
            else:
                raise
        
        # OPT: Removed torch.cuda.synchronize() - was blocking CPU until GPU
        # finished every single frame, stalling the entire pipeline by
        # 1-3 ms/frame.  The results are accessed on the CPU via .cpu().numpy()
        # downstream which implicitly synchronizes at that point only.
        return pred['matches0'][0], pred['matching_scores0'][0]
