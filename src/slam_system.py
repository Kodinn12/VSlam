from .camera.oakd_manager import OakDStereoManager
from .camera.depth_source_selector import DepthSourceSelector
from .stereo.stereo_engine import StereoEngine
from .imu.preintegrator import IMUPreIntegrator
from .features.superpoint_lightglue import SuperPointLightGlue
from .tracking.particle_filter import SE3ParticleFilter
from .tracking.pose_refiner import LMPoseRefiner
from .tracking.relocalizer import GhostParticleRelocalizer
from .mapping.gaussian_bubbles import ChunkedBubbleMap
from .mapping.tsdf_voxel import ThreadedCupyVoxelManager
from .mapping.keyframe import Keyframe
from .mapping.keyframe_manager import KeyframeManager
from .optimization.pose_graph_optimizer import PoseGraphOptimizer
from .optimization.bundle_adjuster import CuPyBundleAdjuster
from .loop_closure.loop_detector import LoopClosureDetector
from .visualisation.pyvista_viz import PyVistaVisualizer, USE_PYVISTA
from .utils.logger import get_logger
from .utils.se3_ops import PoseTransform
from .utils.pnp import _batched_gpu_pnp_ransac
from .utils.gpu_acceleration import GPUAccelerationManager
from .utils.array_utils import to_numpy_safe
from .utils.depth_utils import bilinear_depth_gpu
from .core.profiling import Profiler
from .core.config import coerce_config_types
# from .dataset.dataset_generator import DatasetGenerator  # DISABLED - User requested no dataset saving
import numpy as np
import logging
import cv2
import torch
import os
import time
import copy
import threading
import math
from typing import Optional, Dict, List, Tuple
from enum import Enum

from .utils.cupy_utils import cupy_manager, USE_TORCH, cp

# Lazy accessor for the array module (xp) to prevent import-time GPU initialization
class _LazyXP:
    def __getattr__(self, name):
        # This will trigger CuPyManager initialization on first use
        return getattr(cupy_manager.get_array_module(), name)
    
    def __call__(self, *args, **kwargs):
        return cupy_manager.get_array_module()(*args, **kwargs)

xp = _LazyXP()
USE_CUPY = cupy_manager.is_available()

try:
    if USE_CUPY:
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()
    else:
        mempool = None
        pinned_mempool = None
except Exception:
    mempool = None
    pinned_mempool = None

try:
    from .utils.depth_utils import _get_bld_kernel
except ImportError:
    def _get_bld_kernel():
        pass

try:
    import kornia
    _HAS_KORNIA = True
except ImportError:
    _HAS_KORNIA = False

try:
    from .visualisation.pyvista_viz import PyVistaVisualizer, USE_PYVISTA
except ImportError:
    PyVistaVisualizer = None
    USE_PYVISTA = False

# Open3D removed - using PyVista only
USE_OPEN3D = False

logger = get_logger(__name__)

class TrackingState(Enum):
    UNINITIALIZED = 0
    TRACKING = 1
    RELOCALIZING = 2
    LOST = 3
    RECOVERY = 4
    FAILED = 5

class _AsyncOptWorker:
    """Async optimization worker for BA, PGO, and loop closure to prevent frame blocking."""
    
    def __init__(self, config):
        config = coerce_config_types(config)
        self.config = config
        self.K = config.get("K", np.eye(3, dtype=np.float64))
        self.ba_queue   = []
        self.pgo_queue  = []
        self.loop_queue = []
        self.generic_queue = []  # Added for generic job submission
        self._lock = threading.Lock()
        
        self._shutdown_event = threading.Event()
        
        self._ba_thread   = threading.Thread(target=self._ba_worker,   daemon=True)
        self._pgo_thread  = threading.Thread(target=self._pgo_worker,  daemon=True)
        self._loop_thread = threading.Thread(target=self._loop_worker, daemon=True)
        # We'll use the PGO thread or a new one for generic jobs. 
        # Let's keep it simple and use the PGO thread for PGO/BA generic jobs.
        self._ba_thread.start()
        self._pgo_thread.start()
        self._loop_thread.start()
        self._latest_corrections = None
        self._corrections_available = False
        self._latest_loop_result = None
        self._loop_result_available = False
        
        # Profiler for async workers
        self.profiler = Profiler(enabled=config.get('enable_profiling', False))
    
    def _ba_worker(self):
        """Bundle adjustment worker thread."""
        while not self._shutdown_event.is_set():
            try:
                with self._lock:
                    if self.ba_queue:
                        task = self.ba_queue.pop(0)
                    else:
                        task = None
                
                if task:
                    # Profile bundle adjustment
                    self.profiler.start('bundle_adjustment')
                    
                    if callable(task):
                        # Support for generic job functions
                        try:
                            result = task()
                            if result is not None:
                                with self._lock:
                                    self._latest_corrections = result
                                    self._corrections_available = True
                        except Exception as e:
                            print(f"[AsyncOptWorker] BA job error: {e}")
                    else:
                        # Traditional task tuple
                        keyframes, poses = task
                        # Placeholder for actual BA processing
                        corrections = self._process_bundle_adjustment(keyframes, poses)
                        
                        with self._lock:
                            self._latest_corrections = corrections
                            self._corrections_available = True
                    
                    self.profiler.end('bundle_adjustment')
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[AsyncOptWorker] BA worker error: {e}")
                time.sleep(0.01)
    
    def _pgo_worker(self):
        """Pose graph optimization worker thread."""
        while not self._shutdown_event.is_set():
            try:
                with self._lock:
                    if self.pgo_queue:
                        task = self.pgo_queue.pop(0)
                    else:
                        task = None
                
                if task:
                    # Profile pose graph optimization
                    self.profiler.start('pose_graph_optimization')
                    
                    if callable(task):
                        # Support for generic job functions
                        try:
                            result = task()
                            if result is not None:
                                with self._lock:
                                    self._latest_corrections = result
                                    self._corrections_available = True
                        except Exception as e:
                            print(f"[AsyncOptWorker] PGO job error: {e}")
                    else:
                        # Traditional task tuple
                        keyframes, loop_edges = task
                        # Placeholder for actual PGO processing
                        corrections = self._process_pose_graph_optimization(keyframes, loop_edges)
                        
                        with self._lock:
                            self._latest_corrections = corrections
                            self._corrections_available = True
                    
                    self.profiler.end('pose_graph_optimization')
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[AsyncOptWorker] PGO worker error: {e}")
                time.sleep(0.01)
    
    def _loop_worker(self):
        """Loop closure detection worker thread."""
        while not self._shutdown_event.is_set():
            try:
                with self._lock:
                    if self.loop_queue:
                        task = self.loop_queue.pop(0)
                    else:
                        task = None
                
                if task:
                    # Profile loop closure detection
                    self.profiler.start('loop_closure')
                    
                    if len(task) == 3 and callable(task[0]):
                        # Support for custom detection function call
                        detector_func, current_kf, candidate_kfs = task
                        try:
                            success, loop_edge = detector_func(current_kf, candidate_kfs)
                        except Exception as e:
                            print(f"[AsyncOptWorker] Loop job error: {e}")
                            success, loop_edge = False, None
                    else:
                        # Traditional task tuple
                        current_kf, candidate_kfs = task
                        # Placeholder for actual loop closure processing
                        success, loop_edge = self._process_loop_closure(current_kf, candidate_kfs)
                    
                    self.profiler.end('loop_closure')
                    
                    with self._lock:
                        self._latest_loop_result = (success, loop_edge)
                        self._loop_result_available = True
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[AsyncOptWorker] Loop worker error: {e}")
                time.sleep(0.01)
    
    def _process_bundle_adjustment(self, keyframes, poses):
        """Process bundle adjustment with actual optimization."""
        try:
            if len(keyframes) < 2:
                return None
            
            # Create bundle adjuster instance
            ba = CuPyBundleAdjuster(self.K, self.config)
            
            # Create a dummy SuperPointLightGlue for cross-KF matching
            # In a real implementation, this would be passed from the main SLAM system
            class DummySP:
                def match(self, feats1, feats2):
                    return np.array([]), np.array([])
            
            sp_lg = DummySP()
            
            # Run bundle adjustment
            corrected_poses, converged = ba.run(keyframes, sp_lg)
            
            if converged:
                corr_map = {}
                for i, kf in enumerate(keyframes):
                    corr_map[kf.id] = corrected_poses[i]
                return corr_map, True
            else:
                return None, False
            
        except Exception as e:
            print(f"[AsyncOptWorker] BA processing error: {e}")
            return None
    
    def _process_pose_graph_optimization(self, keyframes, loop_edges):
        """Process pose graph optimization with actual optimization."""
        try:
            if len(keyframes) < 2 or len(loop_edges) == 0:
                return None
            
            # Create pose graph optimizer instance
            pgo = PoseGraphOptimizer(self.config)
            
            # Run pose graph optimization
            success = pgo.optimize(keyframes)
            
            if success:
                corr_map = {}
                for kf in keyframes:
                    corr_map[kf.id] = kf.pose
                return corr_map, True
            else:
                return None, False
            
        except Exception as e:
            print(f"[AsyncOptWorker] PGO processing error: {e}")
            return None
    
    def _process_loop_closure(self, current_kf, candidate_kfs):
        """Process loop closure detection with actual detection."""
        try:
            if not candidate_kfs or len(candidate_kfs) == 0:
                return False, None
            
            # Create loop closure detector instance
            # Note: LoopClosureDetector requires K_proc and sp_lg parameters
            # In a real implementation, these would be passed from the main SLAM system
            K_proc = np.eye(3)  # Placeholder camera matrix
            class DummySP:
                def match(self, feats1, feats2):
                    return np.array([]), np.array([])
            
            sp_lg = DummySP()
            detector = LoopClosureDetector(self.config, K_proc, sp_lg)
            
            # Run loop closure detection
            success, loop_edge = detector.detect_and_verify(current_kf, candidate_kfs)
            
            return success, loop_edge
            
        except Exception as e:
            print(f"[AsyncOptWorker] Loop closure processing error: {e}")
            return False, None
    
    def submit_bundle_adjustment(self, keyframes, poses):
        """Submit bundle adjustment task for async processing."""
        with self._lock:
            self.ba_queue.append((keyframes, poses))
    
    def submit_pose_graph_optimization(self, keyframes, loop_edges):
        """Submit pose graph optimization task for async processing."""
        with self._lock:
            self.pgo_queue.append((keyframes, loop_edges))
    
    def submit_loop_closure(self, detector_func, current_kf, candidate_kfs):
        """Submit loop closure detection task for async processing."""
        with self._lock:
            # Check if previous loop task is still in queue
            if len(self.loop_queue) > 0:
                return False
            self.loop_queue.append((detector_func, current_kf, candidate_kfs))
            return True

    def submit(self, job_func):
        """Generic submission method for closure-based jobs."""
        with self._lock:
            # Route to appropriate queue based on job name if possible, 
            # or just use PGO queue as a general purpose one.
            name = job_func.__name__ if hasattr(job_func, "__name__") else ""
            if "ba" in name.lower():
                if len(self.ba_queue) > 0: return False
                self.ba_queue.append(job_func)
            else:
                if len(self.pgo_queue) > 0: return False
                self.pgo_queue.append(job_func)
            return True
    
    def take_corrections(self):
        """Return latest corrections if available."""
        with self._lock:
            if self._corrections_available:
                self._corrections_available = False
                return self._latest_corrections  # This is already (corr_map, do_reintegrate)
            return None, False
    
    def take_loop_closure_result(self):
        """Return latest loop closure result if available."""
        with self._lock:
            if self._loop_result_available:
                self._loop_result_available = False
                return self._latest_loop_result
            return False, None
    
    def shutdown(self):
        """Shutdown all worker threads gracefully."""
        self._shutdown_event.set()
        self._ba_thread.join(timeout=1.0)
        self._pgo_thread.join(timeout=1.0)
        self._loop_thread.join(timeout=1.0)

class _StubFeatureExtractor:
    """Stub feature extractor when LightGlue is unavailable."""
    
    def extract(self, image, fast_motion=False):
        """Return empty features."""
        return {
            'keypoints': np.array([], dtype=np.float32).reshape(0, 2),
            'descriptors': np.array([], dtype=np.float32).reshape(0, 256),
            'scores': np.array([], dtype=np.float32),
            'keypoint_scores': np.array([], dtype=np.float32)
        }
    
    def match(self, feats1, feats2):
        """Return empty matches."""
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

class RobustStereoSLAM:
    def __init__(self, config):
        config = coerce_config_types(config)
        self.config = config
        self.proc_size = (640, 400)
        self.frame_id = 0
        self.consecutive_failures = 0
        self.state = TrackingState.UNINITIALIZED
        
        # ----------------------------------------------------------------
        # GPU ACCELERATION MANAGER
        # ----------------------------------------------------------------
        self.gpu_manager = GPUAccelerationManager(config)
        self.accel_config = self.gpu_manager.get_acceleration_config()
        print(f" [GPU] Acceleration mode: {config.get('acceleration_mode', 'unknown')}")
        print(f" [GPU] CuPy: {self.accel_config['use_cupy']}, PyTorch: {self.accel_config['use_torch']}")

        # ----------------------------------------------------------------
        # OAK-D CAMERA + STEREO DEPTH  (OakDStereoManager)
        #   Reads EEPROM calibration, builds StereoDepth pipeline,
        #   exposes P1_rect / baseline_m / f_b_term / R_cam_imu.
        #   Supports dual-mode: hardware depth OR raw L/R for custom SGM
        # ----------------------------------------------------------------
        print(" [OAK] Opening OAK-D and reading EEPROM factory calibration …")
        
        # Depth source selection
        depth_source = config.get("depth_source", "oakd_hardware")
        self.depth_source_selector = DepthSourceSelector(config)
        
        self.cam = OakDStereoManager({
            "camera_fps":       config.get("camera_fps", 60),
            "depth_resolution": self.proc_size,
            "enable_imu":       config.get("enable_imu", True),
            "depth_source":     depth_source,
        })
        self.cam.start()
        
        # Initialize StereoEngine if using custom SGM or Ultimate Stereo
        self.stereo_engine = None
        self.ultimate_stereo = None
        
        current_source = self.depth_source_selector.get_source()
        if current_source == 'custom_sgm':
            print(" [STEREO] Initializing custom SGM stereo engine")
            stereo_mode = 'full_gpu' if self.accel_config['use_cupy'] else 'cpu_only'
            self.stereo_engine = StereoEngine(config, acceleration_mode=stereo_mode)
        elif current_source == 'ultimate_stereo':
            print(" [STEREO] Initializing Ultimate Zero-Noise Stereo Processor")
            from .camera.ultimate_stereo import UltimateStereoProcessor
            self.ultimate_stereo = UltimateStereoProcessor(self.cam, config)

        # ----------------------------------------------------------------
        # IMU PRE-INTEGRATOR
        # R_cam_imu (from EEPROM) rotates IMU measurements to left-cam frame.
        # Degrades gracefully to visual-only if IMU queue is unavailable.
        # ----------------------------------------------------------------
        self.imu = None
        if self.cam._imu_queue is not None:
            self.imu = IMUPreIntegrator(
                self.cam._imu_queue,
                R_cam_imu=self.cam.R_cam_imu)
            print(f" [IMU] IMU pre-integrator attached  "
                  f"(R_cam_imu det={np.linalg.det(self.cam.R_cam_imu):.4f})")
        else:
            print(" [WARN] No IMU queue available - IMU features disabled")

        # ----------------------------------------------------------------
        # INTRINSICS  -  derived from EEPROM stereoRectify result (P1_rect)
        #   P1_rect is at proc resolution (640×400).
        #   No .npz calibration file used or required.
        # ----------------------------------------------------------------
        self.K_rect_proc = self.cam.P1_rect[:3, :3].copy()
        self.baseline     = self.cam.baseline_m
        self._intrinsic   = self.K_rect_proc
        self.fx           = float(self.K_rect_proc[0, 0])
        self.fy           = float(self.K_rect_proc[1, 1])
        self.cx           = float(self.K_rect_proc[0, 2])
        self.cy           = float(self.K_rect_proc[1, 2])
        print(f" [CAL] K_rect_proc  fx={self.fx:.2f}  fy={self.fy:.2f}  "
              f"cx={self.cx:.2f}  cy={self.cy:.2f}")

        # ----------------------------------------------------------------
        # LM REFINER
        # ----------------------------------------------------------------
        self.lm_refiner = LMPoseRefiner(
            K=self.K_rect_proc,
            max_iter   =int(config.get("lm_max_iterations", 20)),
            lam0       =float(config.get("lm_initial_lambda", 1e-3)),
            conv_delta =float(config.get("lm_convergence_delta", 1e-7)),
            huber_thresh=float(config.get("huber_threshold_reproj", 2.0)),
            use_gpu    =config.get("lm_use_gpu", True))

        # ----------------------------------------------------------------
        # PARTICLE FILTER
        # ----------------------------------------------------------------
        self.pf = SE3ParticleFilter(
            K               =self.K_rect_proc,
            num_particles   =config.get("pf_num_particles", 80),
            sigma_t_base    =config.get("pf_sigma_t_base", 0.012),
            sigma_r_base    =config.get("pf_sigma_r_base", 0.015),
            sigma_t_min     =config.get("pf_sigma_t_min", 0.003),
            sigma_r_min     =config.get("pf_sigma_r_min", 0.004),
            obs_sigma       =config.get("pf_obs_sigma", 3.5),
            resample_thresh =config.get("pf_resample_threshold", 0.5),
            karcher_iters   =config.get("pf_karcher_iters", 6),
            confidence_scale=config.get("pf_confidence_scale", 40.0),
            huber_thresh    =config.get("huber_threshold_reproj", 2.0))

        # ----------------------------------------------------------------
        # SUPERPOINT + LIGHTGLUE FEATURE EXTRACTOR
        # ----------------------------------------------------------------
        try:
            device = 'cuda' if self.accel_config['use_torch'] else 'cpu'
            self.sp_lg = SuperPointLightGlue(device=device)
            print(" [FEAT] SuperPoint+LightGlue initialized")
        except Exception as e:
            print(f" [WARN] SuperPoint+LightGlue failed: {e}")
            # Create stub implementation
            self.sp_lg = _StubFeatureExtractor()
            print(" [FEAT] Using stub feature extractor")

        # ----------------------------------------------------------------
        # GAUSSIAN BUBBLE MAP (GPU Accelerated)
        # ----------------------------------------------------------------
        # Update config with GPU acceleration settings
        gpu_config = config.copy()
        gpu_config.update({
            'bubble_cuda': self.accel_config['gpu_bubbles'],
            'use_cupy_voxel_grid': self.accel_config['gpu_tsdf']
        })
        
        self.bubble_map = ChunkedBubbleMap(
            K=self.K_rect_proc, baseline=self.baseline, config=gpu_config)

        # ----------------------------------------------------------------
        # MULTI-VIEW KEYFRAME MANAGER (True SLAM Reconstruction)
        # ----------------------------------------------------------------
        self.keyframe_manager = KeyframeManager(config)
        print(" [KF] Multi-view keyframe manager initialized")
        
        # Set keyframe manager in bubble map for multi-view fusion
        # self.bubble_map.threaded_manager.keyframe_manager = self.keyframe_manager

        # ----------------------------------------------------------------
        # CUPY VOXEL MANAGER + LOOP DETECTOR + RELOCALIZER (GPU Accelerated)
        # ----------------------------------------------------------------
        # V59: Disable TSDF Voxel Grid to save 500MB+ VRAM on 6GB GPUs
        gpu_config["enable_tsdf_voxels"] = False
        self.voxel_manager = ThreadedCupyVoxelManager(
            gpu_config, self.K_rect_proc, self.baseline)
        
        accel_backend = "CuPy" if cupy_manager.is_available() else ("TorchXP" if USE_TORCH else "NumPy")
        print(f" [GPU] Voxel manager initialized using {accel_backend} backend (TSDF=OFF)")
        self.loop_detector = LoopClosureDetector(
            config, self.K_rect_proc, self.sp_lg)
        self.relocalizer = GhostParticleRelocalizer(
            config, self.K_rect_proc, self.voxel_manager, self.bubble_map)

        self.relocalizer_trigger_threshold = int(config.get("relocalizer_trigger_failures", 15))
        self.relocalizer_max_attempts      = int(config.get("relocalizer_max_attempts", 12))
        self.relocalizer_attempts          = 0

        # ----------------------------------------------------------------
        # GLOBAL POSE GRAPH OPTIMIZER  (CuPy Gauss-Newton, SE(3))
        # ----------------------------------------------------------------
        self.pgo = PoseGraphOptimizer(config)
        self._pgo_enabled = config.get("enable_global_optimization", True)
        self._pgo_kf_count = 0   # keyframe count for sequential edge tracking

        # ----------------------------------------------------------------
        # WINDOWED BUNDLE ADJUSTER  (CuPy Schur-complement LM)
        # ----------------------------------------------------------------
        self.bundle_adjuster = CuPyBundleAdjuster(self.K_rect_proc, config)
        self._ba_enabled  = config.get("enable_bundle_adjustment", True)
        self._ba_kf_count = 0    # keyframe counter to trigger BA
        self._ba_frequency = config.get("ba_frequency", 8)
        print(f" [V11] PGO: {'ON' if self._pgo_enabled else 'OFF'}  "
              f"BA: {'ON' if self._ba_enabled else 'OFF'}  "
              f"(BA every {self._ba_frequency} keyframes, "
              f"window={config.get('ba_window_size',8)} KFs)")

        # ----------------------------------------------------------------
        # V37: ASYNC BA/PGO WORKER
        # ----------------------------------------------------------------
        # Runs BA and PGO in a background daemon thread so process_frame
        # never stalls waiting for optimization to complete.
        # Corrections are picked up at the start of the next _handle_keyframe.
        self._async_opt = _AsyncOptWorker(config)

        # Initialize tracking variables to avoid UnboundLocalError
        tracked = False; num_inliers = 0

        # ----------------------------------------------------------------
        # POSE STATE
        # ----------------------------------------------------------------
        self.curr_pose          = np.eye(4)
        self.last_keyframe_pose = np.eye(4)
        self.prev_kf_feats      = None
        self.prev_kf_pose       = None
        self.prev_kf_depth      = None
        self.depth_fps_buffer   = []
        self.timing_buffer      = []

        
        # PYVISTA VISUALIZER
        self.visualizer = None
        if config.get("enable_pyvista_visualization", True) and USE_PYVISTA:
            try:
                self.visualizer = PyVistaVisualizer(
                    window_size=(1280,720),
                    title="SLAM 3D  -  OAK-D StereoDepth",
                    max_points=config.get("max_visualization_points", 50000),
                    point_size=config.get("visualization_point_size", 2.0))
            except Exception as e:
                logger.error(f" [Viz] Failed to start visualizer: {e}")
                self.visualizer = None
        
        # ----------------------------------------------------------------
        # DATASET GENERATOR (HDF5)
        # ----------------------------------------------------------------
        self.dataset = None  # DISABLED - User requested no dataset saving
        # Dataset generation completely disabled per user request
        if config.get("enable_dataset_generation", False):  # Force False
            print(" [DATA] Dataset generation disabled per user request")

        # PyVista only - no Open3D fallback
        if self.visualizer is None:
            print("[Viz] PyVista not available, 3D visualization disabled")
            self.visualizer = None

        # ----------------------------------------------------------------
        # FRAME-TO-FRAME tracking fallback
        # Tracking against the previous FRAME (not just last keyframe)
        # prevents losing track during fast motion between keyframes.
        # ----------------------------------------------------------------
        self.prev_frame_feats = None
        self.prev_frame_pose  = None
        self.prev_frame_depth = None
        # V9: GPU-resident previous depth - avoids H2D in bilinear_depth_gpu()
        self.prev_frame_depth_gpu: Optional[object] = None   # xp.ndarray or None

        # Warmup: suppress relocalizer trigger for the first N frames so
        # that the depth pipeline and feature extractor have time to stabilise.
        self._WARMUP_FRAMES = 60

        # ----------------------------------------------------------------
        # IMU TRACKING STATE
        # ----------------------------------------------------------------
        # Accumulated IMU delta for current frame (fetched in process_frame)
        self._imu_delta_R   = np.eye(3, dtype=np.float64)
        self._imu_delta_p   = np.zeros(3, dtype=np.float64)
        self._imu_omega_mag = 0.0
        self._imu_dt        = 0.0
        # Latest gravity vector in camera frame - updated every frame from IMU.
        # Used by: gravity-aided PF constraint, ZUPT, tilt detection.
        self._imu_grav      = np.array([0.0, 9.81, 0.0], dtype=np.float64)

        # Previous frame's inlier count - used to scale PF noise on the
        # NEXT frame's prediction step (avoids using zero for predict noise).
        self._prev_num_inliers = 0

        # IMU holdover: if visual tracking fails, use IMU to propagate pose
        self._imu_holdover_frames = 0   # consecutive frames held over by IMU
        self._imu_holdover_dt     = 0.0 # accumulated dt during holdover
        # cv2.createCLAHE() is a non-trivial allocation.  Creating it inside
        # the per-frame hot-path adds ~0.5 ms per fast-motion frame and wastes
        # memory.  Cache once at init and reuse every frame.
        _clip  = config.get("imu_clahe_clip_limit", 2.0)
        _tgrid = tuple(config.get("imu_clahe_tile_grid", (8, 8)))
        self._clahe = cv2.createCLAHE(clipLimit=_clip, tileGridSize=_tgrid)

        # V43: GPU CLAHE - eliminates ~0.5-1 ms CPU CLAHE on every fast-motion
        # frame.  Tries cv2.cuda.createCLAHE first (full tile-based CLAHE on
        # GPU, requires OpenCV built with CUDA).  Falls back to CuPy global
        # histogram equalization if cv2.cuda is unavailable (fast approximation,
        # no CPU involvement).  Final fallback: existing CPU CLAHE above.
        self._use_gpu_clahe    = False   # cv2.cuda path
        self._use_cupy_heq     = False   # CuPy histogram-eq fallback
        self._clahe_gpu_obj    = None    # cv2.cuda CLAHE object
        if USE_CUPY:
            # ── Attempt 1: cv2.cuda.createCLAHE ─────────────────────────
            try:
                _cg = cv2.cuda.createCLAHE(clipLimit=_clip, tileGridSize=_tgrid)
                # Smoke-test: apply to a tiny dummy mat to force JIT-compile now
                _dummy_gpu = cv2.cuda_GpuMat(
                    np.zeros((8, 8), dtype=np.uint8))
                _result = cv2.cuda_GpuMat()
                _cg.apply(_dummy_gpu, _result, cv2.cuda_Stream_Null())
                self._clahe_gpu_obj = _cg
                self._use_gpu_clahe = True
                print(" [V43] GPU CLAHE ready  (cv2.cuda.createCLAHE)")
            except Exception as _cce:
                # ── Attempt 2: CuPy global histogram equalization ─────────
                # Not tile-limited like CLAHE, but fully GPU-resident and
                # provides contrast enhancement with zero CPU involvement.
                # V43 NOTE: this is a global HEQ approximation, not true CLAHE.
                # It is used ONLY when cv2.cuda is unavailable on this system.
                try:
                    _test_g = xp.zeros((8, 8), dtype=xp.uint8)
                    _hist, _ = xp.histogram(_test_g.ravel(), bins=256,
                                            range=(0, 255))
                    self._use_cupy_heq = True
                    print(f" [V43] cv2.cuda CLAHE unavailable ({_cce}), "
                          f"using CuPy histogram-equalization GPU fallback")
                except Exception as _he:
                    print(f" [V43] GPU CLAHE/HEQ both unavailable - CPU CLAHE"
                          f" ({_he})")
        # Last depth GPU array - retained for the HUD depth visualisation
        self._last_depth_gpu = None

        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if USE_CUPY:
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

        # V9: CuPy memory pool warm-up - pre-allocate common tensor shapes
        # so the first live frame finds hot allocator entries instead of calling
        # cudaMalloc (~0.5 ms each).  Shapes: particles, Jacobian, bubble scratch.
        if USE_CUPY:
            _N = config.get("pf_num_particles", 100)
            _pw, _ph = self.proc_size
            _warmup_shapes = [
                (_N, 4, 4),                    # particles (float64)
                (_N,),                         # weights (float64)
                (2500 * 2, 6),                 # LM Jacobian rows
                (2500, 3),                     # pts3d GPU buffer
                (config.get("voxel_grid_size",
                 (400, 400, 400))[0] * 2, 3), # voxel subset scratch
                (_ph * _pw,),                  # depth flat (proc resolution)
            ]
            _warmup_bufs = []
            for sh in _warmup_shapes:
                try:
                    _warmup_bufs.append(xp.zeros(sh, dtype=xp.float32))
                except Exception:
                    pass
            del _warmup_bufs   # release immediately; allocator keeps the pages

        # FIX BUG 3 (V9): Pre-compile the bilinear depth RawKernel NOW so
        # frame 1 does not pay the ~200 ms JIT spike.  The lazy _get_bld_kernel()
        # call was deferred to the first estimate_pose() call, causing a massive
        # FPS drop on exactly the first tracking frame.
        if USE_CUPY:
            try:
                _get_bld_kernel()   # triggers NVRTC compile; cached for all future calls
                print(" [V9] Bilinear depth CUDA kernel pre-compiled")
            except Exception as _ke:
                print(f" [V9] Bilinear kernel compile warning: {_ke}")
        print(" [V9] CuPy mempool warm-up complete")

        print(" [SLAM] RobustStereoSLAM (OAK-D StereoDepth) initialization complete!")
        
        # Load previous reconstruction if enabled
        if self.config.get("enable_reconstruction_persistence", True) and \
           self.config.get("load_previous_reconstruction", False):
            self.load_reconstruction_state()

    # ----------------------------------------------------------------
    # V43: GPU CLAHE helper - called from process_frame hot-path
    # ----------------------------------------------------------------
    def _apply_clahe_gpu(self, gray_gpu: 'xp.ndarray') -> 'xp.ndarray':
        """
        Apply contrast enhancement to a uint8 CuPy gray image entirely on GPU.

        Priority:
          1. cv2.cuda CLAHE   (full tile-based, requires OpenCV+CUDA)
          2. CuPy global HEQ  (approximate, no tiles, but 100% GPU-resident)
          3. CPU CLAHE        (fallback, 0.5-1 ms; result uploaded back to GPU)

        Returns a uint8 CuPy array of the same shape.
        """
        if self._use_gpu_clahe and self._clahe_gpu_obj is not None:
            # ── cv2.cuda path: CuPy -> GpuMat (zero-copy via __cuda_array_interface__)
            # cv2.cuda_GpuMat.createFromArr uses the same device memory, no copy.
            try:
                # Convert CuPy -> cv2.cuda_GpuMat (zero-copy)
                gpu_mat_in = cv2.cuda_GpuMat(to_numpy_safe(gray_gpu))   # minimal D2H needed here
                # NOTE: cv2.cuda_GpuMat.createFromArr is the true zero-copy path but
                # isn't available in all OpenCV-Python builds.  We use a pragmatic approach:
                # upload via cv2.cuda_GpuMat() which handles the H2D itself on its CUDA stream.
                # This still avoids the CPU CLAHE computation and leverages GPU tiles.
                gpu_mat_out = cv2.cuda_GpuMat()
                self._clahe_gpu_obj.apply(gpu_mat_in, gpu_mat_out,
                                           cv2.cuda_Stream_Null())
                result_np = gpu_mat_out.download()
                return xp.asarray(result_np)
            except Exception:
                pass   # fall through to CuPy HEQ

        if self._use_cupy_heq:
            # ── CuPy global histogram equalization (fully on-device) ──────
            # global HEQ maps pixel intensities so the cumulative histogram
            # is uniform - effectively maximises contrast over the full image.
            gray_flat = gray_gpu.ravel().astype(xp.float32)
            hist, _ = xp.histogram(gray_flat, bins=256, range=(0.0, 255.0))
            cdf = xp.cumsum(hist).astype(xp.float32)
            cdf_min = float(cdf[cdf > 0][0])
            n = float(gray_gpu.size)
            lut = xp.clip(
                xp.round((cdf - cdf_min) / (n - cdf_min + 1e-9) * 255.0),
                0, 255).astype(xp.uint8)
            return lut[gray_gpu.astype(xp.uint8)]

        # ── CPU fallback: apply CPU CLAHE, convert result back to GPU ────
        result_cpu = self._clahe.apply(xp.asnumpy(gray_gpu))
        return xp.asarray(result_cpu)

    # ----------------------------------------------------------------
    def estimate_pose(self, feats_curr, feats_prev, depth_prev, pose_prev,
                      T_hint: Optional[np.ndarray] = None,
                      depth_prev_gpu=None):
        """
        Estimate current pose from feature matches + depth.

        Parameters
        ----------
        T_hint : Optional 4×4 pose - IMU-predicted pose passed as an
                 extrinsic initial guess to solvePnPRansac (rvec/tvec init).
                 When provided, RANSAC converges faster and more reliably
                 during fast motion where features spread widely.
        depth_prev_gpu : Optional xp.ndarray - GPU version of depth_prev.
                         V9: used by bilinear_depth_gpu() to avoid H2D upload.
        """
        if feats_prev is None or depth_prev is None:
            return False, pose_prev, 0, None, None

        matches, scores = self.sp_lg.match(feats_curr, feats_prev)
        valid_mask = (matches > -1) & (scores > self.config['match_threshold'])
        valid_idx  = torch.where(valid_mask)[0]
        if len(valid_idx) < self.config['min_matches_tracking']:
            return False, pose_prev, 0, None, None

        # V11: index keypoints on-device - eliminates 2 of 4 D2H transfers.
        # V44: kp_p/kp_c D2H deferred into the GPU/CPU branches below so the
        # GPU path never hits .cpu().numpy() on the critical path.
        pm_t = matches[valid_idx].long()    # GPU index tensor, no D2H yet

        # V40: keep depth sampling GPU-side, build pts_cam on GPU, avoid H2D until PnP
        _depth_src = depth_prev_gpu if depth_prev_gpu is not None else depth_prev
        if USE_CUPY and isinstance(_depth_src, xp.ndarray):
            # ── Keep kp_p/kp_c as PyTorch tensors on GPU (zero-copy) ────────
            # V44: avoid .cpu().numpy() for kp_p - keep on CUDA for GPU PnP.
            kp_p_t  = feats_prev['keypoints'][pm_t]          # torch, GPU
            kp_c_t  = feats_curr['keypoints'][valid_idx]     # torch, GPU
            # Zero-copy CuPy view via __cuda_array_interface__ for depth sampling
            kp_p_g  = xp.asarray(kp_p_t.contiguous())        # CuPy view, no copy
            z_g     = bilinear_depth_gpu(_depth_src, kp_p_g[:, 0], kp_p_g[:, 1],
                                         return_gpu=True)
            valid_z_g = (z_g > 0.05) & (z_g < 20.0)
            n_valid   = int(xp.sum(valid_z_g))
            if n_valid < self.config['lm_min_inliers']:
                return False, pose_prev, 0, None, None
            # Build pts_world on GPU - CuPy, no D2H
            u_v_g = kp_p_g[valid_z_g, 0]; v_v_g = kp_p_g[valid_z_g, 1]
            z_v_g = z_g[valid_z_g]
            pts_cam_g = xp.column_stack(
                [(u_v_g - self.cx) * z_v_g / self.fx,
                 (v_v_g - self.cy) * z_v_g / self.fy,
                 z_v_g])
            pose_g       = xp.asarray(pose_prev, dtype=xp.float64)
            pts_world_g  = (pose_g[:3, :3] @ pts_cam_g.T).T + pose_g[:3, 3]  # CuPy, no D2H
            # curr_2d: stay on GPU as torch tensor using valid_z_g mask
            # valid_z_g is CuPy bool; convert mask indices via torch for indexing
            # CuPy bool arrays may not be contiguous; ascontiguousarray + named
            # ref guarantees a valid __cuda_array_interface__ for torch.
            _vz_contig  = xp.ascontiguousarray(valid_z_g)
            valid_z_t   = torch.as_tensor(_vz_contig, device='cuda')   # zero-copy
            curr_2d_t = kp_c_t[valid_z_t]                           # torch, GPU
            # D2H only for the legacy CPU-only fallback path below; kept for
            # the numpy pts_world/curr_2d needed by LMPoseRefiner.
            pts_world   = xp.asnumpy(pts_world_g)      # single D2H - needed for LM refiner
            curr_2d     = curr_2d_t.cpu().numpy()       # single D2H - needed for LM refiner
            valid_z     = xp.asnumpy(valid_z_g)
        else:
            kp_p = feats_prev['keypoints'][pm_t].cpu().numpy()
            kp_c = feats_curr['keypoints'][valid_idx].cpu().numpy()
            curr_2d_t   = None
            pts_world_g = None
            z       = bilinear_depth_gpu(_depth_src, kp_p[:, 0], kp_p[:, 1])
            valid_z = (z > 0.05) & (z < 20.0)
            if np.sum(valid_z) < self.config['lm_min_inliers']:
                return False, pose_prev, 0, None, None
            u_v  = kp_p[valid_z, 0]; v_v = kp_p[valid_z, 1]; z_v = z[valid_z]
            pts_cam   = np.column_stack(
                [(u_v - self.cx) * z_v / self.fx,
                 (v_v - self.cy) * z_v / self.fy, z_v])
            pts_world = PoseTransform.transform_points(pose_prev, pts_cam)
            curr_2d   = kp_c[valid_z]
            
            # Numerical stability: filter any non-finite world points (V45)
            finite_mask = np.all(np.isfinite(pts_world), axis=1)
            if not np.all(finite_mask):
                pts_world = pts_world[finite_mask]
                curr_2d = curr_2d[finite_mask]

        # re-alias for PnP call below
        if not isinstance(pts_world, np.ndarray):
            pts_world = np.asarray(pts_world)

        if len(pts_world) < self.config['lm_min_inliers']:
            return False, pose_prev, 0, None, None

        # ── IMU-warm-started PnP initial guess ────────────────────────
        rvec_init = None; tvec_init = None
        use_hint  = False
        T_cw_hint_t = None   # torch tensor for GPU PnP path
        if T_hint is not None:
            try:
                T_cw_hint = PoseTransform.inverse(T_hint)
                rvec_init, _ = cv2.Rodrigues(T_cw_hint[:3, :3])
                tvec_init    = T_cw_hint[:3, 3].reshape(3, 1)
                use_hint     = True
                if _HAS_KORNIA:
                    T_cw_hint_t = torch.as_tensor(
                        T_cw_hint, dtype=torch.float32, device='cuda')
            except Exception:
                rvec_init = None; tvec_init = None; use_hint = False

        # ── V44 Step 1: GPU batched PnP RANSAC (Kornia) ───────────────────
        # Zero-copy: pts_world_g and curr_2d_t are already on CUDA.
        # Falls back to OpenCV CPU PnP when Kornia is unavailable.
        pnp_flags = cv2.SOLVEPNP_EPNP
        T_cw_g    = None   # set by GPU PnP path; None signals CPU path to caller
        if _HAS_KORNIA and pts_world_g is not None and curr_2d_t is not None:
            # Convert CuPy pts_world_g to float32 torch tensor (zero-copy view).
            # Keep a named reference to the float32 CuPy array so CPython's
            # reference counter does NOT free the GPU buffer before the tensor
            # is consumed by _batched_gpu_pnp_ransac.  device='cuda' ensures
            # PyTorch honours __cuda_array_interface__ and never copies to host.
            _pw_f32     = pts_world_g.astype(xp.float32)     # named ref - keeps memory alive
            pts_world_t = torch.as_tensor(_pw_f32, device='cuda')
            curr_2d_f   = curr_2d_t.float()
            K_t         = torch.as_tensor(
                self.K_rect_proc[:3, :3].astype(np.float32), device='cuda')
            n_iter_gpu  = 300 if not use_hint else 250
            ok_g, T_cw_g, inl_mask_g, n_inl_g = _batched_gpu_pnp_ransac(
                pts_world_t, curr_2d_f, K_t,
                reproj_thresh=self.config['tracking_inlier_threshold'],
                n_iter=n_iter_gpu,
                T_cw_hint=T_cw_hint_t)
            if ok_g and n_inl_g >= self.config['lm_min_inliers']:
                ok      = ok_g
                inliers = np.where(inl_mask_g)[0].reshape(-1, 1)
                # T_cw_g (4×4 np.float64) is used directly by the T_pnp_wc
                # builder below; no need to materialise rvec/tvec on the GPU path.
            else:
                # GPU PnP solved but found too few inliers (degenerate scene,
                # motion-blur, etc.).  Fall back to OpenCV CPU PnP so we don't
                # lose tracking on edge-case frames - same robustness as V43.
                ok = False; inliers = None
                if use_hint:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        pts_world.astype(np.float64),
                        curr_2d.astype(np.float64).reshape(-1, 2),
                        self.K_rect_proc.astype(np.float64), None,
                        rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True,
                        reprojectionError=self.config['tracking_inlier_threshold'],
                        iterationsCount=250,
                        flags=cv2.SOLVEPNP_SQPNP
                              if hasattr(cv2, 'SOLVEPNP_SQPNP')
                              else cv2.SOLVEPNP_ITERATIVE)
                else:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        pts_world.astype(np.float64),
                        curr_2d.astype(np.float64).reshape(-1, 2),
                        self.K_rect_proc.astype(np.float64), None,
                        reprojectionError=self.config['tracking_inlier_threshold'],
                        iterationsCount=300, flags=pnp_flags)
                # If CPU fallback succeeded, mark T_cw_g as None so the
                # T_pnp_wc builder below takes the CPU (Rodrigues) path.
                T_cw_g = None
        else:
            # ── OpenCV CPU PnP: Kornia not installed ───────────────────
            T_cw_g = None
            if use_hint:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_world.astype(np.float64),
                    curr_2d.astype(np.float64).reshape(-1, 2),
                    self.K_rect_proc.astype(np.float64), None,
                    rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True,
                    reprojectionError=self.config['tracking_inlier_threshold'],
                    iterationsCount=250,
                    flags=cv2.SOLVEPNP_SQPNP
                          if hasattr(cv2, 'SOLVEPNP_SQPNP')
                          else cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_world.astype(np.float64),
                    curr_2d.astype(np.float64).reshape(-1, 2),
                    self.K_rect_proc.astype(np.float64), None,
                    reprojectionError=self.config['tracking_inlier_threshold'],
                    iterationsCount=300, flags=pnp_flags)

        if not ok or inliers is None or len(inliers) < self.config['lm_min_inliers']:
            return False, pose_prev, 0, pts_world, curr_2d

        # ── Build T_pnp_wc from the solver output ─────────────────────────
        # GPU path (_batched_gpu_pnp_ransac succeeded): T_cw_g is a 4×4 np.float64,
        #   signalled by T_cw_g is not None.
        # CPU path (cv2.solvePnPRansac, either fallback or no-Kornia):
        #   T_cw_g is None -> use rvec/tvec from Rodrigues.
        # Both paths end up with T_pnp_wc (world-from-camera) for the LM refiner.
        if T_cw_g is not None and ok:
            # GPU path: T_cw_g was built inside _batched_gpu_pnp_ransac
            T_pnp    = T_cw_g                    # 4×4 np.float64 camera-from-world
            ifl      = inliers.flatten()
            T_pnp_wc = PoseTransform.inverse(T_pnp)
            # GPU PnP already did inlier refinement; skip cv2.solvePnPRefineLM
            # (Kornia DLT on inlier set IS the refinement pass)
        else:
            R_pnp, _ = cv2.Rodrigues(rvec)
            T_pnp    = np.eye(4); T_pnp[:3,:3]=R_pnp; T_pnp[:3,3]=tvec.flatten()
            ifl      = inliers.flatten()
            # FIX A: OpenCV solvePnP returns T_cw (camera-from-world).
            # LMPoseRefiner.refine() expects T_wc (world-from-camera) because it
            # internally calls inverse(T) to project world points into the camera.
            # Pass inverse(T_pnp) = T_wc so that projection is correct and the
            # LM Jacobian converges on the right pose.  The returned T_refined is
            # also T_wc, consistent with curr_pose / PF particles convention.
            T_pnp_wc  = PoseTransform.inverse(T_pnp)

            # V6: post-RANSAC solvePnPRefineLM - optional 2nd-pass nonlinear
            # refinement of the RANSAC hypothesis on the inlier set BEFORE passing
            # to LMPoseRefiner.  Improves the initial guess by ~0.3-0.8 px which
            # typically reduces LM iterations needed for convergence.
            if len(ifl) >= self.config['lm_min_inliers']:
                try:
                    inl_pts3 = pts_world[ifl].astype(np.float64)
                    inl_pts2 = curr_2d[ifl].astype(np.float64).reshape(-1, 2)
                    rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                        inl_pts3, inl_pts2,
                        self.K_rect_proc.astype(np.float64), None,
                        rvec.copy(), tvec.copy())
                    R_ref, _ = cv2.Rodrigues(rvec_ref)
                    T_pnp_ref = np.eye(4)
                    T_pnp_ref[:3,:3] = R_ref; T_pnp_ref[:3,3] = tvec_ref.flatten()
                    T_pnp_wc = PoseTransform.inverse(T_pnp_ref)
                except Exception:
                    pass   # fall back to original RANSAC result
        # OPT V5: adaptive Huber threshold - tighten as inlier count grows.
        # When we have many inliers the PnP solution is reliable; a tighter
        # Huber threshold pushes the final refinement toward sub-pixel accuracy.
        # With few inliers we stay loose to avoid excluding valid matches.
        n_inl = len(inliers)
        if n_inl >= 40:
            self.lm_refiner.huber_c = 1.0
        elif n_inl >= 20:
            self.lm_refiner.huber_c = 1.2
        else:
            self.lm_refiner.huber_c = self.config.get('huber_threshold_reproj', 1.5)
        T_refined = self.lm_refiner.refine(T_pnp_wc, pts_world[ifl], curr_2d[ifl])
        return True, T_refined, len(inliers), pts_world, curr_2d

    # ----------------------------------------------------------------
    def process_frame(self):
        """Main SLAM processing loop.
        
        IMU integration steps:
          1. Drain IMU delta since last frame.
          2. Compose T_imu_pred = curr_pose @ T_imu_delta.
          3. Use T_imu_pred as mean in SE3ParticleFilter.predict_with_imu().
          4. Pass T_imu_pred as hint to estimate_pose() for warm-started PnP.
          5. Validate visual pose jump against IMU-predicted jump.
          6. During visual tracking failure: propagate pose via IMU holdover.
          7. Adapt keyframe rotation threshold based on gyro angular rate.

        Returns: current camera pose (4x4 numpy)
        """
        frame_start = time.time()
        
        # Initialize tracking variables to avoid UnboundLocalError
        tracked = False; num_inliers = 0
        
        # Debug: Track frame processing
        if self.frame_id % 20 == 0:
            print(f" [DEBUG] Processing frame {self.frame_id}")
        # ── (1) FETCH IMU DELTA ─────────────────────────────────────────
        # Drain all IMU samples accumulated since the previous camera frame.
        # This is always non-blocking; if IMU is unavailable returns identity.
        if self.imu is not None:
            (self._imu_delta_R,
             self._imu_delta_p,
             self._imu_omega_mag,
             self._imu_dt,
             self._imu_grav) = self.imu.get_delta(reset=True)
        else:
            self._imu_delta_R   = np.eye(3, dtype=np.float64)
            self._imu_delta_p   = np.zeros(3, dtype=np.float64)
            self._imu_omega_mag = 0.0
            self._imu_dt        = 1.0 / 60.0   # assume 60 Hz camera
            # _imu_grav keeps last valid value (or default [0,g,0])

        # ── (2) IMU-PREDICTED POSE ──────────────────────────────────────
        # Compose current pose with IMU delta to get predicted next pose.
        # Used as: PF prediction mean, PnP warm-start, jump validator.
        T_imu_delta        = np.eye(4, dtype=np.float64)
        T_imu_delta[:3,:3] = self._imu_delta_R
        T_imu_delta[:3, 3] = self._imu_delta_p
        T_imu_pred         = self.curr_pose @ T_imu_delta   # world frame

        # ── Get stereo frames from OAK-D ────────────────────────────────────
        # get_frames() is non-blocking (tryGet on both queues).
        # Returns:
        #   - oakd_hardware mode: (left_rect_gray uint8, depth_gpu CuPy float32 metres)
        #   - custom_sgm mode: (left_gray uint8, right_gray uint8)
        depth_start = time.time()
        
        if self.depth_source_selector.is_hardware_depth():
            # Mode A: Use OAK-D hardware depth
            gray_rect, depth_gpu = self.cam.get_frames()
            if gray_rect is None or depth_gpu is None:
                return self.curr_pose
        elif self.depth_source_selector.get_source() == "ultimate_stereo":
            # Mode C: Use Ultimate Zero-Noise Stereo
            left_gray, right_gray = self.cam.get_frames()
            if left_gray is None or right_gray is None:
                return self.curr_pose
            
            # Compute depth using Ultimate Stereo Processor
            gray_rect, depth_gpu = self.ultimate_stereo.process(left_gray, right_gray)
        else:
            # Mode B: Use custom SGM
            left_gray, right_gray = self.cam.get_frames()
            if left_gray is None or right_gray is None:
                return self.curr_pose
            
            # Compute depth using custom SGM
            # V45: IMU-driven motion score for temporal smoothing
            motion_score = float(self._imu_omega_mag) if hasattr(self, '_imu_omega_mag') else 0.0
            depth = self.stereo_engine.compute_depth(
                left_gray, right_gray,
                self.cam.baseline_m, self.fx,
                motion_score=motion_score
            )
            # Ensure depth_gpu is CuPy array when USE_CUPY, NumPy otherwise
            if USE_CUPY and not isinstance(depth, xp.ndarray):
                depth_gpu = xp.asarray(depth)
            else:
                depth_gpu = depth
            gray_rect = left_gray
        
        self.frame_id += 1

        # CPU depth array (needed by estimate_pose bilinear sampler)
        depth = to_numpy_safe(depth_gpu) if USE_CUPY else depth_gpu

        depth_time = time.time() - depth_start
        depth_fps  = 1.0 / max(depth_time, 1e-6)
        self.depth_fps_buffer.append(depth_fps)
        self._last_depth_gpu = depth_gpu

        intrinsic = self._intrinsic

        # GPU image pipeline - upload uint8 gray ONCE (144 KB),
        # derive BGR on GPU (zero CPU cvtColor, zero extra H2D of 432 KB).
        if USE_CUPY:
            gray_rect_gpu = xp.asarray(gray_rect)                        # 144 KB H2D only
            left_img_rect_gpu = xp.stack(
                [gray_rect_gpu, gray_rect_gpu, gray_rect_gpu], axis=-1)  # (H,W,3)
            left_img_rect = cv2.cvtColor(gray_rect, cv2.COLOR_GRAY2BGR)
        else:
            gray_rect_gpu     = None
            left_img_rect     = cv2.cvtColor(gray_rect, cv2.COLOR_GRAY2BGR)
            left_img_rect_gpu = None

        # ── Feature extraction ───────────────────────────────────────────
        use_fast_extractor = False
        if (self.imu is not None and
                self._imu_omega_mag > self.config.get("imu_clahe_omega_thresh", 0.40)):
            use_fast_extractor = True
            if USE_CUPY and gray_rect_gpu is not None:
                feat_img_gpu = self._apply_clahe_gpu(gray_rect_gpu)
                feats = self.sp_lg.extract(feat_img_gpu, fast_motion=True)
            else:
                feat_img_cpu = self._clahe.apply(gray_rect)
                feats = self.sp_lg.extract(feat_img_cpu, fast_motion=True)
        else:
            if USE_CUPY and gray_rect_gpu is not None:
                feats = self.sp_lg.extract(gray_rect_gpu, fast_motion=False)
            else:
                feats = self.sp_lg.extract(gray_rect, fast_motion=False)

        # ── IMU-omega bubble uncertainty scale (NEW) ──────────────────────
        # Inflate bubble sigma_par/sigma_per proportionally to angular rate
        # so that depth measurements taken during fast rotation have higher
        # uncertainty and do not over-commit the bubble map.
        _bub_omega_thr = self.config.get("imu_bubble_omega_thresh", 0.25)
        _bub_omega_ref = self.config.get("imu_bubble_omega_ref", 1.0)
        bubble_motion_scale = 1.0
        if self.imu is not None and self._imu_omega_mag > _bub_omega_thr:
            bubble_motion_scale = max(1.0,
                1.0 + (self._imu_omega_mag - _bub_omega_thr) / _bub_omega_ref)

        # ── IMU config shortcuts ─────────────────────────────────────────
        has_imu        = (self.imu is not None)
        imu_pf_w       = self.config.get("imu_pf_weight", 0.7)
        imu_jtol       = self.config.get("imu_jump_tolerance_factor", 2.5)
        imu_hold_max_f = self.config.get("imu_holdover_max_frames", 10)
        imu_hold_max_t = self.config.get("imu_holdover_max_dt", 0.20)

        # V51: Stabilization-first mapping gating. 
        # Force integration during initialization to ensure map bootstraps.
        is_initializing = (self.frame_id < 100)
        should_update_bubble = (self.frame_id % 1 == 0) # Force every frame
        
        # Skip map updates when IMU confirms the camera is stationary, 
        # but only after initial stabilization phase.
        _imu_static = has_imu and self.imu.is_stationary()
        
        if _imu_static and not is_initializing:
            should_update_bubble = False
            
        should_update_voxel = (not _imu_static) or is_initializing or (self.frame_id == 1)

        # ────────────────────────────────────────────────────────────────
        # STATE MACHINE
        # ────────────────────────────────────────────────────────────────
        is_kf = False
        if self.state == TrackingState.TRACKING:
            tracked = False; num_inliers = 0

            # Debug logging
            if self.frame_id % 10 == 0:
                print(f" [DEBUG] Frame {self.frame_id}: should_update_bubble={should_update_bubble}, "
                      f"imu_static={_imu_static}, tracked={tracked}")

            if self.frame_id > 1:
                # ── (3) IMU-AIDED PARTICLE FILTER PREDICTION ─────────────
                # ONE predict per frame - use previous frame's inlier count
                # so noise sigma is properly calibrated from the start of the
                # frame (avoids always using zero-confidence = max noise).
                if has_imu and self._imu_dt > 0:
                    self.pf.predict_with_imu(
                        self._imu_delta_R, self._imu_delta_p,
                        num_inliers=self._prev_num_inliers,
                        imu_weight=imu_pf_w)
                else:
                    self.pf.predict(self._prev_num_inliers)

                # ── Try FRAME-TO-FRAME first ──────────────────────────────
                ftf_success = False
                if self.prev_frame_feats is not None:
                    ftf_success, T_obs, num_inliers, pts3d_w, pts2d = \
                        self.estimate_pose(
                            feats, self.prev_frame_feats,
                            self.prev_frame_depth, self.prev_frame_pose,
                            T_hint=T_imu_pred if has_imu else None,   # (4) PnP hint
                            depth_prev_gpu=self.prev_frame_depth_gpu)  # V9: GPU bilinear

                # ── Fall back to KEYFRAME-TO-FRAME ────────────────────────
                if not ftf_success and self.prev_kf_feats is not None:
                    success_kf, T_obs_kf, ni_kf, pts3d_kf, pts2d_kf = \
                        self.estimate_pose(
                            feats, self.prev_kf_feats,
                            self.prev_kf_depth, self.prev_kf_pose,
                            T_hint=T_imu_pred if has_imu else None)  # (4) PnP hint
                    if success_kf:
                        T_obs, num_inliers = T_obs_kf, ni_kf
                        pts3d_w, pts2d    = pts3d_kf, pts2d_kf
                        ftf_success       = True

                success = ftf_success

                # ── (5) IMU POSE-JUMP VALIDATION ─────────────────────────
                # Accept a visual estimate only if it agrees with IMU within
                # imu_jump_tolerance_factor x the IMU-predicted jump.
                # This rejects wild PnP outliers during fast motion or blur.
                pose_jump = False
                if success:
                    vis_trans = np.linalg.norm(T_obs[:3,3] - self.curr_pose[:3,3])
                    vis_rot   = PoseTransform.angular_distance(
                        T_obs[:3,:3], self.curr_pose[:3,:3])

                    if has_imu and self._imu_dt > 0:
                        # IMU-predicted motion magnitudes
                        imu_trans = np.linalg.norm(self._imu_delta_p)
                        imu_rot   = math.acos(np.clip(
                            (np.trace(self._imu_delta_R) - 1.0) / 2.0, -1.0, 1.0))
                        # Max allowed: config hard limit OR imu_jtol x IMU estimate
                        max_trans = max(self.config["max_pose_jump"],
                                        imu_jtol * max(imu_trans, 0.01))
                        max_rot   = max(self.config["max_rotation_jump"],
                                        imu_jtol * max(imu_rot, 0.01))
                        if vis_trans > max_trans or vis_rot > max_rot:
                            pose_jump = True
                    else:
                        # Original hard-limit validation
                        if (vis_trans > self.config["max_pose_jump"] or
                                vis_rot > self.config["max_rotation_jump"]):
                            pose_jump = True

                # ── Update PF with refined particles ─────────────────────
                # (No second predict here - one predict per frame is correct.
                #  The noise sigma is now governed by the PREVIOUS frame's
                #  inlier count set above, not re-predicted after PnP.)
                if success and not pose_jump:
                    # V11: inject observed pose directly into GPU mirror -
                    # avoids np.argmin + CPU array write that left _parts_gpu stale.
                    if USE_CUPY:
                        wi_g = int(xp.argmin(self.pf._weights_gpu))
                        self.pf._parts_gpu[wi_g] = xp.asarray(T_obs, dtype=xp.float64)
                        med_w = float(xp.median(self.pf._weights_gpu))
                        self.pf._weights_gpu[wi_g] = med_w
                        ws_g = float(self.pf._weights_gpu.sum())
                        if ws_g > 1e-12:
                            self.pf._weights_gpu /= ws_g
                        # CPU mirror synced lazily at next resample/karcher
                        self.pf.weights = xp.asnumpy(self.pf._weights_gpu)
                    else:
                        wi = np.argmin(self.pf.weights)
                        self.pf.particles[wi]  = T_obs.copy()
                        self.pf.weights[wi]    = np.median(self.pf.weights)
                        self.pf.weights       /= self.pf.weights.sum()

                    if pts3d_w is not None and len(pts3d_w) >= 4:
                        MAX_PTS = 80
                        if len(pts3d_w) > MAX_PTS:
                            # V11: torch.topk stays on GPU - no D2H download of all scores
                            sc_t = feats['keypoint_scores']
                            N_sc = min(len(sc_t), len(pts3d_w))
                            if N_sc >= MAX_PTS:
                                tk = torch.topk(sc_t[:N_sc], MAX_PTS).indices.cpu().numpy()
                            else:
                                tk = np.arange(N_sc)
                            pts3d_w_pf = pts3d_w[tk]; pts2d_pf = pts2d[tk]
                        else:
                            pts3d_w_pf = pts3d_w; pts2d_pf = pts2d
                        self.pf.update(pts3d_w_pf, pts2d_pf)

                    # ── (9) GRAVITY-AIDED PF ROLL/PITCH CONSTRAINT ───────────
                    # After visual update, apply IMU gravity direction as a soft
                    # roll/pitch tether.  Particles with wrong tilt angle are
                    # down-weighted.  Provides absolute orientation accuracy even
                    # when visual features are sparse or motion is fast.
                    #
                    # CORRECTED: pass the IMU's world-frame gravity estimate as
                    # the absolute reference.  Previously the PF computed the
                    # reference from the particle MEAN, making it self-referential
                    # and weaker - particles were penalised for disagreeing with
                    # each other rather than for disagreeing with the IMU.
                    if has_imu and np.linalg.norm(self._imu_grav) > 1.0:
                        # V6: smooth exponential gravity sigma - replaces 4-tier
                        # discrete if-elif (which had discontinuous jumps at tier
                        # boundaries and missed the sub-0.10 rad/s ultra-tight case).
                        # Formula: sigma = 0.03 + 0.27*(1 - exp(-omega/0.3))
                        #   omega=0.00 -> sigma≈0.030 rad (ultra-tight, ~1.7°)
                        #   omega=0.05 -> sigma≈0.072 rad
                        #   omega=0.10 -> sigma≈0.106 rad
                        #   omega=0.20 -> sigma≈0.163 rad
                        #   omega=0.50 -> sigma≈0.233 rad
                        #   omega=1.00 -> sigma≈0.281 rad (effectively unweighted)
                        grav_sigma = 0.03 + 0.27 * (1.0 - math.exp(
                            -self._imu_omega_mag / 0.3))
                        # V6: adaptive PF obs_sigma - tightens when many inliers.
                        # More inliers = better-conditioned PnP = reliable obs.
                        # Tight likelihood sharpens the posterior estimate.
                        inlier_scale = self.config.get("pf_adaptive_obs_scale", 0.03)
                        self.pf.obs_sigma = max(
                            self.pf.obs_sigma_base * 0.5,
                            self.pf.obs_sigma_base / (
                                1.0 + inlier_scale * num_inliers /
                                      max(self.config.get("pf_confidence_scale", 40.0), 1.0)))
                        # Use absolute IMU world-gravity if it has been
                        # bootstrapped (non-zero norm); fall back to camera-
                        # frame gravity (slightly weaker but still valid).
                        imu_grav_world = self.imu.get_gravity_world()
                        self.pf.update_with_gravity_constraint(
                            self._imu_grav,
                            gravity_sigma=grav_sigma,
                            gravity_world=imu_grav_world
                                if np.linalg.norm(imu_grav_world) > 1.0
                                else None)

                    # ── (10) ZUPT - Zero-Velocity Update ─────────────────────
                    # When IMU confirms the device is stationary, collapse the
                    # particle cloud to its current mean and zero out velocity.
                    # Prevents drift accumulation during static operation.
                    #
                    # BUG 1 FIX: alpha was 0.80, leaving 20% residual velocity.
                    #   blended = (1-α)*v + α*zeros = 0.2*v ≠ zero!
                    #   ZUPT means zero velocity -> alpha must be 1.0.
                    # BUG 2 FIX: visual velocity correction (below) was running
                    #   unconditionally AFTER ZUPT, partially re-inflating
                    #   the velocity just zeroed.  _zupt_fired flag skips it.
                    _zupt_fired = False
                    if has_imu and self.imu.is_stationary():
                        self.pf.apply_zupt()
                        self.imu.correct_velocity(
                            np.zeros(3, dtype=np.float64), alpha=1.0)  # FIX: was 0.80
                        _zupt_fired = True

                    self.curr_pose = self.pf.estimate()
                    
                    # Divergence detection
                    if not np.all(np.isfinite(self.curr_pose)):
                        print(f" [WARN] Tracking diverged (non-finite pose) at frame {self.frame_id}")
                        self.state = TrackingState.RELOCALIZING
                        self.curr_pose = self.prev_frame_pose.copy() if self.prev_frame_pose is not None else np.eye(4)
                        tracked = False
                    else:
                        tracked        = True
                        self.consecutive_failures = 0
                        self._imu_holdover_frames = 0
                        self._imu_holdover_dt     = 0.0
                        # ── Save inlier count for next frame's PF noise scaling ──
                        self._prev_num_inliers    = num_inliers

                    # ── (8) VISUAL SLAM -> IMU VELOCITY CORRECTION ────────────
                    # Use the ground-truth visual pose change to correct the
                    # IMU's open-loop velocity estimate each frame.
                    # Alpha is dynamic: higher during fast motion (more visual
                    # correction needed to prevent IMU drift), lower during
                    # slow motion (IMU is reliable, preserve momentum).
                    # BUG 2 FIX: skip entirely when ZUPT has fired this frame -
                    # ZUPT already set velocity to zero; running visual correction
                    # after it (even with near-zero v_vis_body from noise) would
                    # partially re-inflate velocity and defeat ZUPT.
                    if (has_imu and self._imu_dt > 0
                            and self.prev_frame_pose is not None
                            and not _zupt_fired):           # <- FIX: guard added
                        vis_dp_world = (self.curr_pose[:3, 3]
                                        - self.prev_frame_pose[:3, 3])
                        R_cw = self.curr_pose[:3, :3].T   # world->body
                        v_vis_body = R_cw @ (vis_dp_world
                                             / max(self._imu_dt, 1e-6))
                        # Higher alpha when fast rotation (IMU may drift more)
                        vcorr_alpha = (0.50 if self._imu_omega_mag > 0.50
                                       else 0.35 if self._imu_omega_mag > 0.20
                                       else 0.25)
                        self.imu.correct_velocity(v_vis_body, alpha=vcorr_alpha)

                    # ── (FIX 2) Re-anchor IMU world rotation to SLAM pose ──
                    # After each successful visual frame, feed the SLAM-
                    # estimated orientation back into the IMU pre-integrator
                    # so _R_world_cam stays locked to the SLAM world frame.
                    # Without this, IMU world frame drifts -> gravity estimate
                    # and IMU-predicted poses gradually become wrong.
                    if has_imu:
                        self.imu.correct_R_world(self.curr_pose[:3, :3])

                else:
                    # ── (6) IMU HOLDOVER during visual failure ────────────
                    # If visual tracking failed and IMU is available and
                    # we haven't been holding over too long, propagate pose
                    # from IMU rather than keeping the stale last pose.
                    self._prev_num_inliers = 0   # reset confidence on failure
                    if (has_imu and self._imu_dt > 0 and
                            self._imu_holdover_frames < imu_hold_max_f and
                            self._imu_holdover_dt    < imu_hold_max_t):
                        # Pose propagated by the pre-computed IMU prediction.
                        # NOTE: particles were already propagated by the
                        # top-of-frame predict_with_imu call above - do NOT
                        # call predict_with_imu again here or particles will
                        # advance by 2× the actual IMU delta while curr_pose
                        # advances by 1×, making them inconsistent.
                        self.curr_pose = T_imu_pred.copy()
                        self._imu_holdover_frames += 1
                        self._imu_holdover_dt     += self._imu_dt
                        if self.frame_id % 10 == 0:
                            print(f" [IMU-HOLD] Frame {self.frame_id}: "
                                  f"pose propagated via IMU "
                                  f"({self._imu_holdover_frames} frames, "
                                  f"dt={self._imu_holdover_dt:.3f}s)")
                    else:
                        # Past holdover limit - accumulate visual failures
                        if self.frame_id > self._WARMUP_FRAMES:
                            self.consecutive_failures += 1
                        if self.frame_id % 30 == 0 and self.consecutive_failures > 0:
                            print(f" [WARN] Tracking fail #{self.consecutive_failures}, "
                                  f"inliers:{num_inliers}  frame:{self.frame_id}")

                    # ── BUG 3 FIX: ZUPT must fire on stationarity regardless ──
                    # Previously ZUPT was ONLY inside the 'if success' block.
                    # When visual tracking fails while the device is stationary:
                    #   • predict_with_imu already spread particles slightly
                    #   • Without ZUPT the PF drifts even though device is still
                    # Apply ZUPT + zero velocity here too so static operation
                    # never drifts, even during repeated visual tracking failures.
                    if has_imu and self.imu.is_stationary():
                        self.pf.apply_zupt()
                        self.imu.correct_velocity(
                            np.zeros(3, dtype=np.float64), alpha=1.0)

                # Only trigger relocalizer outside warmup period
                if (self.frame_id > self._WARMUP_FRAMES and
                        self.consecutive_failures >= self.relocalizer_trigger_threshold):
                    print(f" [!] GHOST relocalization after "
                          f"{self.consecutive_failures} failures")
                    self.state = TrackingState.RELOCALIZING
                    self.relocalizer_attempts = 0
                    # Reset holdover counters so the next TRACKING session
                    # gets the full holdover budget from config.
                    self._imu_holdover_frames = 0
                    self._imu_holdover_dt     = 0.0
                    self.relocalizer.scatter_hypotheses(
                        self.curr_pose, num_particles=256, mode="hybrid")

        # Chunked Bubble Map Updates (V52 Async Reconstruction)
        # ONLY update if tracking is successful or we are initializing (V46/V51)
        if (tracked or self.frame_id == 1) and self.state in [TrackingState.TRACKING, TrackingState.RECOVERY, TrackingState.UNINITIALIZED]:
            self.bubble_map.frame_counter = self.frame_id
            self.bubble_map.update_active_set(self.curr_pose[:3, 3])
            
            # Hybrid Gating: Force every frame during initialization or if moved
            force_mapping = is_initializing or (not _imu_static and self.bubble_map.should_update_bubble(self.curr_pose))
            
            if force_mapping:
                # V58: Increased stride to 8 to reduce point density and VRAM pressure
                stride = self.config.get("bubble_stride", 8)
                # Submit to async manager (Fix 1, 3)
                self.bubble_map.threaded_manager.submit(
                    depth_gpu, self.curr_pose.copy(), left_img_rect_gpu, 
                    stride=stride, is_initializing=is_initializing
                )

            # Periodic pruning every 30 frames (Fix 4)
            periodic_prune_interval = self.config.get("periodic_prune_interval", 30)
            if self.frame_id % periodic_prune_interval == 0:
                self.bubble_map.prune_active()
            
        # Also push on regular intervals even if camera is stationary
        viz_freq = self.config.get("visualization_update_frequency", 5)
        if self.frame_id % viz_freq == 0:
            self.bubble_map.push_to_visualizer()

        # Voxel grid updates (if enabled)
        if self.config.get("use_cupy_voxel_grid", True) and should_update_voxel:
            if self.frame_id % self.config.get("tsdf_integration_frequency", 3) == 0:
                # OPT: pass depth_gpu (CuPy array) directly - CupyVoxelGrid
                # now accepts CuPy input and skips the xp.asarray() copy.
                depth_for_voxel = depth_gpu if (USE_CUPY and depth_gpu is not None) else depth
                # V41: pass GPU image directly to voxel integration
                img_for_voxel   = (left_img_rect_gpu
                                   if (USE_CUPY and left_img_rect_gpu is not None)
                                   else left_img_rect)
                self.voxel_manager.integrate_frame(
                    img_for_voxel, depth_for_voxel, self.curr_pose, intrinsic)
            if self.frame_id % 100 == 0 and len(self.bubble_map) > 0:
                print(f" [MAP] Updated: {len(self.bubble_map)} bubbles")

        if tracked or self.frame_id == 1:
            # (7) Pass IMU omega to _handle_keyframe for threshold adaptation
            is_kf = self._handle_keyframe(feats, depth, left_img_rect, intrinsic,
                                          num_inliers, imu_omega=self._imu_omega_mag)

            # Always store previous frame for frame-to-frame tracking
            self.prev_frame_feats = feats
            self.prev_frame_pose  = self.curr_pose.copy()
            self.prev_frame_depth = depth.copy()
            # V9 FIX BUG 2: depth_gpu is now always a NEW owned CuPy array
            # (returned by xp.nan_to_num in ALL code paths above), never the
            # raw MGM buffer.  Safe to store without .copy() - the array lives
            # until we replace this reference next frame.
            if USE_CUPY and isinstance(depth_gpu, xp.ndarray):
                self.prev_frame_depth_gpu = depth_gpu

        elif self.state == TrackingState.RELOCALIZING:
            # During relocalization: propagate the current pose estimate via
            # IMU so the ghost relocalizer is seeded from a moving target,
            # not a stale frozen pose.  This also keeps the HUD position
            # meaningful and prevents the next scatter from a stale center.
            if has_imu and self._imu_dt > 0:
                self.curr_pose = self.curr_pose @ T_imu_delta
                # Also keep the particle cloud drifting with IMU during reloc
                self.pf.predict_with_imu(
                    self._imu_delta_R, self._imu_delta_p,
                    num_inliers=0, imu_weight=1.0)

            # V43: pass GPU image to evolve_ghosts when available so that
            # _score_photometric_consistency can use GPU BGR->Gray (no H2D).
            _reloc_img = (left_img_rect_gpu
                          if (USE_CUPY and left_img_rect_gpu is not None)
                          else left_img_rect)
            best_pose, score = self.relocalizer.evolve_ghosts(depth, _reloc_img, feats)
            if best_pose is not None:
                print(f" [RELOCALIZED] frame={self.frame_id}  score={score:.3f}")
                self.curr_pose = best_pose; self.pf.reset(best_pose)
                self.state = TrackingState.RECOVERY
                self.recovery_buffer.clear(); self.recovery_buffer.append(best_pose)
                self.recovery_count = 1; self.consecutive_failures = 0
                self.relocalizer_attempts = 0          # <- BUG FIX: reset on success
                self._imu_holdover_frames = 0; self._imu_holdover_dt = 0.0
                if should_update_bubble:
                    self.bubble_map.threaded_manager.submit(
                        depth_gpu, self.curr_pose.copy(), left_img_rect_gpu,
                        stride=self.config.get("bubble_stride", 8),
                        is_initializing=is_initializing)
                if self.config.get("use_cupy_voxel_grid", True) and should_update_voxel:
                    self.voxel_manager.integrate_frame(
                        left_img_rect, depth, self.curr_pose, intrinsic)
                # ── Re-anchor IMU world rotation on successful relocalization ─
                # The relocalizer may have placed the pose far from where the
                # IMU's _R_world_cam was tracking - re-anchor immediately so
                # the gravity estimate and IMU-predicted poses are correct
                # from the very first recovery frame.
                if has_imu:
                    self.imu.correct_R_world(self.curr_pose[:3, :3])
            else:
                self.relocalizer_attempts += 1
                if self.relocalizer_attempts >= self.relocalizer_max_attempts:
                    self.state = TrackingState.FAILED

        elif self.state == TrackingState.RECOVERY:
            # ── IMU propagation during recovery ──────────────────────────
            # Keep pose and particles moving via IMU while visual recovery
            # re-establishes stable tracking.  Analogous to RELOCALIZING.
            if has_imu and self._imu_dt > 0:
                self.pf.predict_with_imu(
                    self._imu_delta_R, self._imu_delta_p,
                    num_inliers=0, imu_weight=1.0)
            if self.prev_kf_feats is not None:
                success, T_obs, num_inliers, pts3d_w, pts2d = self.estimate_pose(
                    feats, self.prev_kf_feats, self.prev_kf_depth, self.prev_kf_pose,
                    T_hint=T_imu_pred if has_imu else None)
                if success:
                    self.recovery_buffer.append(T_obs)
                    wts = np.full(len(self.recovery_buffer), 1.0/len(self.recovery_buffer))
                    self.curr_pose = self._karcher_mean_recovery(
                        np.array(list(self.recovery_buffer)), wts)
                    self.recovery_count += 1
                    # Apply gravity constraint during recovery to stabilise roll/pitch
                    if has_imu and np.linalg.norm(self._imu_grav) > 1.0:
                        imu_grav_world = self.imu.get_gravity_world()
                        self.pf.update_with_gravity_constraint(
                            self._imu_grav, gravity_sigma=0.20,
                            gravity_world=imu_grav_world
                                if np.linalg.norm(imu_grav_world) > 1.0
                                else None)
                    if self.recovery_count >= self.recovery_buffer.maxlen:
                        self.state = TrackingState.TRACKING
                    if should_update_bubble:
                        self.bubble_map.threaded_manager.submit(
                            depth_gpu, self.curr_pose.copy(), left_img_rect_gpu,
                            stride=self.config.get("bubble_stride", 8),
                            is_initializing=is_initializing)
                    if self.config.get("use_cupy_voxel_grid", True) and should_update_voxel:
                        self.voxel_manager.integrate_frame(
                            left_img_rect, depth, self.curr_pose, intrinsic)
                    is_kf = self._handle_keyframe(feats, depth, left_img_rect, intrinsic,
                                          num_inliers, imu_omega=self._imu_omega_mag)
                    # ── Re-anchor IMU world rotation to recovered pose ────────
                    # Same as in TRACKING success: keep _R_world_cam synced so
                    # that gravity and IMU-predicted poses remain correct after
                    # the tracked-down recovery.
                    if has_imu:
                        self.imu.correct_R_world(self.curr_pose[:3, :3])
                    # ── Visual velocity correction in recovery ─────────────
                    if (has_imu and self._imu_dt > 0
                            and self.prev_frame_pose is not None):
                        vis_dp_world = (self.curr_pose[:3, 3]
                                        - self.prev_frame_pose[:3, 3])
                        R_cw = self.curr_pose[:3, :3].T
                        v_vis_body = R_cw @ (vis_dp_world / max(self._imu_dt, 1e-6))
                        vcorr_alpha = (0.50 if self._imu_omega_mag > 0.50
                                       else 0.35 if self._imu_omega_mag > 0.20
                                       else 0.25)
                        self.imu.correct_velocity(v_vis_body, alpha=vcorr_alpha)
                    # ── BUG 4 FIX: update prev_frame in RECOVERY success ──────
                    # prev_frame_feats/pose/depth were only updated inside the
                    # TRACKING state block (L≈3249).  When RECOVERY succeeds and
                    # transitions to TRACKING, the first frame-to-frame attempt
                    # used prev_frame data that was potentially many frames stale,
                    # causing one bad tracking frame at the recovery boundary.
                    self.prev_frame_feats = feats
                    self.prev_frame_pose  = self.curr_pose.copy()
                    self.prev_frame_depth = depth.copy()
                    # V9: keep GPU depth mirror in sync (always an owned array)
                    if USE_CUPY and isinstance(depth_gpu, xp.ndarray):
                        self.prev_frame_depth_gpu = depth_gpu
                else:
                    # Visual recovery failed this frame - keep IMU pose
                    if has_imu and self._imu_dt > 0:
                        self.curr_pose = self.curr_pose @ T_imu_delta
                    self.state = TrackingState.RELOCALIZING
                    self.relocalizer.scatter_hypotheses(
                        self.curr_pose, num_particles=256, mode="hybrid")

        elif self.state == TrackingState.FAILED:
            if self.frame_id % 50 == 0:
                print(" [RECOVERY] Global relocalization attempt ...")
            # ── Advance pose via IMU during FAILED state ──────────────────
            # Without this, curr_pose is stale when we seed the next
            # relocalizer scatter, biasing it to the wrong location.
            if has_imu and self._imu_dt > 0:
                self.curr_pose = self.curr_pose @ T_imu_delta
            scatter_pose = (self.voxel_manager.keyframes[-1].pose
                            if len(self.voxel_manager.keyframes) > 0
                            else self.curr_pose)
            self.relocalizer.scatter_hypotheses(
                scatter_pose, num_particles=256, mode="global")
            self.state = TrackingState.RELOCALIZING
            self.relocalizer_attempts = 0

        # ── Temporal consistency update ───────────────────────────────────
        if self.frame_id % 10 == 0:  # Update every 10 frames
            self.keyframe_manager.update_temporal_consistency()

        # ── 3D visualizer update (throttled) ────────────────────────────
        if self.config.get("enable_3d_visualization", True):
            if self.frame_id % self.config.get("visualization_update_frequency", 1) == 0:
                if self.visualizer and self.visualizer.is_active():
                    # The visualizer now consumes from the bubble_map._viz_queue
                    # so we just need to pass the other managers
                    self.visualizer.update_visualization(
                        bubble_map=self.bubble_map,
                        voxel_manager=self.voxel_manager,
                        relocalizer=(self.relocalizer
                                     if self.state == TrackingState.RELOCALIZING else None),
                        current_pose=self.curr_pose,
                        show_bubbles=self.config.get("show_bubbles", True),
                        show_voxels=self.config.get("show_voxels", True),
                        show_trajectory=self.config.get("show_camera_trajectory", True))

        # ── (12) DATASET RECORDING ──────────────────────────────────────
        if self.dataset is not None:
            try:
                # Record state-action pair
                # action = [ax, ay, az, gx, gy, gz] from IMU
                action = [ax, ay, az, gx, gy, gz] 
            except Exception as e:
                logger.warning(f" [DATA] Recording failed: {e}")

        # ── GPU memory cleanup (every 500 frames, not every 25) ─────────
        # OPT: mempool.free_all_blocks() every 25 frames was destroying
        # CuPy's allocation cache on every 25th frame, forcing re-allocation
        # for the next 25 frames (each alloc ~0.1 ms).  Net cost: 0.1 ms × 25
        # allocs × 2.4 recurrence = ~6 ms wasted per second at 60 fps.
        # Free at 500-frame intervals or on explicit shutdown instead.
        if self.frame_id % 500 == 0:
            if USE_CUPY:
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()

        # ── HUD ─────────────────────────────────────────────────────────
        num_inliers_hud = (0 if self.state != TrackingState.TRACKING
                           else locals().get('num_inliers', 0))
        self._draw_hud(feats, left_img_rect, depth_fps,
                       self.state == TrackingState.TRACKING,
                       num_inliers_hud, frame_start)

        return self.curr_pose

    # ----------------------------------------------------------------
    def _karcher_mean_recovery(self, poses, weights):
        # FIX 6: seed from highest-weight pose (argmax), not lowest.
        best    = np.argmax(weights); T_mean = poses[best].copy()
        for _ in range(6):
            Ti = PoseTransform.inverse(T_mean)
            Tit = np.tile(Ti, (len(poses),1,1))
            T_rel = SE3ParticleFilter._batch_matmul(Tit, poses)
            xi_all = SE3ParticleFilter._batch_log_se3(T_rel)
            xi_sum = (weights[:,None]*xi_all).sum(axis=0)
            T_mean = T_mean @ PoseTransform.exp_se3(xi_sum)
            if np.linalg.norm(xi_sum) < 1e-9: break
        return T_mean

    def _calculate_adaptive_keyframe_score(self, motion_score, quality_score, novelty_score):
        """Calculate adaptive keyframe insertion score using weighted factors."""
        if not self.config.get("adaptive_keyframe_enabled", True):
            # Fallback to simple motion-based decision
            return motion_score
        
        weights = {
            'motion': self.config.get("keyframe_motion_weight", 0.4),
            'quality': self.config.get("keyframe_quality_weight", 0.3),
            'novelty': self.config.get("keyframe_novelty_weight", 0.3)
        }
        
        # Normalize scores to [0, 1]
        motion_norm = np.clip(motion_score, 0, 1)
        quality_norm = np.clip(quality_score, 0, 1)
        novelty_norm = np.clip(novelty_score, 0, 1)
        
        # Calculate weighted score
        adaptive_score = (
            weights['motion'] * motion_norm +
            weights['quality'] * quality_norm +
            weights['novelty'] * novelty_norm
        )
        
        return adaptive_score

    def _calculate_motion_score(self, rel_T):
        """Calculate motion score based on translation and rotation."""
        trans_dist = np.linalg.norm(rel_T[:3, 3])
        rot_dist = PoseTransform.angular_distance(np.eye(3), rel_T[:3, :3])
        
        # Normalize by thresholds
        trans_thresh = self.config.get("keyframe_translation_threshold", 0.03)
        rot_thresh = self.config.get("keyframe_rotation_threshold", 0.05)
        
        trans_score = min(trans_dist / trans_thresh, 1.0)
        rot_score = min(rot_dist / rot_thresh, 1.0)
        
        return max(trans_score, rot_score)

    def _calculate_quality_score(self, num_inliers, expected_inliers):
        """Calculate tracking quality score based on inlier ratio."""
        if expected_inliers == 0:
            return 0.0
        
        inlier_ratio = num_inliers / expected_inliers
        min_ratio = self.config.get("keyframe_min_inlier_ratio", 0.3)
        
        # Higher score for better tracking quality
        return max(0.0, (inlier_ratio - min_ratio) / (1.0 - min_ratio))

    def _calculate_novelty_score(self):
        """Calculate scene novelty score based on new bubble ratio."""
        if not hasattr(self, '_last_bubble_count'):
            self._last_bubble_count = 0
            return 1.0  # First frame is always novel
        
        current_bubble_count = len(self.bubble_map)
        if self._last_bubble_count == 0:
            novelty_score = 1.0
        else:
            # Calculate ratio of new bubbles
            new_bubbles = max(0, current_bubble_count - self._last_bubble_count)
            novelty_ratio = new_bubbles / max(current_bubble_count, 1)
            novelty_thresh = self.config.get("keyframe_novelty_threshold", 0.2)
            novelty_score = min(novelty_ratio / novelty_thresh, 1.0)
        
        self._last_bubble_count = current_bubble_count
        return novelty_score

    def should_add_keyframe(self, pose, last_pose, new_bubbles, total_bubbles, frame_id):
        """Simple and reliable keyframe insertion criteria."""
        import numpy as np
        
        # Motion criteria
        translation = np.linalg.norm(pose[:3, 3] - last_pose[:3, 3])
        rotation = PoseTransform.angular_distance(pose[:3, :3], last_pose[:3, :3])
        
        # Scene novelty
        novelty = new_bubbles / max(total_bubbles, 1)
        
        # Thresholds from config
        t_thr = self.config.get("keyframe_translation_threshold", 0.15)
        r_thr = np.radians(self.config.get("keyframe_rotation_threshold", 5.0))
        n_thr = self.config.get("keyframe_novelty_threshold", 0.10)
        
        # Conditions
        motion_trigger = translation > t_thr or rotation > r_thr
        novelty_trigger = novelty > n_thr
        periodic_trigger = (frame_id % 20 == 0) # Tighter periodic fallback
        
        return motion_trigger or novelty_trigger or periodic_trigger

    # ----------------------------------------------------------------
    def _handle_keyframe(self, feats, depth, left_img_rect, intrinsic,
                         num_inliers, imu_omega: float = 0.0):
        """
        Decide whether current frame should become a keyframe.

        imu_omega : current angular rate (rad/s) from IMU gyroscope.
                    When fast motion is detected (omega > imu_fast_motion_gyro_thresh),
                    the rotation threshold is scaled down so keyframes are
                    captured more frequently and loop closure has denser
                    coverage across rotation.
        """
        # ── V37: Apply any pending async BA/PGO corrections ──────────
        # Corrections arrive as {kf_id: corrected_pose} from the worker.
        # Apply them to the live KF list and propagate the delta to curr_pose.
        pending_corr, do_reintegrate = self._async_opt.take_corrections()
        if pending_corr:
            kfs = self.voxel_manager.keyframes
            # Build id->index map for O(1) lookup
            id_to_kf = {kf.id: kf for kf in kfs}
            # Find the correction closest to curr_pose for T_delta
            best_delta = None
            best_dist  = np.inf
            for kf_id, new_pose in pending_corr.items():
                kf = id_to_kf.get(kf_id)
                if kf is None:
                    continue
                old_pose = kf.pose.copy()
                kf.pose  = new_pose           # write corrected pose in-place
                d = np.linalg.norm(new_pose[:3, 3] - self.curr_pose[:3, 3])
                if d < best_dist:
                    best_dist  = d
                    best_delta = new_pose @ PoseTransform.inverse(old_pose)
            if best_delta is not None:
                self.curr_pose = best_delta @ self.curr_pose
                self.pf.reset(self.curr_pose)
                if self.imu is not None:
                    self.imu.correct_R_world(self.curr_pose[:3, :3])
                self.last_keyframe_pose = self.curr_pose.copy()
            if do_reintegrate:
                print(" [V37] Async correction -> scheduling TSDF reintegration")
                try:
                    self.voxel_manager.reintegrate_map()
                    if hasattr(self.bubble_map, "reintegrate_map"):
                        self.bubble_map.reintegrate_map(self.voxel_manager.keyframes)
                except Exception as _re:
                    print(f" [V37] Reintegration skipped: {_re}")

        # ── (7) IMU-ADAPTED ROTATION THRESHOLD ───────────────────────
        rot_thr = self.config["keyframe_rotation_threshold"]
        fast_gyro_thr = self.config.get("imu_fast_motion_gyro_thresh", 0.30)
        if imu_omega > fast_gyro_thr and self.imu is not None:
            scale   = self.config.get("imu_fast_kf_rot_scale", 0.50)
            rot_thr = rot_thr * scale   # tighter threshold -> more KFs

        rel_T = PoseTransform.inverse(self.last_keyframe_pose) @ self.curr_pose
        
        # Calculate new bubbles for novelty
        if not hasattr(self, '_last_bubble_count'):
            self._last_bubble_count = 0
        current_bubble_count = len(self.bubble_map)
        new_bubbles = max(0, current_bubble_count - self._last_bubble_count)
        
        # Use simple and reliable keyframe criteria
        is_kf = self.should_add_keyframe(
            self.curr_pose, self.last_keyframe_pose, 
            new_bubbles, current_bubble_count, self.frame_id
        )
        
        # Always add first frame as keyframe (frame_id starts at 0, incremented before this call)
        is_kf = is_kf or (self.frame_id == 1)
        
        if not is_kf:
            return False
        
        # Update last bubble count ONLY when a keyframe is added
        self._last_bubble_count = current_bubble_count
        
        # Debug logging for keyframe decision
        translation = np.linalg.norm(rel_T[:3, 3])
        rotation = PoseTransform.angular_distance(np.eye(3), rel_T[:3, :3])
        novelty = new_bubbles / max(current_bubble_count, 1)
        print(f" [KF] Simple criteria: T:{translation:.3f} R:{rotation:.3f} N:{novelty:.3f}")
        keypoint_scores = feats.get('keypoint_scores', feats.get('scores', np.array([], dtype=np.float32)))
        new_kf = Keyframe(
            id=self.frame_id,
            pose=copy.deepcopy(self.curr_pose),
            # Handle both PyTorch tensors and NumPy arrays
            keypoints=feats['keypoints'].cpu().numpy() if hasattr(feats['keypoints'], 'cpu') else feats['keypoints'],
            descriptors=feats['descriptors'].cpu().numpy() if hasattr(feats['descriptors'], 'cpu') else feats['descriptors'],
            scores=keypoint_scores.cpu().numpy() if hasattr(keypoint_scores, 'cpu') else keypoint_scores,
            image=left_img_rect.copy(),
            depth=depth.copy(),
            intrinsics=self.K_rect_proc
        )
        
        # Add to the new global KeyframeManager
        self.keyframe_manager.add_keyframe(new_kf)
        
        # Transition from UNINITIALIZED to TRACKING on first keyframe
        if self.state == TrackingState.UNINITIALIZED:
            self.state = TrackingState.TRACKING
            print(f" [SLAM] System initialized and transitioning to TRACKING state at frame {self.frame_id}")

        # Check if it is done and apply the correction if so.
        lc_detected, lc_corrected_pose = self._async_opt.take_loop_closure_result()
        if lc_detected and lc_corrected_pose is not None:
            past_kfs = self.voxel_manager.keyframes
            if past_kfs and self._pgo_enabled:
                dists = [np.linalg.norm(kf.pose[:3,3] - lc_corrected_pose[:3,3])
                         for kf in past_kfs]
                matched_kf = past_kfs[int(np.argmin(dists))]
                # Use the most recent cached KF as the "current" side of the edge
                if past_kfs:
                    src_kf = past_kfs[-1]
                    self.pgo.add_loop_closure_edge(
                        new_kf.id, matched_kf.id,
                        lc_corrected_pose, matched_kf.pose)
                    print(f" [PGO/LC] Async loop edge: KF#{new_kf.id} ↔ KF#{matched_kf.id}")
            self.curr_pose = lc_corrected_pose
            # ── Patch the most-recently-cached keyframe pose in-place ─────
            # V43 did new_kf.pose = corrected_pose before cache_keyframe().
            # The async path runs AFTER caching, so we patch the stored object
            # here to keep the keyframe list consistent with curr_pose.
            # PGO reintegration will propagate the correction to older frames.
            _all_kfs = self.voxel_manager.keyframes
            if _all_kfs:
                _all_kfs[-1].pose = lc_corrected_pose.copy()
            # ── Also update new_kf (not yet cached) so it is cached correctly ──
            # new_kf was created from the pre-correction self.curr_pose = lc_corrected_pose
            new_kf.pose = lc_corrected_pose.copy()
            self.pf.reset(lc_corrected_pose)
            self.voxel_manager.reintegrate_map()
            if hasattr(self.bubble_map, "reintegrate_map"):
                self.bubble_map.reintegrate_map(self.voxel_manager.keyframes)
            if self.imu is not None:
                self.imu.correct_R_world(self.curr_pose[:3, :3])
            print(f" [LOOP/async] Correction applied; IMU world rotation re-anchored")
            if self._pgo_enabled:
                self._run_pose_graph_optimization()

        if (self.config.get("enable_loop_closure", True) and
                len(self.keyframe_manager.keyframes) > 5):
            # IMU-gated loop closure: skip when angular rate is high.
            _loop_omega_thr = self.config.get("imu_loop_omega_thresh", 0.20)
            loop_allowed = (self.imu is None or imu_omega <= _loop_omega_thr)
            if loop_allowed:
                # Get loop closure candidates from keyframe manager
                candidates = self.keyframe_manager.get_loop_closure_candidates(
                    self.curr_pose, 
                    min_distance=self.config.get("loop_closure_min_distance", 2.0),
                    max_distance=self.config.get("loop_closure_max_distance", 10.0)
                )
                
                if candidates:
                    # Convert candidate IDs to keyframe objects
                    candidate_kfs = [self.keyframe_manager.keyframes[kf_id] for kf_id in candidates]
                    
                    # Submit loop closure detection
                    accepted = self._async_opt.submit_loop_closure(
                        self.loop_detector.detect_and_verify,
                        new_kf, candidate_kfs)
                    if not accepted:
                        if self.frame_id % 50 == 0:
                            print(f" [LOOP/async] Previous detection still running - skipped")
                else:
                    if self.frame_id % 100 == 0:
                        print(f" [LOOP] No suitable candidates found")
            else:
                if self.frame_id % 50 == 0:
                    print(f" [LOOP] Skipped: omega={imu_omega:.2f} > "
                          f"thresh={_loop_omega_thr:.2f} rad/s (IMU gate)")
        self.voxel_manager.cache_keyframe(new_kf)

        # ── (V46) PERIODIC AUTO-SAVE ──────────────────────────────────
        # Save reconstruction state every N keyframes for persistence
        ba_freq = self.config.get("ba_frequency", 8)
        if len(self.voxel_manager.keyframes) % (ba_freq // 2 or 1) == 0:
            threading.Thread(target=self.save_map_checkpoint, 
                             args=(f"checkpoint_kf_{new_kf.id}",), 
                             daemon=True).start()

        # ── Add sequential odometry edge to pose graph ─────────────────
        if self._pgo_enabled and self._pgo_kf_count > 0:
            prev_kf_list = self.voxel_manager.keyframes
            if len(prev_kf_list) >= 2:
                prev_kf = prev_kf_list[-2]   # second-to-last (new_kf is now [-1], just cached)
                self.pgo.add_odometry_edge(
                    prev_kf.id, new_kf.id,
                    prev_kf.pose, new_kf.pose)
        self._pgo_kf_count += 1

        # ── Windowed bundle adjustment ──────────────────────────────────
        self._ba_kf_count += 1
        if (self._ba_enabled and
                self._ba_kf_count >= self._ba_frequency and
                len(self.voxel_manager.keyframes) >= 2):
            self._ba_kf_count = 0
            try:
                self._run_bundle_adjustment()
            except Exception as _ba_ex:
                print(f" [BA] Skipped (exception): {_ba_ex}")
        self.last_keyframe_pose = self.curr_pose.copy()
        if self.frame_id % 25 == 0:
            positions = self.pf.particles[:,:3,3]
            spread    = np.std(positions, axis=0)
            imu_info  = (f"IMU w={imu_omega:.2f}rad/s" if self.imu else "IMU:OFF")
            print(f" [KF] #{self.frame_id:>5d} | Feats:{len(feats['keypoints']):>4d} | "
                  f"Inliers:{num_inliers:>3d} | Bubbles:{len(self.bubble_map):>6d} | "
                  f"State:{self.state.name} | {imu_info}")
        return True

    # ----------------------------------------------------------------
    def _run_pose_graph_optimization(self):
        """
        Global pose graph optimization triggered after loop closure.

        V37: Submits to _AsyncOptWorker (non-blocking).
        Corrections are applied via _apply_opt_corrections().
        """
        kfs = self.voxel_manager.keyframes
        if len(kfs) < 2 or not self._pgo_enabled:
            return

        # Snapshot: copy poses + ids; worker operates independently.
        kf_snaps = [(kf.id, kf.pose.copy()) for kf in kfs]
        # Give the worker a shallow copy of the kfs list so it can call
        # pgo.optimize() with the expected interface (list of Keyframe-like objects).
        # We wrap each snapshot in a tiny SimpleNamespace so .id and .pose work.
        import types
        kfs_copy = []
        for kf_id, kf_pose in kf_snaps:
            ns = types.SimpleNamespace(); ns.id = kf_id; ns.pose = kf_pose.copy()
            kfs_copy.append(ns)

        def _pgo_job(kfs_copy=kfs_copy, kf_snaps=kf_snaps,
                     pgo=self.pgo, config=self.config):
            t0 = time.time()
            print(f" [PGO/async] Optimizing {len(kfs_copy)} KFs …")
            converged = pgo.optimize(kfs_copy)   # modifies kfs_copy[i].pose in-place
            dt = time.time() - t0
            max_c = 0.0
            corr_map = {}
            for i, (kf_id, old_pose) in enumerate(kf_snaps):
                delta = np.linalg.norm(kfs_copy[i].pose[:3, 3] - old_pose[:3, 3])
                max_c = max(max_c, delta)
                corr_map[kf_id] = kfs_copy[i].pose.copy()
            print(f" [PGO/async] Done {dt*1e3:.1f} ms  "
                  f"conv={converged}  max_corr={max_c*100:.2f} cm")
            trig = config.get("ba_correction_trigger", 0.15)
            do_reintegrate = (max_c > trig and
                              config.get("ba_reintegrate_on_correction", True))
            if USE_CUPY:
                try:
                    mempool.free_all_blocks()
                except Exception:
                    pass
            return corr_map, do_reintegrate

        accepted = self._async_opt.submit(_pgo_job)
        if not accepted:
            print(" [PGO/async] Skipped - previous job still running")

    # ----------------------------------------------------------------
    def _run_bundle_adjustment(self):
        """
        Windowed bundle adjustment over the last ba_window_size keyframes.

        Jointly refines camera poses and 3D landmarks using CuPy
        Schur-complement LM.  Updates keyframe poses in-place, then
        propagates the correction to curr_pose, PF, and IMU.

        If the mean pose correction exceeds ba_correction_trigger,
        the TSDF voxel grid is reintegrated with corrected poses.

        V37: Submits work to _AsyncOptWorker (non-blocking).
        Corrections are applied via _apply_opt_corrections() at the
        start of the next _handle_keyframe call.
        """
        kfs = self.voxel_manager.keyframes
        win = self.config.get("ba_window_size", 8)
        if len(kfs) < 2 or not self._ba_enabled:
            return

        # Build snapshot of window poses so the worker thread operates on
        # immutable data - no shared-mutable-state race with main thread.
        win_kfs   = kfs[-win:]
        kf_ids    = [kf.id for kf in win_kfs]
        kf_snaps  = [(kf.id, kf.pose.copy()) for kf in win_kfs]

        # Pass the entire KF list (worker needs it for feature matching)
        # Use list copy so cache_keyframe() appends don't affect the job.
        kfs_snap  = list(kfs)

        def _ba_job(kfs_snap=kfs_snap, kf_snaps=kf_snaps,
                    win=win, config=self.config,
                    bundle_adjuster=self.bundle_adjuster,
                    sp_lg=self.sp_lg):
            t0    = time.time()
            N_win = len(kf_snaps)
            print(f" [BA/async] Windowed BA over {N_win} KFs …")
            try:
                corrected_poses, converged = bundle_adjuster.run(kfs_snap, sp_lg)
            except Exception as _e:
                print(f" [BA/async] Error: {_e}")
                return None
            if not corrected_poses or len(corrected_poses) != N_win:
                return None
            dt      = time.time() - t0
            max_c   = 0.0
            corr_map = {}
            for i, (kf_id, old_pose) in enumerate(kf_snaps):
                delta = np.linalg.norm(corrected_poses[i][:3, 3] - old_pose[:3, 3])
                max_c = max(max_c, delta)
                corr_map[kf_id] = corrected_poses[i].copy()
            print(f" [BA/async] Done {dt*1e3:.1f} ms  "
                  f"conv={converged}  max_corr={max_c*100:.2f} cm")
            trig = config.get("ba_correction_trigger", 0.15)
            do_reintegrate = (max_c > trig and
                              config.get("ba_reintegrate_on_correction", True))
            if USE_CUPY:
                try:
                    mempool.free_all_blocks()
                except Exception:
                    pass
            return corr_map, do_reintegrate

        accepted = self._async_opt.submit(_ba_job)
        if not accepted:
            print(" [BA/async] Skipped - previous job still running")

    # ----------------------------------------------------------------
    def _get_depth_for_bubbles(self, depth_gpu, depth_cpu):
        """Helper function to get the correct depth array for bubble processing."""
        if USE_CUPY and depth_gpu is not None:
            return depth_gpu
        return depth_cpu
    
    def _get_image_for_bubbles(self, left_img_rect_gpu, left_img_rect):
        """Helper function to get the correct image array for bubble processing."""
        if USE_CUPY and left_img_rect_gpu is not None:
            return left_img_rect_gpu
        return left_img_rect

    # ----------------------------------------------------------------
    def _draw_hud(self, feats, left_img_rect, depth_fps, tracked, num_inliers, t0):
        """
        Build 2D HUD windows.
        """
        # ── StereoDepth depth visualisation (mirrors CPUOCKD viewer) ────
        if self._last_depth_gpu is not None:
            try:
                # V57: Aggressive cleanup before HUD allocation
                if USE_CUPY:
                    cp.get_default_memory_pool().free_all_blocks()

                d_gpu = self._last_depth_gpu   # xp.ndarray float32 metres
                # Convert to millimetres for percentile-based adaptive range
                d_mm_gpu = d_gpu * 1000.0
                
                # V57: More memory-efficient percentile calculation
                # Avoid mask indexing 'd_mm_gpu[d_mm_gpu > 0]' which creates a copy
                if USE_CUPY:
                    # Use a subsampled version for percentile to save memory
                    sub = d_mm_gpu[::4, ::4].ravel()
                    valid_mask = sub > 0
                    if bool(xp.any(valid_mask).item() if hasattr(xp.any(valid_mask), "item") else xp.any(valid_mask)):
                        valid = sub[valid_mask]
                        d_min = float(xp.percentile(valid, 5).item())
                        d_max = float(xp.percentile(valid, 95).item())
                    else:
                        d_min, d_max = 0.0, 10000.0
                else:
                    valid = d_mm_gpu[d_mm_gpu > 0]
                    if valid.size > 10:
                        d_min = float(np.percentile(valid, 5))
                        d_max = float(np.percentile(valid, 95))
                    else:
                        d_min, d_max = 0.0, 10000.0

                if d_max - d_min < 200:          # minimum 200 mm range
                    mid   = (d_min + d_max) / 2
                    d_min = max(0.0, mid - 100)
                    d_max = mid + 100

                # Normalise on GPU -> uint8 -> D2H for OpenCV colourmap
                vis_gpu  = xp.clip(d_mm_gpu, d_min, d_max) if USE_CUPY \
                           else np.clip(d_mm_gpu, d_min, d_max)
                vis_gpu  = (vis_gpu - d_min) / max(d_max - d_min, 1e-3)
                vis_u8   = (vis_gpu * 255).astype(xp.uint8 if USE_CUPY else np.uint8)
                vis_np   = to_numpy_safe(vis_u8)
                # Ensure single-channel uint8 for color map
                if vis_np.ndim != 2:
                    vis_np = vis_np.squeeze()  # Remove extra dimensions
                vis_np = vis_np.astype(np.uint8)

                depth_colored = cv2.applyColorMap(vis_np, cv2.COLORMAP_TURBO)
                depth_vis     = cv2.resize(depth_colored, (1280, 800))
                coverage_pct  = float((vis_np > 0).mean() * 100)

                # Overlay text (depth range + FPS + coverage)
                info = (f"Depth {d_min/1000:.2f}-{d_max/1000:.2f} m  "
                        f"cov={coverage_pct:.1f}%  fps={depth_fps:.1f}")
                cv2.putText(depth_vis, info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                # Centre-pixel depth readout
                cy_px, cx_px = depth_vis.shape[0]//2, depth_vis.shape[1]//2
                d_mid = float(d_gpu[d_gpu.shape[0]//2, d_gpu.shape[1]//2])
                cv2.putText(depth_vis, f"{d_mid:.2f}m",
                            (cx_px - 30, cy_px - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                cv2.circle(depth_vis, (cx_px, cy_px), 5, (255,255,255), -1)
            except Exception as e:
                print(f" [DEPTH ERROR] Depth visualization failed: {e}")
                import traceback
                traceback.print_exc()
                depth_vis    = np.zeros((800, 1280, 3), dtype=np.uint8)
                coverage_pct = 0.0
        else:
            depth_vis    = np.zeros((800, 1280, 3), dtype=np.uint8)
            coverage_pct = 0.0

        # ── Feature overlay ──────────────────────────────────────────────
        vis_img = left_img_rect.copy()
        # Handle both PyTorch tensors and NumPy arrays for keypoints
        if hasattr(feats['keypoints'], 'cpu'):
            kpts = feats['keypoints'].cpu().numpy()
        else:
            kpts = feats['keypoints']
        step    = max(1, len(kpts)//150)
        for pt in kpts[::step]:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0,255,0), -1)

        # ── FPS ──────────────────────────────────────────────────────────
        frame_time = time.time() - t0
        self.timing_buffer.append(frame_time)
        avg_fps   = (1.0 / (sum(self.timing_buffer)/len(self.timing_buffer))
                     if self.timing_buffer else 0.0)
        avg_d_fps = (sum(self.depth_fps_buffer)/len(self.depth_fps_buffer)
                     if self.depth_fps_buffer else 0.0)

        # V58: Return early to avoid HUD overhead and VRAM allocations
        return

        # State colour
        state_color = {
            TrackingState.TRACKING:     (0,255,0),
            TrackingState.RELOCALIZING: (0,165,255),
            TrackingState.RECOVERY:     (255,255,0),
            TrackingState.FAILED:       (0,0,255),
        }.get(self.state, (255,255,255))

        lines = [
            f"Frame:{self.frame_id} | KFs:{len(self.voxel_manager.keyframes)}",
            f"State:{self.state.name}",
            f"SLAM FPS:{avg_fps:.1f} | Depth FPS:{avg_d_fps:.1f}",
            f"Track:{'OK' if tracked else 'LOST'} | Inliers:{num_inliers}",
            f"Features:{len(kpts)} | Bubbles:{len(self.bubble_map)}",
            f"Depth res:{self.proc_size[0]}x{self.proc_size[1]} (StereoDepth) | "
            f"cov:{coverage_pct:.1f}%",
            f"Pos:({self.curr_pose[0,3]:.2f},{self.curr_pose[1,3]:.2f},{self.curr_pose[2,3]:.2f})",
            f"PGO:{'ON' if self._pgo_enabled else 'OFF'} "
            f"edges(odom={len(self.pgo._odom_edges)},loop={len(self.pgo._loop_edges)}) | "
            f"BA:{'ON' if self._ba_enabled else 'OFF'} "
            f"next_in={max(0,self._ba_frequency-self._ba_kf_count)}KFs",
        ]
        # IMU HUD line with bias norms and stationary/moving indicator
        if self.imu:
            g_bias, a_bias = self.imu.get_bias_norms()
            stat_str = "STAT/ZUPT" if self.imu.is_stationary() else "MOV"
            imu_line = (f"IMU  w={self._imu_omega_mag:.3f}rad/s  "
                        f"dt={self._imu_dt*1e3:.1f}ms  "
                        f"hold={self._imu_holdover_frames}f  "
                        f"g={np.linalg.norm(self._imu_grav):.2f}m/s²  "
                        f"{stat_str}  "
                        f"Bg={g_bias*1e3:.1f}mrad/s  Ba={a_bias*1e3:.1f}mm/s²")
        else:
            imu_line = "IMU: disabled"
        lines.append(imu_line)
        y = 25
        for txt in lines:
            cv2.putText(vis_img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, state_color, 1)
            y += 18
        cv2.circle(vis_img, (vis_img.shape[1]-25, 25), 8, state_color, -1)

        cv2.imshow("SLAM  -  OAK-D StereoDepth [CuPy]", vis_img)
        cv2.imshow("OAK-D Depth  (640×400 -> 1280×800)", depth_vis)

    # ----------------------------------------------------------------
    def load_reconstruction_state(self):
        """Load reconstruction state from disk if available."""
        output_folder = self.config.get("output_folder", "scan_output")
        
        if not os.path.exists(output_folder):
            print(f" [LOAD] No previous reconstruction found at {output_folder}")
            return False
        
        loaded_any = False
        
        # Load keyframes
        try:
            keyframes_file = os.path.join(output_folder, "keyframes.npy")
            if os.path.exists(keyframes_file):
                keyframes_data = np.load(keyframes_file, allow_pickle=True)
                for kf_dict in keyframes_data:
                    kf = Keyframe(
                        id=kf_dict['id'],
                        pose=kf_dict['pose'],
                        image=np.zeros((1, 1), dtype=np.uint8),  # Placeholder
                        depth=np.zeros((1, 1), dtype=np.float32),
                        intrinsics=self.K_rect_proc,
                        keypoints=kf_dict.get('keypoints'),
                        descriptors=kf_dict.get('descriptors'),
                        scores=kf_dict.get('scores'),
                    )
                    self.voxel_manager.keyframes.append(kf)
                print(f" [LOAD] Restored {len(keyframes_data)} keyframes")
                loaded_any = True
        except Exception as e:
            print(f" [WARN] Failed to load keyframes: {e}")
        
        # Load voxel grid
        try:
            voxel_file = os.path.join(output_folder, "voxel_grid.npz")
            if os.path.exists(voxel_file) and self.voxel_manager.voxel_grid is not None:
                grid_data = np.load(voxel_file, allow_pickle=True)
                # Restore TSDF, weight, color if using CuPy
                if USE_CUPY:
                    self.voxel_manager.voxel_grid.tsdf = xp.asarray(grid_data['tsdf'])
                    self.voxel_manager.voxel_grid.weight = xp.asarray(grid_data['weight'])
                    self.voxel_manager.voxel_grid.color = xp.asarray(grid_data['color'])
                print(f" [LOAD] Restored voxel grid")
                loaded_any = True
        except Exception as e:
            print(f" [WARN] Failed to load voxel grid: {e}")
        
        # Load bubble map
        try:
            bubble_file = os.path.join(output_folder, "bubble_map.npz")
            if os.path.exists(bubble_file) and self.bubble_map is not None:
                bubble_data = np.load(bubble_file, allow_pickle=True)
                # Use load_bubbles method instead of direct property assignment
                self.bubble_map.load_bubbles(
                    mu=bubble_data['mu'],
                    Sigma=bubble_data['Sigma'],
                    weight=bubble_data['weight'],
                    color=bubble_data['color']
                )
                print(f" [LOAD] Restored bubble map with {len(self.bubble_map)} points")
                loaded_any = True
        except Exception as e:
            print(f" [WARN] Failed to load bubble map: {e}")
        
        if loaded_any:
            print(f" [LOAD] Reconstruction state restored from {output_folder}")
        
        return loaded_any


    def get_global_reconstruction(self):
        """Get globally fused reconstruction from all keyframes."""
        if self.keyframe_manager is not None:
            return self.keyframe_manager.get_global_reconstruction()
        return np.empty((0, 3)), np.empty((0, 3, 3)), np.empty(0), np.empty((0, 3))

    def extract_tsdf_mesh(self, min_weight=1.0, use_open3d=True):
        """
        Extract TSDF mesh for proper 3D reconstruction output.
        
        Args:
            min_weight: Minimum weight threshold for mesh extraction
            use_open3d: Use Open3D if available, otherwise PyVista
            
        Returns:
            Tuple of (vertices, triangles, colors) or None if extraction fails
        """
        if self.voxel_manager is None or self.voxel_manager.voxel_grid is None:
            print(" [TSDF] No voxel grid available for mesh extraction")
            return None
        
        try:
            # PyVista only - no Open3D option
            verts, tris, colors = self.voxel_manager.get_voxel_mesh_pyvista(min_weight)
            
            if verts is not None:
                print(f" [TSDF] Extracted mesh: {len(verts)} vertices, {len(tris)} triangles")
                return verts, tris, colors
            else:
                print(" [TSDF] Mesh extraction failed - no vertices")
                return None
                
        except Exception as e:
            print(f" [TSDF] Mesh extraction error: {e}")
            return None

    def get_reconstruction_output(self, include_mesh=True, mesh_min_weight=1.0):
        """
        Get complete reconstruction output including bubbles and mesh.
        
        Args:
            include_mesh: Whether to extract TSDF mesh
            mesh_min_weight: Minimum weight for mesh extraction
            
        Returns:
            dict with 'bubbles' and 'mesh' keys
        """
        output = {
            'bubbles': {
                'points': None,
                'colors': None,
                'count': 0
            },
            'mesh': {
                'vertices': None,
                'triangles': None,
                'colors': None,
                'vertex_count': 0,
                'triangle_count': 0
            }
        }
        
        # Get global bubble reconstruction
        mu, Sigma, weight, color = self.get_global_reconstruction()
        if len(mu) > 0:
            output['bubbles'] = {
                'points': mu,
                'colors': color,
                'count': len(mu)
            }
            print(f" [RECONSTRUCTION] Global bubbles: {len(mu)} points")
        
        # Get TSDF mesh if requested
        if include_mesh:
            mesh_result = self.extract_tsdf_mesh(mesh_min_weight)
            if mesh_result is not None:
                verts, tris, colors = mesh_result
                output['mesh'] = {
                    'vertices': verts,
                    'triangles': tris,
                    'colors': colors,
                    'vertex_count': len(verts),
                    'triangle_count': len(tris)
                }
                print(f" [RECONSTRUCTION] TSDF mesh: {len(verts)} vertices, {len(tris)} triangles")
        
        return output

    def save_map_checkpoint(self, name="latest"):
        """
        Periodically save reconstruction state for persistence.
        Runs in background to avoid blocking the tracking thread.
        """
        # DISABLED: Output folder creation and file saving removed per user request
        return
        
        # try:
        #     # 1. Save metadata
        #     state = {
        #         'frame_id': self.frame_id,
        #         'num_keyframes': len(self.voxel_manager.keyframes),
        #         'num_bubbles': len(self.bubble_map) if self.bubble_map else 0,
        #         'timestamp': time.time()
        #     }
        #     # import yaml  # DISABLED - File saving removed per user request
        #     # with open(os.path.join(checkpoint_dir, "metadata.yaml"), 'w') as f:
        #     #     yaml.dump(state, f)
        #         
        #     # 2. Save bubble map (GPU resident)
        #     if self.bubble_map is not None and len(self.bubble_map) > 0:
        #         bubble_data = {
        #             'mu': to_numpy_safe(self.bubble_map.mu),
        #             'Sigma': to_numpy_safe(self.bubble_map.Sigma),
        #             'weight': to_numpy_safe(self.bubble_map.weight),
        #             'color': to_numpy_safe(self.bubble_map.color),
        #         }
        #         np.savez_compressed(os.path.join(checkpoint_dir, "bubble_map.npz"), **bubble_data)
        #         
        #     # 3. Save voxel grid
        #     if self.voxel_manager.voxel_grid is not None:
        #         vg = self.voxel_manager.voxel_grid
        #         grid_data = {
        #             'sdf': to_numpy_safe(vg.sdf_grid),
        #             'weight': to_numpy_safe(vg.weight_grid),
        #             'color': to_numpy_safe(vg.color_grid),
        #             'origin': to_numpy_safe(vg.origin),
        #             'voxel_length': float(vg.voxel_length)
        #         }
        #         np.savez_compressed(os.path.join(checkpoint_dir, "voxel_grid.npz"), **grid_data)
        #         
        #     # 4. Save keyframe metadata
        #     kf_data = []
        #     for kf in self.voxel_manager.keyframes:
        #         kf_data.append({
        #             'id': kf.id,
        #             'pose': kf.pose,
        #             'scores_mean': float(np.mean(kf.scores)) if kf.scores is not None and len(kf.scores) > 0 else 0
        #         })
        #     np.save(os.path.join(checkpoint_dir, "keyframes_meta.npy"), np.array(kf_data, dtype=object))
        #     
        #     # Atomic update of 'latest' symlink/copy if needed
        #     if name != "latest":
        #         latest_dir = os.path.join(output_folder, "latest")
        #         # In Windows, we just copy or rename. For now, we'll just let them coexist.
        #         
        #     logger.info(f" [SAVE] Map checkpoint '{name}' saved successfully")
        # except Exception as e:
        #     logger.error(f" [SAVE] Checkpoint failed: {e}")

    def save_reconstruction_state(self):
        """Save the current reconstruction state to file (metadata only)."""
        try:
            # import yaml  # DISABLED - File saving removed per user request
            # from datetime import datetime
            
            state = {
                'timestamp': datetime.now().isoformat(),
                'frame_id': self.frame_id,
                'num_keyframes': len(self.voxel_manager.keyframes),
                'bubble_map_size': len(self.bubble_map) if self.bubble_map else 0,
                'config': {k: v for k, v in self.config.items() if isinstance(v, (int, float, str, bool))}
            }
            
            output_dir = self.config.get("output_folder", "scan_output")
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, "reconstruction_state.yaml")
            
            with open(filename, 'w') as f:
                yaml.dump(state, f, default_flow_style=False)
            
            print(f" [INFO] Metadata saved to {filename}")
        except Exception as e:
            print(f" [ERROR] Failed to save metadata: {e}")

    def shutdown(self):
        """Save reconstruction state (keyframes, voxel grid, bubble map, dataset) to disk."""
        # DISABLED: Output folder creation removed per user request
        # output_folder = self.config.get("output_folder", "scan_output")
        # os.makedirs(output_folder, exist_ok=True)
        
        # DISABLED: File saving operations removed per user request
        # try:
        #     self.save_reconstruction_state()
        # except Exception as e:
        #     print(f" [ERROR] Failed to save metadata: {e}")
        
        # # 2. Save dataset (DISABLED)
        # if self.dataset is not None:
        #     print(" [DATA] Dataset saving disabled per user request")
        
        # # 3. Save keyframes with poses (DISABLED)
        # try:
        #     keyframes_data = []
        #     for kf in self.voxel_manager.keyframes:
        #         kf_dict = {
        #             'id': kf.id,
        #             'pose': kf.pose,
        #             'keypoints': kf.keypoints,
        #             'descriptors': kf.descriptors,
        #         }
        #         keyframes_data.append(kf_dict)
        #     
        #     with open(os.path.join(output_folder, "keyframes.yaml"), 'w') as f:
        #         yaml.dump(keyframes_data, f)
        #     print(f" [SAVE] Saved {len(keyframes_data)} keyframes")
        # except Exception as e:
        #     print(f" [ERROR] Failed to save keyframes: {e}")
        print(" [SAVE] File saving disabled per user request")

        # 6. Stop hardware and worker threads
        self.cam.stop()
        if self.imu is not None:
            self.imu.stop()

        if self.visualizer:
            self.visualizer.close()
        cv2.destroyAllWindows()

        # 7. Print Statistics
        print("\n" + "="*60)
        print("3D RECONSTRUCTION STATISTICS")
        print("="*60)
        print(f"Total Frames Processed  : {self.frame_id}")
        print(f"Total Keyframes         : {len(self.voxel_manager.keyframes)}")
        print(f"Final Bubble Map Size   : {len(self.bubble_map) if self.bubble_map is not None else 0} points")
        if self.frame_id > 0:
            srate = (self.frame_id-self.consecutive_failures)/self.frame_id*100
            print(f"Tracking Success Rate   : {srate:.1f}%")
        
        if self.imu is not None:
            g_bias, a_bias = self.imu.get_bias_norms()
            print(f"IMU gyro bias norm      : {g_bias*1e3:.2f} mrad/s")
            print(f"IMU accel bias norm     : {a_bias*1e3:.2f} mm/s²")
        
        print(f"PGO                     : {'enabled' if self._pgo_enabled else 'disabled'}  "
              f"({len(self.pgo._odom_edges)} odom edges, "
              f"{len(self.pgo._loop_edges)} loop edges)")
        print(f"Bundle Adjustment       : {'enabled' if self._ba_enabled else 'disabled'}  "
              f"(window={self.config.get('ba_window_size',8)} KFs, "
              f"freq={self._ba_frequency} KFs)")
        print("="*60)

        # 8. Cleanup mapping resources AFTER stats
        self.voxel_manager.shutdown()
        self.bubble_map.shutdown()

        # 9. Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if USE_CUPY:
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        
        try:
            self.relocalizer._score_executor.shutdown(wait=False)
        except Exception:
            pass
        
        print(" [SLAM] Shutdown complete. File saving disabled per user request")
