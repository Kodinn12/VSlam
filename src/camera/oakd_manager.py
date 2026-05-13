"""OAK-D StereoDepth manager with IMU."""

import cv2
import numpy as np
import depthai as dai
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    cp = np
    USE_CUPY = False

class OakDStereoManager:
    """Manages OAK-D stereo depth pipeline, provides rectified left + depth maps."""

    def __init__(self, config: dict):
        self.config = config
        self.fps = config.get("camera_fps", 60)
        dres = config.get("depth_resolution", (640, 400))
        self.proc_w, self.proc_h = int(dres[0]), int(dres[1])

        self.P1_rect = None
        self.baseline_m = 0.0
        self.f_b_term = 0.0
        self.R_cam_imu = np.eye(3, dtype=np.float64)
        self._imu_queue = None

        self._load_calibration()
        self.pipeline = dai.Pipeline()
        self._build_pipeline()
        logger.info(f"Pipeline built: {self.proc_w}x{self.proc_h} @ {self.fps} fps")

    def _load_calibration(self):
        try:
            with dai.Device() as tmp:
                calib = tmp.readCalibration()
        except Exception as e:
            raise RuntimeError(f"Cannot read EEPROM calibration: {e}")

        w, h = self.proc_w, self.proc_h
        M1 = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, w, h), dtype=np.float64)
        D1 = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B), dtype=np.float64)
        M2 = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, w, h), dtype=np.float64)
        D2 = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C), dtype=np.float64)
        ext = np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C), dtype=np.float64)
        R = ext[:3, :3]
        T = ext[:3, 3]

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M1, D1, M2, D2, (w, h), R, T, alpha=0)
        self.P1_rect = P1.copy()
        self.baseline_m = abs(P2[0, 3]) / P1[0, 0] if P1[0, 0] > 0 else 0.075
        self.f_b_term = abs(P2[0, 3])

        # IMU extrinsics
        try:
            if hasattr(dai.CameraBoardSocket, 'IMU'):
                raw = calib.getCameraExtrinsics(dai.CameraBoardSocket.IMU, dai.CameraBoardSocket.CAM_B)
                if raw is not None:
                    R_imu = np.array(raw, dtype=np.float64).squeeze()[:3, :3]
                    U, _, Vt = np.linalg.svd(R_imu)
                    self.R_cam_imu = U @ Vt
                    logger.info("R_cam_imu loaded from EEPROM")
        except Exception:
            logger.warning("IMU extrinsics not available, using identity")

    def _build_pipeline(self):
        p = self.pipeline
        qsize = max(4, self.fps // 10)

        cam_left = p.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_left.setFps(self.fps)

        cam_right = p.create(dai.node.MonoCamera)
        cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setFps(self.fps)

        stereo = p.create(dai.node.StereoDepth)
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
        except AttributeError:
            pass
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        stereo.setOutputSize(self.proc_w, self.proc_h)

        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        self._q_depth = stereo.depth.createOutputQueue(maxSize=qsize, blocking=False)
        self._q_left = stereo.rectifiedLeft.createOutputQueue(maxSize=qsize, blocking=False)

        # IMU
        if self.config.get("enable_imu", True):
            try:
                imu_node = p.create(dai.node.IMU)
                imu_node.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
                imu_node.setBatchReportThreshold(1)
                imu_node.setMaxBatchReports(10)
                self._imu_queue = imu_node.out.createOutputQueue(maxSize=50, blocking=False)
                logger.info("IMU node added (ACCEL+GYRO @200Hz)")
            except Exception as e:
                logger.warning(f"IMU injection failed: {e}")

    def start(self):
        self.device = self.pipeline.start()
        logger.info("OAK-D pipeline started")

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def get_frames(self):
        left_msg = self._q_left.tryGet()
        depth_msg = self._q_depth.tryGet()
        if left_msg is None or depth_msg is None:
            return None, None
        left_gray = left_msg.getCvFrame()
        if left_gray.ndim == 3:
            left_gray = cv2.cvtColor(left_gray, cv2.COLOR_BGR2GRAY)
        depth_np = depth_msg.getFrame().astype(np.float32) / 1000.0
        depth_np[depth_np <= 0.0] = 0.0
        if USE_CUPY:
            depth_gpu = cp.asarray(depth_np)
            depth_gpu = cp.nan_to_num(depth_gpu, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            depth_gpu = np.nan_to_num(depth_np)
        return left_gray, depth_gpu
