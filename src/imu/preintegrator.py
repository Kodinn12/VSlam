"""IMU pre-integration from OAK-D ACCEL/gyro streams."""

import threading
import time
import math
import numpy as np
from collections import deque
from ..utils.logger import get_logger

logger = get_logger(__name__)

class IMUPreIntegrator:
    """Pre-integrates IMU data between camera frames, provides delta_R, delta_p, gravity."""

    GRAVITY = 9.81
    BIAS_WINDOW = 200
    GYRO_STAT_TH = 0.015
    ACCEL_LP = 0.005
    MAX_DELTA_P_PER_AXIS = 0.15
    MAX_VELOCITY = 2.0
    ACCEL_SPIKE_THRESH = 4.0 * GRAVITY
    STATIONARY_HYSTERESIS = 20
    MOVING_HYSTERESIS = 5

    def __init__(self, imu_queue, R_cam_imu: np.ndarray = None):
        self._q = imu_queue
        self._R_cam_imu = R_cam_imu.copy() if R_cam_imu is not None else np.eye(3)
        self._lock = threading.Lock()
        self._delta_R = np.eye(3)
        self._delta_v = np.zeros(3)
        self._delta_p = np.zeros(3)
        self._dt_acc = 0.0
        self._omega_mag = 0.0
        self._accel_cam = np.zeros(3)
        self._gravity_cam = np.zeros(3)
        self._gravity_world = np.zeros(3)
        self._gravity_init = False
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._stat_gyro_buf = deque(maxlen=self.BIAS_WINDOW)
        self._stat_accel_buf = deque(maxlen=self.BIAS_WINDOW)
        self._velocity = np.zeros(3)
        self._R_world_cam = np.eye(3)
        self._renorm_counter = 0
        self._stat_raw_count = 0
        self._moving_raw_count = 0
        self._is_stationary = False
        self._last_ts = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("IMU pre-integrator thread started")

    def _run(self):
        while self._running:
            try:
                imu_msg = self._q.tryGet()
                if imu_msg is None:
                    time.sleep(0.001)
                    continue
                for pkt in imu_msg.packets:
                    self._process_packet(pkt)
            except Exception:
                pass

    def _process_packet(self, pkt):
        accel = pkt.acceleroMeter
        gyro = pkt.gyroscope
        a_imu = np.array([accel.x, accel.y, accel.z], dtype=np.float64)
        w_imu = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float64)
        a_raw = self._R_cam_imu @ a_imu
        w_raw = self._R_cam_imu @ w_imu
        ts_s = accel.getTimestampDevice().total_seconds()
        if self._last_ts is None:
            self._last_ts = ts_s
            return
        dt = ts_s - self._last_ts
        if dt <= 0.0 or dt > 0.1:
            self._last_ts = ts_s
            return
        self._last_ts = ts_s
        accel_spike = np.linalg.norm(a_raw) > self.ACCEL_SPIKE_THRESH
        if not accel_spike:
            self._stat_gyro_buf.append(w_raw)
            self._stat_accel_buf.append(a_raw)
        new_gyro_bias = None
        new_accel_bias = None
        if len(self._stat_gyro_buf) == self.BIAS_WINDOW:
            gyro_arr = np.array(self._stat_gyro_buf)
            gyro_mean = gyro_arr.mean(axis=0)
            gyro_resid = gyro_arr - gyro_mean
            if np.all(np.linalg.norm(gyro_resid, axis=1) < self.GYRO_STAT_TH):
                new_gyro_bias = gyro_mean
                a_mean = np.mean(np.array(self._stat_accel_buf), axis=0)
                g_dir = a_mean / (np.linalg.norm(a_mean) + 1e-9)
                new_accel_bias = a_mean - g_dir * self.GRAVITY
        with self._lock:
            if new_gyro_bias is not None:
                self._gyro_bias = new_gyro_bias
                self._accel_bias = new_accel_bias
            w = w_raw - self._gyro_bias
            a = a_raw - self._accel_bias
            angle = np.linalg.norm(w) * dt
            if angle > 1e-12:
                axis = w / (np.linalg.norm(w) + 1e-12)
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]], dtype=np.float64)
                dR = np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)
            else:
                dR = np.eye(3)
            self._R_world_cam = self._R_world_cam @ dR
            self._renorm_counter += 1
            if self._renorm_counter >= 100:
                U, _, Vt = np.linalg.svd(self._R_world_cam)
                self._R_world_cam = U @ Vt
                self._renorm_counter = 0
            a_world = self._R_world_cam @ a
            if not self._gravity_init:
                if new_gyro_bias is not None:
                    a_boot = np.mean(np.array(self._stat_accel_buf), axis=0)
                    a_boot_bc = a_boot - self._accel_bias
                    g_boot_world = self._R_world_cam @ a_boot_bc
                    gn = np.linalg.norm(g_boot_world)
                    self._gravity_world = g_boot_world * (self.GRAVITY / max(gn, 1e-6))
                    self._gravity_cam = self._R_world_cam.T @ self._gravity_world
                    self._gravity_init = True
                    logger.info("Gravity bootstrapped from stationary window")
            else:
                self._gravity_world = (1.0 - self.ACCEL_LP) * self._gravity_world + self.ACCEL_LP * a_world
                gw_norm = np.linalg.norm(self._gravity_world)
                if gw_norm > 1e-6:
                    self._gravity_world *= self.GRAVITY / gw_norm
                self._gravity_cam = self._R_world_cam.T @ self._gravity_world
            a_lin = a - self._gravity_cam if self._gravity_init else None
            if not accel_spike and a_lin is not None:
                v_start = self._velocity + self._delta_v
                dv = a_lin * dt
                dp = v_start * dt + 0.5 * a_lin * dt * dt
                self._delta_R = self._delta_R @ dR
                self._delta_v += dv
                self._delta_p += dp
            else:
                self._delta_R = self._delta_R @ dR
            self._dt_acc += dt
            self._omega_mag = float(np.linalg.norm(w))
            self._accel_cam = a_raw.copy()
            # Stationary hysteresis
            a_norm_now = float(np.linalg.norm(a_raw))
            raw_stat = (self._omega_mag < self.GYRO_STAT_TH and
                        0.85*self.GRAVITY < a_norm_now < 1.15*self.GRAVITY)
            if raw_stat:
                self._stat_raw_count = min(self._stat_raw_count + 1, self.STATIONARY_HYSTERESIS + 1)
                self._moving_raw_count = 0
                if self._stat_raw_count >= self.STATIONARY_HYSTERESIS:
                    self._is_stationary = True
            else:
                self._moving_raw_count = min(self._moving_raw_count + 1, self.MOVING_HYSTERESIS + 1)
                self._stat_raw_count = 0
                if self._moving_raw_count >= self.MOVING_HYSTERESIS:
                    self._is_stationary = False

    def get_delta(self, reset: bool = True):
        with self._lock:
            dR = self._delta_R.copy()
            omg = self._omega_mag
            dt = self._dt_acc
            grav = self._gravity_cam.copy()
            dp = np.clip(self._delta_p, -self.MAX_DELTA_P_PER_AXIS, self.MAX_DELTA_P_PER_AXIS)
            if reset:
                decay = math.exp(-0.3 * max(dt, 0.0))
                new_vel = (self._velocity + self._delta_v) * decay
                vel_mag = np.linalg.norm(new_vel)
                if vel_mag > self.MAX_VELOCITY:
                    new_vel = new_vel * (self.MAX_VELOCITY / vel_mag)
                self._velocity = new_vel
                self._delta_R = np.eye(3)
                self._delta_v = np.zeros(3)
                self._delta_p = np.zeros(3)
                self._dt_acc = 0.0
        return dR, dp, omg, dt, grav

    def correct_R_world(self, R_world_cam: np.ndarray):
        if R_world_cam is None or R_world_cam.shape != (3, 3):
            return
        with self._lock:
            R = R_world_cam.astype(np.float64)
            c0 = R[:, 0]; n0 = np.linalg.norm(c0)
            if n0 < 1e-9:
                return
            c0 /= n0
            c1 = R[:, 1]; c1 -= c0 * np.dot(c0, c1); n1 = np.linalg.norm(c1)
            if n1 < 1e-9:
                return
            c1 /= n1
            c2 = np.cross(c0, c1)
            self._R_world_cam = np.column_stack([c0, c1, c2])
            if self._gravity_init:
                self._gravity_cam = self._R_world_cam.T @ self._gravity_world

    def is_stationary(self) -> bool:
        with self._lock:
            return self._is_stationary

    def get_gravity_world(self) -> np.ndarray:
        with self._lock:
            return self._gravity_world.copy()

    def get_bias_norms(self):
        with self._lock:
            return float(np.linalg.norm(self._gyro_bias)), float(np.linalg.norm(self._accel_bias))

    def get_gyro_bias(self) -> np.ndarray:
        """Get current gyroscope bias vector."""
        with self._lock:
            return self._gyro_bias.copy()

    def get_accel_bias(self) -> np.ndarray:
        """Get current accelerometer bias vector."""
        with self._lock:
            return self._accel_bias.copy()

    def reset_biases(self):
        """Reset IMU biases to zero."""
        with self._lock:
            self._gyro_bias = np.zeros(3)
            self._accel_bias = np.zeros(3)
            logger.info("IMU biases reset to zero")

    def correct_velocity(self, v_visual: np.ndarray, alpha: float = 0.3):
        with self._lock:
            blended = (1.0 - alpha) * self._velocity + alpha * v_visual
            blen_mag = np.linalg.norm(blended)
            if blen_mag > self.MAX_VELOCITY:
                blended = blended * (self.MAX_VELOCITY / blen_mag)
            self._velocity = blended

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        logger.info("IMU pre-integrator stopped")