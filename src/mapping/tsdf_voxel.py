"""CuPy TSDF voxel grid with sliding window and threaded integration."""

import numpy as np
import threading
from queue import Queue
from ..utils.logger import get_logger
logger = get_logger(__name__)
from ..utils.cupy_utils import cupy_manager, cp, USE_CUPY, batch_matrix_multiply, batch_matrix_inverse
from ..utils.array_utils import to_numpy_safe
xp = cp

def _check_pyvista():
    try:
        import pyvista as pv
        return True, pv
    except ImportError:
        return False, None

class CupyVoxelGrid:
    def __init__(self, voxel_length=0.008, sdf_trunc=0.040, max_depth=3.0,
                 grid_size=(256,256,256), origin=(-1.0,-1.0,-0.5), weight_max=200.0):
        # V57: Reduced grid_size to 256^3 (~540MB) from 320^3 (~1.05GB) for 6GB GPUs
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc
        self.max_depth = max_depth
        self.grid_size = grid_size
        # GPU zone: use CuPy arrays for voxel grid
        self.origin = xp.array(origin, dtype=xp.float32)
        self.weight_max = weight_max
        total = grid_size[0] * grid_size[1] * grid_size[2]
        self.sdf_grid = xp.ones(total, dtype=xp.float32) * sdf_trunc
        self.weight_grid = xp.zeros(total, dtype=xp.float32)
        self.color_grid = xp.zeros((total, 3), dtype=xp.float32)
        self._init_voxel_coords()
        self._first_frame = True
        logger.info(f"Voxel grid: {grid_size}, weight_max={weight_max}")

    def _init_voxel_coords(self):
        # GPU zone: use CuPy for voxel coordinate generation
        nx, ny, nz = self.grid_size
        x = xp.arange(nx, dtype=xp.float32) * self.voxel_length + self.origin[0]
        y = xp.arange(ny, dtype=xp.float32) * self.voxel_length + self.origin[1]
        z = xp.arange(nz, dtype=xp.float32) * self.voxel_length + self.origin[2]
        xv, yv, zv = xp.meshgrid(x, y, z, indexing='ij')
        self.voxel_coords = xp.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)

    def _maybe_reorigin(self, cam_pos_world):
        """Sliding TSDF window - keep grid centered on camera. See naaaap_modified.py lines ~3101+ for full implementation."""
        try:
            nx, ny, nz = self.grid_size
            threshold  = 0.20
            steps_f    = (cam_pos_world - self.origin) / self.voxel_length

            shifts = [0, 0, 0]
            sizes  = [nx, ny, nz]
            for ax in range(3):
                s  = float(steps_f[ax])
                lo = threshold * sizes[ax]
                hi = (1.0 - threshold) * sizes[ax]
                if s < lo:
                    shifts[ax] = int(lo - s) + 1
                elif s > hi:
                    shifts[ax] = -(int(s - hi) + 1)

            if all(sh == 0 for sh in shifts):
                return

            sdf_3d    = self.sdf_grid.reshape(nx, ny, nz)
            wgt_3d    = self.weight_grid.reshape(nx, ny, nz)
            col_3d    = self.color_grid.reshape(nx, ny, nz, 3)
            sdf_fill  = float(self.sdf_trunc)

            for ax, sh in enumerate(shifts):
                if sh == 0:
                    continue
                sdf_3d = xp.roll(sdf_3d, sh, axis=ax)
                wgt_3d = xp.roll(wgt_3d, sh, axis=ax)
                col_3d = xp.roll(col_3d, sh, axis=ax)
                slab = [slice(None), slice(None), slice(None)]
                if sh > 0:
                    slab[ax] = slice(0, sh)
                else:
                    slab[ax] = slice(sizes[ax] + sh, sizes[ax])
                sdf_3d[tuple(slab)] = sdf_fill
                wgt_3d[tuple(slab)] = 0.0
                col_3d[tuple(slab)] = 0.0
                self.origin[ax] -= sh * self.voxel_length

            self.sdf_grid    = sdf_3d.reshape(-1)
            self.weight_grid = wgt_3d.reshape(-1)
            self.color_grid  = col_3d.reshape(-1, 3)
            self._init_voxel_coords()
        except Exception:
            pass

    def integrate_frame(self, depth, pose, image, K):
        """Integrate a depth frame into the TSDF. See naaaap_modified.py lines ~3156+ for full implementation."""
        if depth is None:
            return False
        try:
            if self._first_frame:
                t_cam = pose[:3, 3]
                nx, ny, nz = self.grid_size
                target_cam_voxel = np.array([nx * 0.50,
                                             ny * 0.50,
                                             nz * 0.15], dtype=np.float32)
                origin_np = (t_cam.astype(np.float32)
                             - target_cam_voxel * self.voxel_length)
                self.origin[:] = xp.asarray(origin_np)
                self._init_voxel_coords()
                self._first_frame = False

            dg  = depth if isinstance(depth, xp.ndarray) else xp.asarray(depth)
            ig  = (image if isinstance(image, xp.ndarray) else xp.asarray(image))
            pg  = xp.asarray(pose, dtype=xp.float32)
            Kg  = xp.asarray(K,    dtype=xp.float32)
            h, w = dg.shape
            fx, fy = float(Kg[0, 0]), float(Kg[1, 1])
            cx, cy = float(Kg[0, 2]), float(Kg[1, 2])
            R_wc = pg[:3, :3]; t_wc = pg[:3, 3]

            self._maybe_reorigin(t_wc)

            max_d  = float(self.max_depth)
            min_d  = 0.05
            ds = xp.array([min_d, max_d], dtype=xp.float32)
            us = xp.array([0.0, float(w - 1)], dtype=xp.float32)
            vs = xp.array([0.0, float(h - 1)], dtype=xp.float32)
            dg3, ug3, vg3 = xp.meshgrid(ds, us, vs, indexing='ij')
            dg3f = dg3.ravel(); ug3f = ug3.ravel(); vg3f = vg3.ravel()
            frust_cam_g  = xp.stack(
                [(ug3f - cx) / fx * dg3f,
                 (vg3f - cy) / fy * dg3f,
                 dg3f], axis=1).astype(xp.float32)
            frust_world  = (R_wc @ frust_cam_g.T).T + t_wc
            lo = frust_world.min(axis=0) - self.voxel_length
            hi = frust_world.max(axis=0) + self.voxel_length

            vc = self.voxel_coords
            aabb_mask = (
                (vc[:, 0] >= lo[0]) & (vc[:, 0] <= hi[0]) &
                (vc[:, 1] >= lo[1]) & (vc[:, 1] <= hi[1]) &
                (vc[:, 2] >= lo[2]) & (vc[:, 2] <= hi[2]))
            aabb_idx = xp.where(aabb_mask)[0]
            if len(aabb_idx) == 0:
                return True
            vc_sub = vc[aabb_idx]

            pts_cam = (vc_sub - t_wc) @ R_wc
            z_cam   = pts_cam[:, 2]
            valid_z = (z_cam > 0.1) & (z_cam < self.max_depth)
            if not xp.any(valid_z):
                return True
            u = (pts_cam[:, 0] * fx / (z_cam + 1e-12)) + cx
            v = (pts_cam[:, 1] * fy / (z_cam + 1e-12)) + cy
            in_img = (u >= 0) & (u <= w-2) & (v >= 0) & (v <= h-2) & valid_z
            vi     = xp.where(in_img)[0]
            if len(vi) == 0:
                return True
            uf = u[vi]; vf = v[vi]; zf = z_cam[vi]
            u0 = xp.floor(uf).astype(xp.int32); v0 = xp.floor(vf).astype(xp.int32)
            u1 = u0 + 1; v1 = v0 + 1; du = uf - u0; dv = vf - v0
            d00 = dg[v0, u0]; d01 = dg[v0, u1]
            d10 = dg[v1, u0]; d11 = dg[v1, u1]
            d_val = ((1-du)*(1-dv)*d00 + du*(1-dv)*d01 +
                     (1-du)*dv*d10 + du*dv*d11)
            c00 = ig[v0, u0, :]; c01 = ig[v0, u1, :]
            c10 = ig[v1, u0, :]; c11 = ig[v1, u1, :]
            c_val = ((1-du[:, None])*(1-dv[:, None])*c00 +
                      du[:, None]*(1-dv[:, None])*c01 +
                     (1-du[:, None])*dv[:, None]*c10 +
                      du[:, None]*dv[:, None]*c11)
            sdf = d_val - zf
            vm  = (d_val > 0.1) & (sdf >= -self.sdf_trunc)
            fi  = aabb_idx[vi[vm]]
            fs  = xp.clip(sdf[vm], -self.sdf_trunc, self.sdf_trunc)
            fc  = c_val[vm] / 255.0
            cw  = self.weight_grid[fi]; cs = self.sdf_grid[fi]; cc = self.color_grid[fi]
            nw  = xp.minimum(cw + 1.0, self.weight_max)
            self.sdf_grid[fi]   = (cs * cw + fs) / nw
            self.color_grid[fi] = (cc * cw[:, None] + fc) / nw[:, None]
            self.weight_grid[fi] = nw
            return True
        except Exception:
            return False

    def extract_mesh_open3d(self, min_weight=0.5):
        """Extract mesh from TSDF. See naaaap_modified.py lines ~3291+ for full implementation."""
        has_o3d, o3d = _check_open3d()
        if not has_o3d:
            return None
        try:
            sdf_cpu = to_numpy_safe(self.sdf_grid).reshape(self.grid_size)
            w_cpu   = to_numpy_safe(self.weight_grid).reshape(self.grid_size)
            sdf_cpu[w_cpu < min_weight] = self.sdf_trunc
            if np.abs(sdf_cpu).min() > self.sdf_trunc*0.9:
                return None
            from skimage.measure import marching_cubes
            verts, faces, _, _ = marching_cubes(sdf_cpu, level=0.0,
                                                spacing=(self.voxel_length,)*3)
            verts += to_numpy_safe(self.origin)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            return mesh
        except Exception:
            return None

    def extract_mesh_pyvista(self, min_weight=0.5):
        """Extract mesh from TSDF (format compatible with PyVista)."""
        # Note: This method doesn't actually require pyvista installed to return arrays
        try:
            sdf_cpu = to_numpy_safe(self.sdf_grid).reshape(self.grid_size)
            w_cpu   = to_numpy_safe(self.weight_grid).reshape(self.grid_size)
            sdf_cpu[w_cpu < min_weight] = self.sdf_trunc
            if np.abs(sdf_cpu).min() > self.sdf_trunc*0.9:
                return None
            from skimage.measure import marching_cubes
            verts, faces, _, _ = marching_cubes(sdf_cpu, level=0.0,
                                                spacing=(self.voxel_length,)*3)
            verts += to_numpy_safe(self.origin)
            return verts, faces
        except Exception:
            return None

    def get_statistics(self):
        try:
            vv = int(xp.sum(self.weight_grid > 0))
            aw = float(xp.mean(self.weight_grid[self.weight_grid > 0])) if vv > 0 else 0.0
            return {'total_voxels': self.weight_grid.size, 'valid_voxels': vv,
                    'fill_ratio': vv / self.weight_grid.size, 'avg_weight': aw}
        except Exception:
            return {}

class ThreadedCupyVoxelManager:
    def __init__(self, config, K, baseline):
        self.config = config
        self.K = K
        self.baseline = baseline
        self.keyframes = []
        if config.get("enable_tsdf_voxels", True):
            # V57: Reduced grid_size to 256^3 for 6GB GPUs
            gs = config.get("voxel_grid_size", (256,256,256))
            self.voxel_grid = CupyVoxelGrid(
                voxel_length=config["tsdf_voxel_length"],
                sdf_trunc=config["tsdf_sdf_trunc"],
                max_depth=config["tsdf_max_depth"],
                grid_size=gs, origin=(-1.5,-1.5,-0.5))
        else:
            self.voxel_grid = None
        self.use_threading = config.get("use_threaded_integration", True)
        self.integration_queue = Queue(maxsize=2)
        self.integration_thread = None
        self.running = False
        self._lock = threading.Lock()
        if self.use_threading:
            self._start_thread()
        logger.info("Threaded voxel manager initialized")

    def _start_thread(self):
        self.running = True
        self.integration_thread = threading.Thread(target=self._worker, daemon=True)
        self.integration_thread.start()

    def _worker(self):
        while self.running:
            try:
                item = self.integration_queue.get(timeout=0.05)
            except Exception:
                continue
            if item is None:
                self.integration_queue.task_done()
                break
            if item == "REBUILD":
                try:
                    with self._lock:
                        kfs_snap = list(self.keyframes)
                        if self.voxel_grid is not None:
                            # V57: Use 256^3 for 6GB GPUs
                            gs = self.config.get("voxel_grid_size", (256,256,256))
                            self.voxel_grid = CupyVoxelGrid(
                                voxel_length=self.config["tsdf_voxel_length"],
                                sdf_trunc=self.config["tsdf_sdf_trunc"],
                                max_depth=self.config["tsdf_max_depth"],
                                grid_size=gs, origin=(-1.5,-1.5,-0.5))
                    for kf in kfs_snap:
                        try:
                            with self._lock:
                                self.voxel_grid.integrate_frame(kf.depth, kf.pose, kf.image, kf.intrinsics)
                        except Exception:
                            continue
                except Exception as e:
                    logger.error(f"REBUILD failed: {e}")
                finally:
                    self.integration_queue.task_done()
                continue
            try:
                self._do_integrate(*item)
            finally:
                self.integration_queue.task_done()

    def _do_integrate(self, image, depth, pose, intrinsic):
        try:
            with self._lock:
                if self.voxel_grid:
                    return self.voxel_grid.integrate_frame(depth, pose, image, intrinsic)
                return False
        except Exception:
            return False

    def integrate_frame(self, image, depth, pose, intrinsic):
        if self.voxel_grid is None:
            return False
        if self.use_threading:
            try:
                if self.integration_queue.full():
                    self.integration_queue.get_nowait()
                # PASS GPU TENSORS DIRECTLY - NO D2H TRANSFER HERE
                self.integration_queue.put_nowait((image, depth, pose, intrinsic))
                return True
            except Exception:
                return False
        return self._do_integrate(image, depth, pose, intrinsic)

    def cache_keyframe(self, kf):
        self.keyframes.append(kf)
        if len(self.keyframes) > self.config["max_keyframes_in_buffer"]:
            self.keyframes.pop(0)

    # Open3D removed - use get_voxel_mesh_pyvista only

    def get_voxel_mesh_pyvista(self, min_weight=1.0):
        if not self.voxel_grid:
            return None, None, None
        mesh_data = self.voxel_grid.extract_mesh_pyvista(min_weight)
        if mesh_data is None:
            return None, None, None
        verts, tris = mesh_data
        colors = np.ones((len(verts), 3)) * 128  # Default gray color
        return verts, tris, colors

    def get_statistics(self):
        return self.voxel_grid.get_statistics() if self.voxel_grid else {}

    def reintegrate_map(self):
        if not self.voxel_grid:
            return
        if self.use_threading and self.running and self.integration_thread is not None:
            while not self.integration_queue.empty():
                try:
                    self.integration_queue.get_nowait()
                    self.integration_queue.task_done()
                except Exception:
                    break
            try:
                self.integration_queue.put("REBUILD", timeout=1.0)
                logger.info("REBUILD queued for background worker")
            except Exception as e:
                logger.error(f"Could not queue REBUILD: {e}")
            return
        # synchronous fallback
        with self._lock:
            kfs_snap = list(self.keyframes)
            gs = self.config.get("voxel_grid_size", (320,320,320))
            self.voxel_grid = CupyVoxelGrid(
                voxel_length=self.config["tsdf_voxel_length"],
                sdf_trunc=self.config["tsdf_sdf_trunc"],
                max_depth=self.config["tsdf_max_depth"],
                grid_size=gs, origin=(-1.5,-1.5,-0.5))
        for kf in kfs_snap:
            try:
                with self._lock:
                    self.voxel_grid.integrate_frame(kf.depth, kf.pose, kf.image, kf.intrinsics)
            except Exception:
                continue
        logger.info("Synchronous REBUILD complete")

    def shutdown(self):
        if self.use_threading and self.running:
            self.running = False
            self.integration_queue.put(None)
            if self.integration_thread:
                self.integration_thread.join(timeout=2.0)
