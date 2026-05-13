"""Spawn-isolated PyVista visualizer for SLAM bubble rendering."""

import importlib.util
import logging
import multiprocessing as mp
import queue
import time

import numpy as np

USE_PYVISTA = importlib.util.find_spec("pyvista") is not None

logger = logging.getLogger(__name__)


def _to_numpy(arr):
    """Convert CuPy / torch / NumPy arrays to NumPy float32 arrays."""
    if arr is None:
        return None
    if hasattr(arr, "get"):
        return arr.get().astype(np.float32)
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy().astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def _to_pyvista(arr):
    """
    Convert OpenCV camera coordinates to PyVista display coordinates.

    OpenCV: X right, Y down, Z into scene.
    PyVista/VTK display: X right, Y up, Z toward viewer.
    """
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    out = arr.copy()
    out[:, 1] = -arr[:, 1]
    out[:, 2] = -arr[:, 2]
    return out


def _vtk_faces(tris):
    """Convert Nx3 triangle indices into VTK faces format."""
    tris = np.asarray(tris, dtype=np.int32)
    if tris.ndim != 2 or tris.shape[1] != 3:
        return tris.flatten().astype(np.int32)

    n = tris.shape[0]
    faces = np.empty(n * 4, dtype=np.int32)
    faces[0::4] = 3
    faces[1::4] = tris[:, 0]
    faces[2::4] = tris[:, 1]
    faces[3::4] = tris[:, 2]
    return faces


def _pv_is_closed(plotter):
    """Best-effort guard for a user-closed PyVista/VTK window."""
    try:
        if getattr(plotter, "_closed", False):
            return True
        interactor = getattr(plotter, "iren", None)
        if interactor is not None and getattr(interactor, "initialized", True) is False:
            return True
    except Exception:
        return True
    return False


def _remove_actor(plotter, actors, name):
    actor = actors.pop(name, None)
    if actor is None:
        return
    try:
        plotter.remove_actor(actor)
    except Exception:
        pass


def _compute_follow_camera(points, previous=None, smooth=0.1):
    """
    Compute an over-the-shoulder camera for PyVista display coordinates.

    Display convention after _to_pyvista:
    X = right, Y = up, Z = toward viewer. Scans extend mostly toward -Z,
    so the camera should sit at +Z and look into the reconstruction.
    """
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    radius = max(np.linalg.norm(points - center, axis=1).max(), 0.5)
    dist = radius * 2.2
    target_pos = np.array([
        center[0],
        center[1] + dist * 0.35,
        center[2] + dist,
    ], dtype=np.float32)
    target = (target_pos, center, (0, 1, 0))

    if previous is None:
        return target

    return (
        (1 - smooth) * np.asarray(previous[0], dtype=np.float32) + smooth * target_pos,
        (1 - smooth) * np.asarray(previous[1], dtype=np.float32) + smooth * center,
        (0, 1, 0),
    )


def _viz_process_fn(q, window_size=(1280, 720), title="SLAM 3D", point_size=2.0):
    import os

    import numpy as np
    import pyvista as pv

    os.environ["QT_API"] = "none"
    os.environ["PYVISTA_OFF_SCREEN"] = "false"

    print("[PyVista] subprocess starting")
    try:
        plotter = pv.Plotter(window_size=window_size, title=title, off_screen=False)
        plotter.set_background("black")
        plotter.enable_anti_aliasing()
        plotter.add_axes()
        plotter.camera_position = [(0, 2, 6), (0, 0, 0), (0, 1, 0)]
        plotter.show(interactive_update=True, auto_close=False)
        print("[PyVista] window ready")
    except Exception as e:
        print(f"[PyVista] init failed: {e}")
        return

    actors = {}
    cam_pos = None
    smooth = 0.1

    while True:
        if _pv_is_closed(plotter):
            break

        try:
            data = q.get(timeout=0.05)
        except KeyboardInterrupt:
            break
        except queue.Empty:
            try:
                plotter.update()
            except Exception:
                break
            continue
        except Exception as e:
            print(f"[PyVista] queue read error: {e}")
            break

        if data is None:
            break

        bpts = data.get("bubble_pts")
        bcols = data.get("bubble_colors")
        vox_verts = data.get("voxel_verts")
        vox_faces = data.get("voxel_faces")
        vox_colors = data.get("voxel_colors")
        traj = data.get("trajectory")

        _remove_actor(plotter, actors, "bc")
        if bpts is not None and len(bpts) > 0:
            try:
                bpts = np.asarray(bpts, dtype=np.float32)
                valid = np.all(np.isfinite(bpts), axis=1)
                bpts = bpts[valid]
                if bcols is not None:
                    bcols = np.asarray(bcols, dtype=np.uint8)[valid]

                if len(bpts) > 0:
                    if bcols is None or len(bcols) != len(bpts):
                        depth = -bpts[:, 2]
                        dmin, dmax = depth.min(), depth.max()
                        drange = max(dmax - dmin, 1e-5)
                        norm = (depth - dmin) / drange
                        bcols = np.column_stack([
                            (255 * (1.0 - norm)).astype(np.uint8),
                            (255 * norm).astype(np.uint8),
                            np.full(len(norm), 50, dtype=np.uint8),
                        ])

                    cloud = pv.PolyData(bpts)
                    cloud["colors"] = bcols
                    actors["bc"] = plotter.add_mesh(
                        cloud,
                        scalars="colors",
                        rgb=True,
                        style="points",
                        point_size=point_size,
                        render_points_as_spheres=True,
                        lighting=False,
                    )

                    cam_pos = _compute_follow_camera(bpts, previous=cam_pos, smooth=smooth)
                    plotter.camera_position = cam_pos

            except Exception as e:
                print(f"[PyVista] bubble render error: {e}")

        _remove_actor(plotter, actors, "voxels")
        if vox_verts is not None and len(vox_verts) > 0 and vox_faces is not None:
            try:
                vox_verts = np.asarray(vox_verts, dtype=np.float32)
                vox_faces = _vtk_faces(vox_faces)
                mesh = pv.PolyData(vox_verts, vox_faces)
                if vox_colors is not None:
                    vox_colors = np.asarray(vox_colors, dtype=np.uint8)
                    mesh["colors"] = vox_colors
                    actors["voxels"] = plotter.add_mesh(
                        mesh,
                        scalars="colors",
                        rgb=True,
                        opacity=0.7,
                        smooth_shading=True,
                    )
                else:
                    actors["voxels"] = plotter.add_mesh(
                        mesh,
                        color="lightblue",
                        opacity=0.7,
                        smooth_shading=True,
                    )
            except Exception as e:
                print(f"[PyVista] voxel render error: {e}")

        _remove_actor(plotter, actors, "traj")
        if traj is not None and len(traj) > 1:
            try:
                traj = np.asarray(traj, dtype=np.float32)
                poly = pv.lines_from_points(traj)
                actors["traj"] = plotter.add_mesh(poly, color="red", line_width=3)
            except Exception as e:
                print(f"[PyVista] traj render error: {e}")

        try:
            plotter.update()
            plotter.render()
        except Exception:
            break

    try:
        plotter.close()
    except Exception:
        pass

    print("[PyVista] subprocess closed")


class PyVistaVisualizer:
    def __init__(
        self,
        window_size=(1280, 720),
        title="SLAM 3D Reconstruction",
        max_points=50000,
        point_size=2.0,
    ):
        self.trajectory_points = []
        self._proc = None
        self._q = None
        self.max_points = int(max_points)
        self.point_size = float(point_size)

        if not USE_PYVISTA:
            print("[PyVista] not installed")
            return

        try:
            ctx = mp.get_context("spawn")
            self._q = ctx.Queue(maxsize=2)
            self._proc = ctx.Process(
                target=_viz_process_fn,
                args=(self._q, window_size, title, self.point_size),
                daemon=False,
            )
            self._proc.start()
            logger.info(f"[Viz] subprocess started PID={self._proc.pid}")
            time.sleep(2.0)
        except Exception as e:
            logger.error(f"[Viz] failed start: {e}")

    def update_visualization(
        self,
        bubble_map=None,
        voxel_manager=None,
        relocalizer=None,
        current_pose=None,
        show_bubbles=True,
        show_voxels=True,
        show_trajectory=True,
    ):
        if self._proc is None or not self._proc.is_alive():
            return

        render_data = {}

        if show_bubbles and bubble_map is not None:
            try:
                if hasattr(bubble_map, "get_point_cloud_pyvista"):
                    pts, colors = bubble_map.get_point_cloud_pyvista(max_points=self.max_points)
                    pts = _to_numpy(pts)
                    colors = np.asarray(colors, dtype=np.uint8) if colors is not None else None
                else:
                    pts, colors, _weights = bubble_map.get_full_point_cloud()
                    pts = _to_numpy(pts)
                    colors = _to_numpy(colors) if colors is not None else None
                    if colors is not None:
                        colors = (colors * 255.0).clip(0, 255).astype(np.uint8)

                if pts is not None and len(pts) > 0:
                    render_data["bubble_pts"] = _to_pyvista(pts)
                    if colors is not None and len(colors) == len(pts):
                        render_data["bubble_colors"] = colors
                else:
                    logger.debug("[Viz] no bubble points to render")
            except Exception as e:
                logger.error(f"[Viz] bubble extraction error: {e}")

        if show_voxels and voxel_manager is not None:
            try:
                verts, tris, colors = voxel_manager.get_voxel_mesh_pyvista()
                if verts is not None and len(verts) > 0 and tris is not None:
                    verts = _to_numpy(verts)
                    render_data["voxel_verts"] = _to_pyvista(verts)
                    render_data["voxel_faces"] = _vtk_faces(tris)
                    if colors is not None:
                        render_data["voxel_colors"] = np.asarray(colors, dtype=np.uint8)
            except Exception as e:
                logger.error(f"[Viz] voxel extraction error: {e}")

        if show_trajectory and current_pose is not None:
            self.trajectory_points.append(current_pose[:3, 3].copy())
            if len(self.trajectory_points) > 10000:
                self.trajectory_points = self.trajectory_points[-10000:]
            if len(self.trajectory_points) > 1:
                render_data["trajectory"] = _to_pyvista(
                    np.array(self.trajectory_points, dtype=np.float32)
                )

        try:
            self._q.put_nowait(render_data)
        except queue.Full:
            logger.debug("[Viz] dropped visualization frame because queue is full")
        except Exception as e:
            logger.debug(f"[Viz] visualization queue send failed: {e}")

    def is_active(self):
        return self._proc is not None and self._proc.is_alive()

    def close(self):
        if self._proc is None:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(None)
            except Exception:
                pass
        except Exception:
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        logger.info("[Viz] subprocess terminated")
