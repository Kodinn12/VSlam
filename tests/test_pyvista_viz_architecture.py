"""Tests for the PyVista visualization process boundary helpers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visualisation.pyvista_viz import _compute_follow_camera, _to_pyvista, _vtk_faces


def test_to_pyvista_flips_y_and_z_axes():
    pts = np.array([
        [1.0, 2.0, 3.0],
        [-4.0, -5.0, 6.0],
    ], dtype=np.float32)

    converted = _to_pyvista(pts)

    np.testing.assert_allclose(
        converted,
        np.array([
            [1.0, -2.0, -3.0],
            [-4.0, 5.0, -6.0],
        ], dtype=np.float32),
    )
    np.testing.assert_allclose(pts[:, 1:], np.array([[2.0, 3.0], [-5.0, 6.0]], dtype=np.float32))


def test_vtk_faces_packs_triangle_indices():
    tris = np.array([
        [0, 1, 2],
        [2, 3, 0],
    ], dtype=np.int32)

    faces = _vtk_faces(tris)

    np.testing.assert_array_equal(
        faces,
        np.array([3, 0, 1, 2, 3, 2, 3, 0], dtype=np.int32),
    )


def test_follow_camera_uses_y_up_and_positive_z_viewer_side():
    pts = _to_pyvista(np.array([
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, 5.0],
        [0.0, -1.0, 3.0],
        [0.0, 1.0, 3.0],
    ], dtype=np.float32))

    position, focal_point, view_up = _compute_follow_camera(pts)

    assert view_up == (0, 1, 0)
    assert position[1] > focal_point[1]
    assert position[2] > focal_point[2]
