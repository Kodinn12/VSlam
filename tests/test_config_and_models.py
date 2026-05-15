"""Smoke tests for config coercion and shared SLAM data models."""

import numpy as np

from src.core.config import coerce_config_types
from src.core.data_model import GaussianBubbleBatch, KeypointSet, PoseEstimate
from src.mapping.gaussian_bubbles import ChunkedBubbleMap


def test_config_numeric_strings_are_coerced():
    """YAML string scalars become native bool/int/float values."""
    cfg = coerce_config_types(
        {
            "ba_window_size": "6",
            "ba_lambda_init": "1e-3",
            "enable_bundle_adjustment": "true",
            "nested": {"threshold": "0.25"},
        }
    )
    assert cfg["ba_window_size"] == 6
    assert isinstance(cfg["ba_window_size"], int)
    assert cfg["ba_lambda_init"] == 1e-3
    assert cfg["enable_bundle_adjustment"] is True
    assert cfg["nested"]["threshold"] == 0.25


def test_data_models_validate_shapes():
    """Shared dataclasses reject boundary shape drift early."""
    PoseEstimate(np.eye(4))
    KeypointSet(np.array([[1.0, 2.0]], dtype=np.float32), scores=np.array([0.9]))
    batch = GaussianBubbleBatch(
        mu=np.zeros((2, 3)),
        Sigma=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        weight=np.ones(2),
        color=np.ones((2, 3)),
    )
    assert len(batch) == 2


def test_bubble_map_accepts_batch_and_array_inputs():
    """ChunkedBubbleMap normalizes both legacy arrays and GaussianBubbleBatch."""
    cfg = {
        "bubble_cuda": False,
        "chunk_size": 4.0,
        "local_bubble_buffer_size": 32,
        "max_new_bubbles_per_frame": 32,
        "enable_chunk_offloading": False,
    }
    K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    bubble_map = ChunkedBubbleMap(K, baseline=0.075, config=cfg)
    try:
        mu = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.1]], dtype=np.float32)
        Sigma = np.tile(np.eye(3, dtype=np.float32) * 0.01, (2, 1, 1))
        weight = np.ones(2, dtype=np.float32)
        color = np.full((2, 3), 0.5, dtype=np.float32)

        bubble_map.add_bubbles(mu, weight, color, Sigma)
        bubble_map.add_bubbles(GaussianBubbleBatch(mu=mu + 0.2, Sigma=Sigma, weight=weight, color=color))
        assert len(bubble_map) > 0
    finally:
        bubble_map.shutdown()


def test_chunk_key_uses_numpy_for_numpy_points_even_in_gpu_mode():
    """CPU-side chunk partitioning remains valid when the map runs in GPU mode."""
    cfg = {
        "bubble_cuda": False,
        "chunk_size": 4.0,
        "local_bubble_buffer_size": 32,
        "enable_chunk_offloading": False,
    }
    K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    bubble_map = ChunkedBubbleMap(K, baseline=0.075, config=cfg)
    try:
        bubble_map.shutdown()
        bubble_map.use_gpu = True
        pts = np.array([[0.0, 0.0, 1.0], [4.1, -0.2, 8.0]], dtype=np.float32)
        keys = bubble_map._world_to_chunk_key(pts)
        assert isinstance(keys, np.ndarray)
        assert keys.shape == (2,)
        assert keys.dtype == np.int64
    finally:
        bubble_map.shutdown()
