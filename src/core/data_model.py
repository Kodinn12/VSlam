"""Shared typed data structures for SLAM module boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    """Camera pose and optional tracking metadata exchanged between modules."""

    matrix: np.ndarray
    frame_id: int | None = None
    inliers: int | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"PoseEstimate.matrix must be (4, 4), got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("PoseEstimate.matrix contains non-finite values")
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class KeypointSet:
    """Validated 2D keypoints with optional descriptors and scores."""

    keypoints: np.ndarray
    descriptors: np.ndarray | None = None
    scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 2)
        if not np.all(np.isfinite(keypoints)):
            raise ValueError("KeypointSet.keypoints contains non-finite values")
        object.__setattr__(self, "keypoints", keypoints)
        if self.scores is not None:
            scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
            if len(scores) != len(keypoints):
                raise ValueError("KeypointSet.scores length must match keypoints")
            object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class GaussianBubbleBatch:
    """Batch of Gaussian map bubbles with aligned point, covariance, weight, and color arrays."""

    mu: np.ndarray
    Sigma: np.ndarray
    weight: np.ndarray
    color: np.ndarray

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=np.float32).reshape(-1, 3)
        Sigma = np.asarray(self.Sigma, dtype=np.float32).reshape(-1, 3, 3)
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        color = np.asarray(self.color, dtype=np.float32).reshape(-1, 3)
        n = len(mu)
        if len(Sigma) != n or len(weight) != n or len(color) != n:
            raise ValueError(
                "GaussianBubbleBatch arrays must have matching first dimensions: "
                f"mu={len(mu)}, Sigma={len(Sigma)}, weight={len(weight)}, color={len(color)}"
            )
        finite = (
            np.all(np.isfinite(mu), axis=1)
            & np.all(np.isfinite(Sigma.reshape(n, -1)), axis=1)
            & np.isfinite(weight)
            & np.all(np.isfinite(color), axis=1)
        )
        if not np.all(finite):
            raise ValueError("GaussianBubbleBatch contains non-finite values")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "Sigma", Sigma)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "color", np.clip(color, 0.0, 1.0).astype(np.float32))

    def __len__(self) -> int:
        """Return the number of bubbles in the batch."""
        return len(self.mu)

    @classmethod
    def from_arrays(
        cls,
        mu: Any,
        weight: Any,
        color: Any,
        Sigma: Any,
        *,
        to_numpy,
    ) -> "GaussianBubbleBatch":
        """Build a validated batch from NumPy/CuPy-like arrays."""
        return cls(
            mu=to_numpy(mu),
            Sigma=to_numpy(Sigma),
            weight=to_numpy(weight),
            color=to_numpy(color),
        )

