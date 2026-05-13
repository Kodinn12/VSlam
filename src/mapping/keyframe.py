"""Keyframe data structure."""

from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class Keyframe:
    id: int
    pose: np.ndarray
    image: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    keypoints: np.ndarray = None
    descriptors: np.ndarray = None
    scores: np.ndarray = None
    _gpu_feats: dict = None

    def get_gpu_feats(self, device):
        if self._gpu_feats is not None:
            return self._gpu_feats
        if self.keypoints is None or self.descriptors is None or self.scores is None:
            return None
        self._gpu_feats = {
            'keypoints': torch.from_numpy(self.keypoints).float().to(device),
            'descriptors': torch.from_numpy(self.descriptors).float().to(device),
            'keypoint_scores': torch.from_numpy(self.scores).float().to(device),
            'image_size': torch.tensor(self.image.shape[:2], device=device),
        }
        return self._gpu_feats