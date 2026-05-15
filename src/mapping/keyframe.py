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

    def get_3d_points(self, intrinsics):
        """Backproject keypoints to 3D using depth and intrinsics."""
        if self.keypoints is None or self.depth is None:
            return None, None
            
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Sample depth at keypoints
        u, v = self.keypoints[:, 0], self.keypoints[:, 1]
        h, w = self.depth.shape
        
        ui, vi = np.clip(u.astype(int), 0, w-1), np.clip(v.astype(int), 0, h-1)
        z = self.depth[vi, ui]
        
        valid = (z > 0.1) & (z < 10.0) & np.isfinite(z)
        if not np.any(valid):
            return None, None
            
        z_v = z[valid]
        u_v = u[valid]
        v_v = v[valid]
        
        x = (u_v - cx) * z_v / fx
        y = (v_v - cy) * z_v / fy
        pts_cam = np.stack([x, y, z_v], axis=1)
        
        # Transform to world coordinates
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        pts_world = (R @ pts_cam.T).T + t
        
        return pts_world, np.where(valid)[0]