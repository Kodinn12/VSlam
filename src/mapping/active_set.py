import numpy as np
from typing import Tuple, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ActiveSetSelector:
    """
    Selects the active set of map elements (Gaussians/Voxels) based on
    frustum culling, visibility, and distance-based LOD (Level of Detail).
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 1.0)  # 1m^3 chunks
        self.lod_distance = self.config.get('lod_distance', 5.0)  # meters for LOD transition
        logger.info(f"ActiveSetSelector initialized (chunk_size={self.chunk_size}m, lod_distance={self.lod_distance}m)")

    def frustum_culling(self, mu: np.ndarray, pose: np.ndarray, K: np.ndarray, 
                        width: int, height: int, near: float = 0.1, far: float = 50.0) -> np.ndarray:
        """
        Filter 3D points that are outside the camera frustum.
        
        Args:
            mu: (N, 3) array of 3D positions in world frame.
            pose: (4, 4) camera to world transform matrix (T_wc).
            K: (3, 3) camera intrinsics matrix.
            width: Image width.
            height: Image height.
            near: Near clipping plane.
            far: Far clipping plane.
            
        Returns:
            Boolean array of length N indicating visibility.
        """
        if len(mu) == 0:
            return np.array([], dtype=bool)

        # Pose is T_wc (Camera to World). We need T_cw (World to Camera)
        R_wc = pose[:3, :3]
        t_wc = pose[:3, 3]
        
        # Transform points to camera frame: P_cam = R_cw * (P_world - t_wc)
        # Since R_cw = R_wc.T
        P_cam = (mu - t_wc) @ R_wc

        # Z-clipping (behind camera or too far)
        valid_z_mask = (P_cam[:, 2] > near) & (P_cam[:, 2] < far)
        
        # Only project points within valid Z range
        P_proj = P_cam[valid_z_mask]
        
        if len(P_proj) == 0:
            return np.zeros(len(mu), dtype=bool)

        # Project to 2D image plane
        uv_hom = (K @ P_proj.T).T
        uv = uv_hom[:, :2] / (uv_hom[:, 2:3] + 1e-8)
        
        # UV clipping
        valid_uv_mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & \
                        (uv[:, 1] >= 0) & (uv[:, 1] < height)

        # Combine masks
        final_mask = np.zeros(len(mu), dtype=bool)
        valid_z_indices = np.where(valid_z_mask)[0]
        final_mask[valid_z_indices[valid_uv_mask]] = True
        
        return final_mask

    def compute_lod(self, mu: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Compute distance-based Level of Detail.
        
        Args:
            mu: (N, 3) array of 3D positions.
            pose: (4, 4) camera to world transform.
            
        Returns:
            Integer array of LOD levels (0: High, 1: Low).
        """
        if len(mu) == 0:
            return np.array([], dtype=np.int32)
            
        t_wc = pose[:3, 3]
        distances = np.linalg.norm(mu - t_wc, axis=1)
        
        lod = np.zeros(len(mu), dtype=np.int32)
        lod[distances > self.lod_distance] = 1
        
        return lod
