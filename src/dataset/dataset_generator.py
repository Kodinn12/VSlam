import h5py
import numpy as np
import os
import time
from typing import Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DatasetGenerator:
    """
    Generates an HDF5 dataset for World Model (WAM) Training.
    Records the sequence: State_t -> Action_t (IMU) -> State_t+1
    """
    def __init__(self, output_dir: str = "dataset_output", run_name: str = None):
        # DISABLED: Folder creation removed per user request
        # self.output_dir = output_dir
        # os.makedirs(self.output_dir, exist_ok=True)
        
        # self.run_name = run_name or f"run_{int(time.time())}"
        # self.h5_path = os.path.join(self.output_dir, f"{self.run_name}.h5")
        
        self.file = h5py.File(self.h5_path, 'w')
        
        # Create expandable datasets
        # We assume typical OAK-D RGB resolution of 400x640 or similar.
        # Maxshape=None allows infinite appending
        self.file.create_dataset("images", shape=(0, 400, 640, 3), maxshape=(None, None, None, 3), dtype=np.uint8)
        self.file.create_dataset("poses", shape=(0, 4, 4), maxshape=(None, 4, 4), dtype=np.float32)
        self.file.create_dataset("imu_actions", shape=(0, 6), maxshape=(None, 6), dtype=np.float32) # dv, dw
        self.file.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
        
        # Optional: Save visible map chunks (State_t) as JSON metadata or serialized
        self.map_group = self.file.create_group("map_states")
        
        self.frame_count = 0
        logger.info(f"Dataset Generator initialized. Saving to: {self.h5_path}")
        
    def step(self, image: np.ndarray, pose: np.ndarray, imu_action: np.ndarray, timestamp: float):
        """
        Record a single step: State_t, Action_t
        """
        # Dynamic resizing based on image shape to handle first frame size
        if self.frame_count == 0:
            h, w, c = image.shape
            self.file["images"].resize((1, h, w, c))
        else:
            h, w, c = image.shape
            self.file["images"].resize((self.frame_count + 1, h, w, c))
            
        self.file["poses"].resize((self.frame_count + 1, 4, 4))
        self.file["imu_actions"].resize((self.frame_count + 1, 6))
        self.file["timestamps"].resize((self.frame_count + 1,))
        
        # Insert data
        self.file["images"][self.frame_count] = image
        self.file["poses"][self.frame_count] = pose
        self.file["imu_actions"][self.frame_count] = imu_action
        self.file["timestamps"][self.frame_count] = timestamp
        
        self.frame_count += 1
        
        # Periodic flush to disk to prevent data loss on crash
        if self.frame_count % 50 == 0:
            self.file.flush()
        
    def save_map_state(self, mu: np.ndarray, chunk_id: np.ndarray):
        """
        Periodically save the state of the active map chunks.
        """
        # Store as separate datasets within the map_states group
        idx = str(self.frame_count)
        grp = self.map_group.create_group(f"frame_{idx}")
        grp.create_dataset("mu", data=mu.astype(np.float32))
        grp.create_dataset("chunk_id", data=chunk_id.astype(np.int32))
        
    def close(self):
        """Clean up and close the HDF5 file."""
        self.file.close()
        logger.info(f"Dataset saved with {self.frame_count} frames.")
