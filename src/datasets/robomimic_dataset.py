import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

class RobomimicSE3Dataset(Dataset):
    """
    Research-grade SE(3) dataset for Robomimic HDF5 files.
    Optimized for memory efficiency and resource management.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        try:
            self.f = h5py.File(file_path, 'r')
            self.demos = list(self.f['data'].keys())
        except Exception as e:
            print(f"Error loading HDF5 file at {file_path}: {e}")
            raise

        self.index_map = []
        for demo in self.demos:
            num_steps = self.f[f'data/{demo}/states'].shape[0]
            for step in range(num_steps):
                self.index_map.append((demo, step))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        demo, step = self.index_map[idx]
        
        # 1. Image Observation (Normalized to float32)
        img = self.f[f'data/{demo}/obs/agentview_image'][step]
        img = (img.transpose(2, 0, 1) / 255.0).astype(np.float32) # [C, H, W]
        
        # 2. EE Pose [x, y, z, qx, qy, qz, qw]
        pos = self.f[f'data/{demo}/obs/robot0_eef_pos'][step]
        quat = self.f[f'data/{demo}/obs/robot0_eef_quat'][step]
        
        # 3. Convert to 6D (Position + Axis-Angle)
        # Standard practice for SE(3) manifold regression
        rot_vec = R.from_quat(quat).as_rotvec()
        pose_6d = np.concatenate([pos, rot_vec]).astype(np.float32)
        
        return {
            "image": torch.from_numpy(img),
            "pose": torch.from_numpy(pose_6d)
        }

    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()