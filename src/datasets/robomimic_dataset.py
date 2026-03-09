import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T

class RobomimicSE3Dataset(Dataset):
    """
    Research-grade SE(3) dataset for Robomimic HDF5 files.
    Optimized for memory efficiency and resource management.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.index_map = []
        with h5py.File(file_path, 'r') as f:
            demos = list(f['data'].keys())
            for demo in demos:
                num_steps = f[f'data/{demo}/states'].shape[0]
                for step in range(num_steps):
                    self.index_map.append((demo, step))
        
        self.f = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.file_path, 'r')

        demo, step = self.index_map[idx]
        
        # 1. Image Observation (Normalized to float32)
        img = self.f[f'data/{demo}/obs/agentview_image'][step]
        img = (img.transpose(2, 0, 1) / 255.0).astype(np.float32)
        img_tensor = torch.from_numpy(img)
        img_tensor = self.transform(img_tensor)
        
        horizon = 5
        max_step = self.f[f'data/{demo}/states'].shape[0] - 1
        target_step = min(step + horizon, max_step)
        
        # 2. EE Pose [x, y, z, qx, qy, qz, qw]
        pos = self.f[f'data/{demo}/obs/robot0_eef_pos'][target_step]
        quat = self.f[f'data/{demo}/obs/robot0_eef_quat'][target_step]
        
        # 3. Convert to 6D (Position + Axis-Angle)
        rot_vec = R.from_quat(quat).as_rotvec()
        pose_6d = np.concatenate([pos, rot_vec]).astype(np.float32)
        
        return {
            "image": img_tensor, 
            "pose": torch.from_numpy(pose_6d)
        }

    def __del__(self):
        if getattr(self, 'f', None) is not None:
            self.f.close()
