import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T

class RobomimicSE3Dataset(Dataset):
    """
    Research-grade SE(3) dataset for Robomimic HDF5 files.
    Optimized for multi-view fusion and action chunking (prediction_horizon=16).
    """
    def __init__(self, file_path, prediction_horizon=16):
        self.file_path = file_path
        self.pred_horizon = prediction_horizon
        
        # Normalization parameters for 6-channel fused images
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
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
        
        # 1. Multi-view Image Fusion (3rd person + wrist)
        img_view = self.f[f'data/{demo}/obs/agentview_image'][step]
        img_hand = self.f[f'data/{demo}/obs/robot0_eye_in_hand_image'][step]
        
        img_view = (img_view.transpose(2, 0, 1) / 255.0).astype(np.float32)
        img_hand = (img_hand.transpose(2, 0, 1) / 255.0).astype(np.float32)
        img_fused = np.concatenate([img_view, img_hand], axis=0)
        
        img_tensor = self.transform(torch.from_numpy(img_fused))
        
        # 2. Future Trajectory Generation (Action Chunking)
        max_step = self.f[f'data/{demo}/states'].shape[0] - 1
        trajectory = []
        
        for i in range(self.pred_horizon):
            t_step = min(step + i, max_step)
            pos = self.f[f'data/{demo}/obs/robot0_eef_pos'][t_step]
            quat = self.f[f'data/{demo}/obs/robot0_eef_quat'][t_step]
            rot_vec = R.from_quat(quat).as_rotvec()
            pose_6d = np.concatenate([pos, rot_vec])
            trajectory.append(pose_6d)
            
        traj_tensor = torch.from_numpy(np.array(trajectory).flatten().astype(np.float32))
        
        return {"image": img_tensor, "pose": traj_tensor}

    def __del__(self):
        if getattr(self, 'f', None) is not None:
            self.f.close()
