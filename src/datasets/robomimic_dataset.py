import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T

class RobomimicSE3Dataset(Dataset):
    """
    Research-grade SE(3) dataset for Robomimic HDF5 files.
    [DP Standard] Employs obs_horizon=2 (temporal context) and strict [-1, 1] Min-Max Normalization.
    """
    def __init__(self, file_path, prediction_horizon=16, obs_horizon=2):
        self.file_path = file_path
        self.pred_horizon = prediction_horizon
        self.obs_horizon = obs_horizon
        
        # Normalization parameters for 12-channel fused images (2 frames * 6 channels)
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406] * 4, 
                        std=[0.229, 0.224, 0.225] * 4)
        ])
        
        self.index_map = []
        with h5py.File(file_path, 'r') as f:
            demos = list(f['data'].keys())
            for demo in demos:
                num_steps = f[f'data/{demo}/states'].shape[0]
                for step in range(num_steps):
                    self.index_map.append((demo, step))
        
        self.f = None
        # [DP Standard] Compute dynamic global bounds for strict [-1, 1] normalization
        self.stats = self._compute_action_stats()

    def _compute_action_stats(self):
        print("[REFLEX] Pre-computing Action Min-Max Normalization Stats for [-1, 1] scaling...")
        sample_indices = np.linspace(0, len(self.index_map)-1, num=1000, dtype=int)
        all_traj = []
        
        with h5py.File(self.file_path, 'r') as f:
            for idx in sample_indices:
                demo, step = self.index_map[idx]
                max_step = f[f'data/{demo}/states'].shape[0] - 1
                curr_pos = f[f'data/{demo}/obs/robot0_eef_pos'][step]
                curr_rot_vec = R.from_quat(f[f'data/{demo}/obs/robot0_eef_quat'][step]).as_rotvec()
                
                trajectory = []
                for i in range(self.pred_horizon):
                    t_step = min(step + i, max_step)
                    pos = f[f'data/{demo}/obs/robot0_eef_pos'][t_step]
                    rot_vec = R.from_quat(f[f'data/{demo}/obs/robot0_eef_quat'][t_step]).as_rotvec()
                    
                    trajectory.append(np.concatenate([pos - curr_pos, rot_vec - curr_rot_vec]))
                all_traj.append(np.array(trajectory).flatten())
                
        all_traj = np.array(all_traj) # (1000, 96)
        return {
            "min": np.min(all_traj, axis=0) - 1e-5, # Added epsilon to prevent division by zero
            "max": np.max(all_traj, axis=0) + 1e-5
        }

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.file_path, 'r')

        demo, step = self.index_map[idx]
        
        # 1. [DP Standard] Multi-view & Multi-frame Fusion (obs_horizon=2)
        step_t0 = max(0, step - 1)
        step_t1 = step
        
        imgs = []
        for s in [step_t0, step_t1]:
            img_view = (self.f[f'data/{demo}/obs/agentview_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32)
            img_hand = (self.f[f'data/{demo}/obs/robot0_eye_in_hand_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32)
            imgs.extend([img_view, img_hand])
            
        img_fused = np.concatenate(imgs, axis=0) # 12 Channels
        img_tensor = self.transform(torch.from_numpy(img_fused))
        
        # 2. Relative Trajectory Generation
        max_step = self.f[f'data/{demo}/states'].shape[0] - 1
        curr_pos = self.f[f'data/{demo}/obs/robot0_eef_pos'][step]
        curr_rot_vec = R.from_quat(self.f[f'data/{demo}/obs/robot0_eef_quat'][step]).as_rotvec()
        
        trajectory = []
        for i in range(self.pred_horizon):
            t_step = min(step + i, max_step)
            pos = self.f[f'data/{demo}/obs/robot0_eef_pos'][t_step]
            rot_vec = R.from_quat(self.f[f'data/{demo}/obs/robot0_eef_quat'][t_step]).as_rotvec()
            
            trajectory.append(np.concatenate([pos - curr_pos, rot_vec - curr_rot_vec]))
            
        traj_raw = np.array(trajectory).flatten().astype(np.float32)
        
        # 3. [DP Standard] Strict Min-Max Normalization to [-1, 1]
        traj_norm = (traj_raw - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
        traj_norm = (traj_norm * 2.0) - 1.0 
        traj_tensor = torch.from_numpy(traj_norm)
        
        return {"image": img_tensor, "pose": traj_tensor}

    def __del__(self):
        if getattr(self, 'f', None) is not None:
            self.f.close()
