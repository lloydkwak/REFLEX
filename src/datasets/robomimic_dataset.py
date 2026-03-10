import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T

class RobomimicSE3Dataset(Dataset):
    """
    Research-grade SE(3) dataset for Robomimic HDF5 files.
    [DP Standard] Employs Absolute Action Space, Late Fusion Image Stacking, 
    Proprioceptive State Extraction, and Min-Max Normalization.
    """
    def __init__(self, file_path, prediction_horizon=16, obs_horizon=2, split="train", stats=None):
        self.file_path = file_path
        self.pred_horizon = prediction_horizon
        self.obs_horizon = obs_horizon
        self.split = split
        
        # Standard 3-channel normalization for independent ResNet evaluation
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.index_map = []
        self.demos = []
        
        with h5py.File(file_path, 'r') as f:
            if 'mask' in f and split in f['mask']:
                self.demos = [elem.decode("utf-8") for elem in np.array(f[f'mask/{split}'])]
            else:
                all_demos = list(f['data'].keys())
                split_idx = int(len(all_demos) * 0.9)
                self.demos = all_demos[:split_idx] if split == "train" else all_demos[split_idx:]
                
            for demo in self.demos:
                num_steps = f[f'data/{demo}/states'].shape[0]
                for step in range(num_steps):
                    self.index_map.append((demo, step))
        
        self.f = None
        
        if split == "train":
            self.stats = self._compute_action_stats()
        else:
            if stats is None:
                raise ValueError("Validation dataset requires 'stats' from the train dataset.")
            self.stats = stats

    def _compute_action_stats(self):
        print(f"[REFLEX] Computing Absolute Action Bounds (Min-Max) from {len(self.demos)} Train demos...")
        sample_indices = np.linspace(0, len(self.index_map)-1, num=min(1000, len(self.index_map)), dtype=int)
        all_traj = []
        
        with h5py.File(self.file_path, 'r') as f:
            for idx in sample_indices:
                demo, step = self.index_map[idx]
                max_step = f[f'data/{demo}/states'].shape[0] - 1
                
                trajectory = []
                for i in range(self.pred_horizon):
                    t_step = min(step + i, max_step)
                    pos = f[f'data/{demo}/obs/robot0_eef_pos'][t_step]
                    rot_vec = R.from_quat(f[f'data/{demo}/obs/robot0_eef_quat'][t_step]).as_rotvec()
                    
                    # [DP Standard] Store Absolute Poses directly
                    trajectory.append(np.concatenate([pos, rot_vec]))
                all_traj.append(np.array(trajectory))
                
        # Compute min/max across all samples and timesteps for the 6 DOFs
        all_traj = np.array(all_traj)
        return {
            "min": (np.min(all_traj, axis=(0, 1)) - 1e-5).astype(np.float32), 
            "max": (np.max(all_traj, axis=(0, 1)) + 1e-5).astype(np.float32)
        }

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.file_path, 'r')

        demo, step = self.index_map[idx]
        
        step_t0 = max(0, step - 1)
        step_t1 = step
        
        # 1. [DP Standard] Late Fusion Image Buffer (Shape: 4, 3, 224, 224)
        imgs = []
        for s in [step_t0, step_t1]:
            imgs.append((self.f[f'data/{demo}/obs/agentview_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32))
            imgs.append((self.f[f'data/{demo}/obs/robot0_eye_in_hand_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32))
            
        img_tensor = torch.stack([self.transform(torch.from_numpy(img)) for img in imgs], dim=0) 
        
        # 2. [DP Standard] Proprioception (Robot State) Modality
        state_t0 = np.concatenate([
            self.f[f'data/{demo}/obs/robot0_eef_pos'][step_t0], 
            R.from_quat(self.f[f'data/{demo}/obs/robot0_eef_quat'][step_t0]).as_rotvec()
        ])
        state_t1 = np.concatenate([
            self.f[f'data/{demo}/obs/robot0_eef_pos'][step_t1], 
            R.from_quat(self.f[f'data/{demo}/obs/robot0_eef_quat'][step_t1]).as_rotvec()
        ])
        
        # Apply strict Min-Max normalization to the proprioceptive state
        state_raw = np.concatenate([state_t0, state_t1]).astype(np.float32) # (12,)
        state_norm_t0 = (state_t0 - self.stats["min"]) / (self.stats["max"] - self.stats["min"]) * 2.0 - 1.0
        state_norm_t1 = (state_t1 - self.stats["min"]) / (self.stats["max"] - self.stats["min"]) * 2.0 - 1.0
        state_tensor = torch.from_numpy(np.concatenate([state_norm_t0, state_norm_t1])).float()

        # 3. Absolute Trajectory Generation
        max_step = self.f[f'data/{demo}/states'].shape[0] - 1
        
        trajectory = []
        for i in range(self.pred_horizon):
            t_step = min(step + i, max_step)
            pos = self.f[f'data/{demo}/obs/robot0_eef_pos'][t_step]
            rot_vec = R.from_quat(self.f[f'data/{demo}/obs/robot0_eef_quat'][t_step]).as_rotvec()
            trajectory.append(np.concatenate([pos, rot_vec]))
            
        traj_raw = np.array(trajectory).astype(np.float32) # (16, 6)
        
        # Absolute Min-Max Normalization to [-1, 1]
        traj_norm = (traj_raw - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
        traj_norm = (traj_norm * 2.0) - 1.0 
        traj_tensor = torch.from_numpy(traj_norm).float() 
        
        return {"image": img_tensor, "state": state_tensor, "pose": traj_tensor}

    def __del__(self):
        if getattr(self, 'f', None) is not None:
            self.f.close()
