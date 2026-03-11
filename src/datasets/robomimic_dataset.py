import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T

class RobomimicSE3Dataset(Dataset):
    """
    Dataset for learning Target Velocity and Gripper Intent.
    Transforms raw states into a 7-DOF prediction setup (6D Velocity + 1D Gripper).
    [Fix] Replaced Min-Max with Z-Score (Standardization) to preserve velocity distribution.
    """
    def __init__(self, file_path, prediction_horizon=16, obs_horizon=2, split="train", stats=None):
        self.file_path = file_path
        self.pred_horizon = prediction_horizon
        self.obs_horizon = obs_horizon
        self.split = split
        
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
                raise ValueError("Validation dataset requires stats from the train dataset.")
            self.stats = stats

    def _compute_action_stats(self):
        """ Computes Z-Score bounds (mean, std) for the 7D velocity and gripper actions. """
        sample_indices = np.linspace(0, len(self.index_map)-1, num=min(1000, len(self.index_map)), dtype=int)
        all_traj = []
        
        with h5py.File(self.file_path, 'r') as f:
            for idx in sample_indices:
                demo, step = self.index_map[idx]
                max_step = f[f'data/{demo}/states'].shape[0] - 1
                
                trajectory = []
                for i in range(self.pred_horizon):
                    t_step = min(step + i, max_step)
                    action = f[f'data/{demo}/actions'][t_step]
                    trajectory.append(action)
                all_traj.append(np.array(trajectory))
                
        all_traj = np.array(all_traj)
        
        # Z-Score computation
        mean = np.mean(all_traj, axis=(0, 1)).astype(np.float32)
        std = np.std(all_traj, axis=(0, 1)).astype(np.float32)
        # Prevent division by zero for dimensions with zero variance
        std = np.clip(std, 1e-4, None)
        
        return {"mean": mean, "std": std}

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.file_path, 'r')

        demo, step = self.index_map[idx]
        step_t0 = max(0, step - 1)
        step_t1 = step
        
        # 1. Vision
        imgs = []
        for s in [step_t0, step_t1]:
            imgs.append((self.f[f'data/{demo}/obs/agentview_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32))
            imgs.append((self.f[f'data/{demo}/obs/robot0_eye_in_hand_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32))
        img_tensor = torch.stack([self.transform(torch.from_numpy(img)) for img in imgs], dim=0) 
        
        # 2. Proprioception (7D)
        def get_state(s):
            pos = self.f[f'data/{demo}/obs/robot0_eef_pos'][s]
            rot = R.from_quat(self.f[f'data/{demo}/obs/robot0_eef_quat'][s]).as_rotvec()
            grip = self.f[f'data/{demo}/actions'][s][-1:]
            return np.concatenate([pos, rot, grip])

        state_t0 = get_state(step_t0)
        state_t1 = get_state(step_t1)
        
        # Z-Score Normalization
        state_norm_t0 = (state_t0 - self.stats["mean"]) / self.stats["std"]
        state_norm_t1 = (state_t1 - self.stats["mean"]) / self.stats["std"]
        state_tensor = torch.from_numpy(np.concatenate([state_norm_t0, state_norm_t1])).float()

        # 3. Target Trajectory (7D Velocity + Gripper)
        max_step = self.f[f'data/{demo}/states'].shape[0] - 1
        trajectory = []
        for i in range(self.pred_horizon):
            t_step = min(step + i, max_step)
            action = self.f[f'data/{demo}/actions'][t_step]
            trajectory.append(action)
            
        traj_raw = np.array(trajectory).astype(np.float32)
        
        # Z-Score Normalization
        traj_norm = (traj_raw - self.stats["mean"]) / self.stats["std"]
        traj_tensor = torch.from_numpy(traj_norm).float() 
        
        return {"image": img_tensor, "state": state_tensor, "pose": traj_tensor}

    def __del__(self):
        if getattr(self, 'f', None) is not None:
            self.f.close()
