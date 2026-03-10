import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as T

class RobomimicSE3Dataset(Dataset):
    """
    Research-grade SE(3) dataset for Robomimic HDF5 files.
    [DP Standard] Employs obs_horizon=2, Min-Max Normalization, and Train/Valid splits.
    """
    def __init__(self, file_path, prediction_horizon=16, obs_horizon=2, split="train", stats=None):
        self.file_path = file_path
        self.pred_horizon = prediction_horizon
        self.obs_horizon = obs_horizon
        self.split = split
        
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406] * 4, 
                        std=[0.229, 0.224, 0.225] * 4)
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
        print(f"[REFLEX] Computing Action Min-Max Stats from {len(self.demos)} Train demos...")
        sample_indices = np.linspace(0, len(self.index_map)-1, num=min(1000, len(self.index_map)), dtype=int)
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
                
        all_traj = np.array(all_traj)
        return {
            "min": (np.min(all_traj, axis=0) - 1e-5).astype(np.float32), 
            "max": (np.max(all_traj, axis=0) + 1e-5).astype(np.float32)
        }

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.file_path, 'r')

        demo, step = self.index_map[idx]
        
        step_t0 = max(0, step - 1)
        step_t1 = step
        
        imgs = []
        for s in [step_t0, step_t1]:
            img_view = (self.f[f'data/{demo}/obs/agentview_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32)
            img_hand = (self.f[f'data/{demo}/obs/robot0_eye_in_hand_image'][s].transpose(2, 0, 1) / 255.0).astype(np.float32)
            imgs.extend([img_view, img_hand])
            
        img_fused = np.concatenate(imgs, axis=0)
        img_tensor = self.transform(torch.from_numpy(img_fused))
        
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
        
        traj_norm = (traj_raw - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
        traj_norm = (traj_norm * 2.0) - 1.0 
        traj_tensor = torch.from_numpy(traj_norm).float() 
        
        return {"image": img_tensor, "pose": traj_tensor}

    def __del__(self):
        if getattr(self, 'f', None) is not None:
            self.f.close()
