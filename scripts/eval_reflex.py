import os
import sys
import torch
import numpy as np
import argparse
import torchvision.transforms as T
import robosuite as suite
from collections import deque
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation as R

from src.planner.model import ReflexFMNetwork
from src.planner.flow_matching import ReflexFlowMatcher
from src.planner.sampler import PotentialFieldTranslator
from src.envs.custom_env import FaultInjectionWrapper
from src.backend.controller import FaultTolerantOSC

def reflex_run_controller(self):
    if hasattr(self, 'reflex_x_IR'):
        return self.compute_reflex_torques(self.reflex_x_IR, self.reflex_Kp, self.reflex_Kd)
    from robosuite.controllers.osc import OperationalSpaceController
    return OperationalSpaceController.run_controller(self)

FaultTolerantOSC.run_controller = reflex_run_controller
import robosuite.controllers.controller_factory
sys.modules['robosuite.controllers.controller_factory'].OperationalSpaceController = FaultTolerantOSC

def preprocess_image(obs_buffer, transform, device):
    """ [DP Standard] Independent stacking for Late Fusion: (1, 4, 3, 224, 224) """
    fused_images = []
    for o in obs_buffer:
        fused_images.append((o['agentview_image'].transpose(2, 0, 1) / 255.0).astype(np.float32))
        fused_images.append((o['robot0_eye_in_hand_image'].transpose(2, 0, 1) / 255.0).astype(np.float32))
        
    img_tensors = [transform(torch.from_numpy(img)) for img in fused_images]
    img_tensor = torch.stack(img_tensors, dim=0).unsqueeze(0).to(device)
    return img_tensor

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PickPlaceCan")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["normal", "fault"], default="normal")
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--exec_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"[REFLEX] Loading DP-Perfect Brain from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    
    model = ReflexFMNetwork(pred_horizon=args.tp).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    stats = ckpt['stats']
    action_min = torch.from_numpy(stats['min']).to(args.device)
    action_max = torch.from_numpy(stats['max']).to(args.device)
    
    fm_engine = ReflexFlowMatcher(model).to(args.device)
    translator = PotentialFieldTranslator(base_kp=150.0)
    
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=args.task, robots="Panda", controller_configs=config,
        has_renderer=True, has_offscreen_renderer=True, control_freq=20,
        horizon=400, use_object_obs=False, use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=224, camera_widths=224,
    )
    
    if args.mode == "fault":
        env = FaultInjectionWrapper(env, max_faults=1, fault_type="lock", trigger_range=(0.3, 0.7))

    obs = env.reset()
    obs_buffer = deque(maxlen=2)
    obs_buffer.append(obs)
    obs_buffer.append(obs) 
    
    done = False
    step_count = 0
    success = False
    
    while not done and step_count < env.horizon:
        img_tensor = preprocess_image(obs_buffer, transform, args.device)
        
        # [DP Standard Fixed] Extract and Normalize Proprioceptive State accurately per dimension
        states = []
        for o in obs_buffer:
            pos = o['robot0_eef_pos']
            rot = R.from_quat(o['robot0_eef_quat']).as_rotvec()
            state_single = np.concatenate([pos, rot]) # (6,)
            # Apply exact 6D min-max stats before concatenation
            state_norm_single = (state_single - stats['min']) / (stats['max'] - stats['min']) * 2.0 - 1.0
            states.append(state_norm_single)
            
        state_norm = np.concatenate(states) # (12,)
        state_tensor = torch.from_numpy(state_norm).float().unsqueeze(0).to(args.device)
        
        # Predict normalized trajectory [-1, 1]
        pred_norm, _ = fm_engine.sample(image=img_tensor, state=state_tensor, num_steps=20)
        
        # Inverse Min-Max Normalization -> Yields Absolute Space Poses
        pred_trajectory = (pred_norm + 1.0) / 2.0 * (action_max - action_min) + action_min
        pred_trajectory = pred_trajectory.squeeze(0).cpu().numpy() # (16, 6)
        
        for i in range(args.exec_steps):
            if step_count >= env.horizon: break
                
            curr_pos = obs['robot0_eef_pos']
            curr_quat = obs['robot0_eef_quat']
            pose_curr = np.concatenate([curr_pos, curr_quat])
            
            # The prediction is the absolute target pose
            pose_target = pred_trajectory[i]
            
            x_err_ir, Kp, Kd = translator.compute_ir(pose_curr, pose_target)
            
            x_err_ir[:3] = np.clip(x_err_ir[:3], -0.05, 0.05)
            x_err_ir[3:] = np.clip(x_err_ir[3:], -0.2, 0.2)
            
            env.robots[0].controller.reflex_x_IR = x_err_ir
            env.robots[0].controller.reflex_Kp = Kp
            env.robots[0].controller.reflex_Kd = Kd
            
            gripper_action = 1.0 if obs['robot0_eef_pos'][2] < 0.85 else -1.0
            action = np.concatenate([np.zeros(6), [gripper_action]])
            
            obs, reward, done, info = env.step(action)
            obs_buffer.append(obs) 
            env.render()
            
            step_count += 1
            if env._check_success():
                success = True
                break
        if success: break

    print(f"\n[REFLEX] Evaluation Finished. Result: {'SUCCESS' if success else 'FAILED'} (Steps: {step_count})")
    env.close()

if __name__ == "__main__":
    evaluate()
