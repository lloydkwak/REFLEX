import numpy as np
from scipy.spatial.transform import Rotation as R
from src.utils.geometry import compute_spatial_error

class PotentialFieldTranslator:
    """
    REFLEX Mid-End: Translates SE(3) fields into physical instructions.
    Generates (spatial_error, Kp, Kd) ensuring dimensional and physical consistency.
    """
    def __init__(self, base_kp=150.0):
        """
        Initializes the translator with baseline stiffness.
        
        Args:
            base_kp (float): Baseline proportional gain for task-space stiffness.
        """
        self.base_kp = base_kp

    def compute_ir(self, pose_curr, pose_target, field_curvature=None):
        """
        Samples the field to generate impedance control parameters.
        """
        if hasattr(pose_curr, 'cpu'):
            pose_curr = pose_curr.detach().cpu().numpy()
        if hasattr(pose_target, 'cpu'):
            pose_target = pose_target.detach().cpu().numpy()
        if hasattr(field_curvature, 'cpu') and field_curvature is not None:
            field_curvature = field_curvature.detach().cpu().numpy()

        # 1. Manifold Translation (RotVec -> Quaternion)
        target_rot_vec = pose_target[3:]
        target_quat = R.from_rotvec(target_rot_vec).as_quat()
        
        # 2. Extract Spatial Error
        x_err_ir = compute_spatial_error(
            pos_curr=pose_curr[:3], quat_curr=pose_curr[3:],
            pos_target=pose_target[:3], quat_target=target_quat 
        )
        
        # 3. Adaptive Stiffness (Kp) via Field Curvature
        if field_curvature is not None:
            stiffness_scale = np.clip(field_curvature, 0.2, 5.0)
            Kp_diag = self.base_kp * stiffness_scale
        else:
            Kp_diag = np.ones(6) * self.base_kp
            
        Kp = np.diag(Kp_diag)
        
        # 4. Critical Damping (Kd) Optimization
        Kd_diag = 2.0 * np.sqrt(Kp_diag)
        Kd = np.diag(Kd_diag)
        
        return x_err_ir, Kp, Kd
