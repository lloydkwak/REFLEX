import numpy as np
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
        
        Args:
            pose_curr (np.ndarray): Current EE pose [x, y, z, qx, qy, qz, qw].
            pose_target (np.ndarray): Target EE pose from Flow Matching.
            field_curvature (np.ndarray, optional): (6,) vector of field steepness.
                
        Returns:
            x_err_ir (np.ndarray): 6D spatial position/orientation error.
            Kp (np.ndarray): (6, 6) Stiffness matrix.
            Kd (np.ndarray): (6, 6) Damping matrix.
        """
        # 1. Extract Spatial Error (Acting as the Potential Gradient driver)
        # Calculates the positional and orientational error driving the OSC force
        x_err_ir = compute_spatial_error(
            pos_curr=pose_curr[:3], quat_curr=pose_curr[3:],
            pos_target=pose_target[:3], quat_target=pose_target[3:]
        )
        
        # 2. Adaptive Stiffness (Kp) via Field Curvature
        # Modulates compliance based on the neural network's confidence/curvature
        if field_curvature is not None:
            # Clip scaling factor to maintain physical stability (e.g., [0.2, 5.0])
            stiffness_scale = np.clip(field_curvature, 0.2, 5.0)
            Kp_diag = self.base_kp * stiffness_scale
        else:
            Kp_diag = np.ones(6) * self.base_kp
            
        Kp = np.diag(Kp_diag)
        
        # 3. Critical Damping (Kd) Optimization
        # Optimized: square root is applied to 1D array before diagonal expansion
        Kd_diag = 2.0 * np.sqrt(Kp_diag)
        Kd = np.diag(Kd_diag)
        
        return x_err_ir, Kp, Kd