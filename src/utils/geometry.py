import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_spatial_error(pos_curr, quat_curr, pos_target, quat_target):
    """
    Computes the decoupled spatial error between two poses for OSC.
    Unlike a strict SE(3) log map, this separates linear and angular errors,
    which is standard practice for task-space impedance control.
    
    Args:
        pos_curr, pos_target (np.ndarray): 3D translations [x, y, z].
        quat_curr, quat_target (np.ndarray): Quaternions in [x, y, z, w] format.
        
    Returns:
        spatial_error (np.ndarray): 6D error vector (dp, dr) serving as the 
                                    driving force in the potential field.
    """
    # 1. Linear Error (R^3 Euclidean space)
    dp = pos_target - pos_curr
    
    # 2. Angular Error (SO(3) Log Map)
    R_c = R.from_quat(quat_curr)
    R_t = R.from_quat(quat_target)
    
    # Orientation error: R_err = R_target * R_curr^-1 (in base frame)
    R_err = R_t * R_c.inv()
    
    # Extract the rotation vector (axis * angle)
    dr = R_err.as_rotvec()
    
    # 3. Decoupled Spatial Error Command
    spatial_error = np.concatenate([dp, dr])
    
    return spatial_error

def is_pose_converged(spatial_error, lin_tol=1e-3, ang_tol=1e-2):
    """
    Evaluates if the current spatial error satisfies the convergence tolerances.
    """
    lin_err = np.linalg.norm(spatial_error[:3])
    ang_err = np.linalg.norm(spatial_error[3:])
    return (lin_err < lin_tol) and (ang_err < ang_tol)