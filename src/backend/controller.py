import numpy as np
from robosuite.controllers.osc import OperationalSpaceController
import robosuite.utils.transform_utils as T

class FaultTolerantOSC(OperationalSpaceController):
    """
    REFLEX Backend: Fault-tolerant Operational Space Controller.
    Fully integrated with robosuite API. Employs Damped Least Squares (DLS) and 
    Null-space projection to ensure stability even when active DOF < 6 due to failures.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fault_mask = None

    def update_fault_mask(self, mask):
        """Updates the real-time fault status of robot joints."""
        self.fault_mask = np.array(mask)

    def compute_reflex_torques(self, x_IR, Kp, Kd):
        """
        Calculates task-space torques based on Intermediate Representation (IR).
        This bypasses the standard set_goal() to directly accept dynamic Field parameters.
        """
        # 1. Acquire current kinematic and dynamic properties directly from robosuite
        J_full = np.vstack([self.J_pos, self.J_ori])  # (6, N)
        M = self.mass_matrix                          # (N, N)
        M_inv = np.linalg.inv(M)
        
        num_joints = M.shape[0]
        
        if self.fault_mask is None:
            self.fault_mask = np.zeros(num_joints)
            
        # Current end-effector spatial velocity
        ee_vel = J_full @ self.joint_vel # (6,)
        
        # 2. Apply Fault Masking
        # Nullify Jacobian columns to strictly isolate failed joints
        mask_matrix = np.diag(1 - self.fault_mask)
        J_active = J_full @ mask_matrix
        
        # 3. Dynamically Consistent Generalized Inverse with DLS
        # DLS prevents torque explosion near kinematic singularities or when active DOF < 6
        Lambda_inv = J_active @ M_inv @ J_active.T
        damping_factor = 1e-4
        Lambda = np.linalg.pinv(Lambda_inv + np.eye(6) * damping_factor)
        
        # J_bar = M^-1 * J_a^T * Lambda
        J_bar = M_inv @ J_active.T @ Lambda
        
        # 4. Compute Task-space Wrench (Virtual Force)
        # desired_dynamics = Spring force (Kp) + Damping force (Kd)
        desired_dynamics = (Kp @ x_IR) - (Kd @ ee_vel)
        wrench = Lambda @ desired_dynamics
        
        # 5. Torque Mapping via Active Jacobian
        tau_task = J_active.T @ wrench
        
        # 6. Null-space Projection for Energy Dissipation
        # N^T = I - J_a^T * J_bar^T (Dynamically consistent null-space projection matrix)
        I_N = np.eye(num_joints)
        Null_projector = I_N - (J_active.T @ J_bar.T)
        
        # Apply slight damping to active joints in the null-space to prevent oscillation
        tau_posture = -2.0 * self.joint_vel
        tau_null = Null_projector @ tau_posture
        
        # 7. Final Dynamics Compensation
        # robosuite handles gravity and coriolis in self.torque_compensation
        torques = tau_task + tau_null + self.torque_compensation
        
        # 8. Strict Hardware Isolation
        # Ensure exact 0.0 N*m for failed joints to simulate mechanical lock
        torques = torques * (1 - self.fault_mask)
        
        return torques
