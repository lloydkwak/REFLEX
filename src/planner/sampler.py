import numpy as np

class PotentialFieldTranslator:
    """
    Translates SE(3) velocity fields into physical impedance parameters.
    """
    def __init__(self, base_kp=150.0):
        self.base_kp = base_kp

    def compute_ir(self, velocity_target, field_curvature=None):
        """
        Directly uses the generated target velocity as the intermediate representation.
        """
        if hasattr(velocity_target, 'cpu'):
            velocity_target = velocity_target.detach().cpu().numpy()

        # The field model now natively outputs the 6D spatial error command
        x_err_ir = velocity_target
        
        if field_curvature is not None:
            stiffness_scale = np.clip(field_curvature, 0.2, 5.0)
            Kp_diag = self.base_kp * stiffness_scale
        else:
            Kp_diag = np.ones(6) * self.base_kp
            
        Kp = np.diag(Kp_diag)
        Kd_diag = 2.0 * np.sqrt(Kp_diag)
        Kd = np.diag(Kd_diag)
        
        return x_err_ir, Kp, Kd
