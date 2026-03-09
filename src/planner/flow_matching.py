import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import ExactMarginalOptimalTransportConditionalFlowMatcher

class ReflexFlowMatcher(nn.Module):
    """
    REFLEX Brain: Flow Matching Engine based on Optimal Transport.
    Handles the generation of ODE trajectories and MSE loss computation for the SE(3) vector field.
    """
    def __init__(self, model, sigma=0.0):
        """
        Args:
            model (nn.Module): The ReflexFMNetwork (Predicts v_theta and kappa).
            sigma (float): Noise scheduling parameter (0.0 for deterministic OT-CFM).
        """
        super().__init__()
        self.model = model
        self.cfm = ExactMarginalOptimalTransportConditionalFlowMatcher(sigma=sigma)

    def compute_loss(self, x_1, image):
        """
        Computes the Flow Matching regression loss during training.
        """
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        
        t, x_t, u_t = self.cfm.sample_location_and_conditional_flow(x_0, x_1)
        
        v_pred, kappa_pred, _ = self.model(x_t, t.squeeze(-1), image)
        
        loss = torch.nn.functional.mse_loss(v_pred, u_t)
        return loss

    @torch.no_grad()
    def sample(self, image, num_steps=50):
        """
        Generates the target SE(3) pose and field curvature via ODE integration.
        Used during real-time inference in the robot control loop.
        """
        B = image.shape[0]
        device = image.device
        
        x_0 = torch.randn(B, self.model.pose_dim, device=device)
        _, _, cached_context = self.model(x_t=x_0, time=torch.zeros(B, device=device), image=image)        
        
        def ode_func(t, x):
            t_batch = t.view(1).expand(B)
            v_pred, _, _ = self.model(x, t_batch, image=None, context=cached_context)
            return v_pred

        t_eval = torch.linspace(0.0, 1.0, num_steps, device=device)
        trajectory = odeint(ode_func, x_0, t_eval, method='euler')
        
        x_1_pred = trajectory[-1]
        _, kappa_pred, _ = self.model(x_1_pred, torch.ones(B, device=device), image=None, context=cached_context)        
        
        return x_1_pred, kappa_pred
