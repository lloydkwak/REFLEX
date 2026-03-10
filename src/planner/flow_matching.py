import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

class ReflexFlowMatcher(nn.Module):
    """
    REFLEX Brain: OT-CFM Engine for chunked trajectory generation.
    Predicts the entire future field v_theta(x_traj, t | c).
    """
    def __init__(self, model, sigma=0.0):
        super().__init__()
        self.model = model
        self.cfm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    def compute_loss(self, x_1, image, state):
        """ Computes loss over the entire 16-step trajectory chunk with proprioception. """
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        
        t, x_t, u_t = self.cfm.sample_location_and_conditional_flow(x_0, x_1)
        
        # Predicted velocity field on the trajectory manifold
        v_pred, _, _ = self.model(x_t, t.squeeze(-1), image, state)
        
        return torch.nn.functional.mse_loss(v_pred, u_t)

    @torch.no_grad()
    def sample(self, image, state, num_steps=50):
        """ Generates 16-step SE(3) trajectory chunk via ODE integration. """
        B = image.shape[0]
        device = image.device
        
        x_0 = torch.randn(B, self.model.pred_horizon, self.model.pose_dim, device=device)
        
        # Optimization: Cache visual and state context before entering ODE loop
        _, _, cached_tokens = self.model(x_t=x_0, time=torch.zeros(B, device=device), image=image, state=state)        
        
        def ode_func(t, x):
            # Broadcast scalar t to batch size and reshape to (B,)
            t_batch = t.view(1).expand(B)
            v_pred, _, _ = self.model(x, t_batch, image=None, state=None, context_tokens=cached_tokens)
            return v_pred

        t_eval = torch.linspace(0.0, 1.0, num_steps, device=device)
        trajectory = odeint(ode_func, x_0, t_eval, method='euler')
        
        return trajectory[-1], None
