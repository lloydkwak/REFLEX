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
        
        # Exact Marginal OT-CFM enforces straight-line paths between noise and data:
        # x_t = t * x_1 + (1 - t) * x_0
        # u_t = x_1 - x_0 (Constant target velocity)
        self.cfm = ExactMarginalOptimalTransportConditionalFlowMatcher(sigma=sigma)

    def compute_loss(self, x_1, image):
        """
        Computes the Flow Matching regression loss during training.
        
        Args:
            x_1 (torch.Tensor): Ground truth SE(3) target pose. Shape: (B, 6).
            image (torch.Tensor): Camera observation context. Shape: (B, C, H, W).
            
        Returns:
            loss (torch.Tensor): Mean Squared Error between predicted and target vector fields.
        """
        B = x_1.shape[0]
        # 1. Sample base distribution (Gaussian noise)
        x_0 = torch.randn_like(x_1)
        
        # 2. Sample location and target flow using torchcfm
        # t: Random timestep ~ U(0, 1)
        # x_t: Interpolated pose at time t
        # u_t: Target velocity vector at time t
        t, x_t, u_t = self.cfm.sample_location_and_conditional_flow(x_0, x_1)
        
        # 3. Predict the vector field and curvature
        # Squeeze t to match the network's expected shape (B,)
        v_pred, kappa_pred = self.model(x_t, t.squeeze(), image)
        
        # 4. Compute Vector Field Loss (MSE)
        loss = torch.nn.functional.mse_loss(v_pred, u_t)
        
        # Note: kappa_pred can be heavily regularized or supervised if ground-truth 
        # curvature data is available. For now, it learns implicitly or via auxiliary loss.
        return loss

    @torch.no_grad()
    def sample(self, image, num_steps=50):
        """
        Generates the target SE(3) pose and field curvature via ODE integration.
        Used during real-time inference in the robot control loop.
        
        Args:
            image (torch.Tensor): Camera observation context. Shape: (B, C, H, W).
            num_steps (int): Number of integration steps (NFE - Number of Function Evaluations).
            
        Returns:
            x_1_pred (torch.Tensor): Predicted SE(3) target pose at t=1. Shape: (B, 6).
            kappa_pred (torch.Tensor): Predicted field curvature at t=1. Shape: (B, 6).
        """
        B = image.shape[0]
        device = image.device
        
        # 1. Sample initial noise
        x_0 = torch.randn(B, self.model.pose_dim, device=device)
        
        # 2. Define the ODE function compatible with torchdiffeq
        def ode_func(t, x):
            # Broadcast scalar t to batch size
            t_batch = t.expand(B)
            # Model predicts velocity (v_pred) and curvature (kappa)
            # We only integrate the velocity to find the pose
            v_pred, _ = self.model(x, t_batch, image)
            return v_pred

        # 3. Solve the ODE from t=0 to t=1
        t_eval = torch.linspace(0.0, 1.0, num_steps, device=device)
        
        # Euler method is highly efficient for OT-CFM due to straight-line trajectories
        trajectory = odeint(ode_func, x_0, t_eval, method='euler')
        
        # 4. Extract final state at t=1
        x_1_pred = trajectory[-1]
        
        # 5. Evaluate final curvature at the destination (t=1)
        _, kappa_pred = self.model(x_1_pred, torch.ones(B, device=device), image)
        
        return x_1_pred, kappa_pred