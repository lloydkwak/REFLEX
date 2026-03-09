import torch
import torch.nn as nn
import torchvision.models as models
import math

class SinusoidalPosEmb(nn.Module):
    """
    Standard Sinusoidal Positional Embedding for time conditioning (t).
    Translates scalar time [0, 1] into a high-dimensional feature vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ReflexFMNetwork(nn.Module):
    """
    REFLEX Brain: SE(3) Flow Matching Neural Network.
    Predicts the vector field v_theta(x_t, t | c) and the field curvature \kappa.
    """
    def __init__(self, pose_dim=6, context_dim=512, time_dim=256, hidden_dim=512):
        super().__init__()
        self.pose_dim = pose_dim
        
        # 1. Vision Encoder (Context Extractor)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Time & Pose Embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.pose_emb = nn.Linear(pose_dim, hidden_dim)
        self.context_emb = nn.Linear(context_dim, hidden_dim)
        
        # 3. Core Flow Matching MLP
        self.joint_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        
        # 4. Dual Output Heads
        self.velocity_head = nn.Linear(hidden_dim, pose_dim)
        self.curvature_head = nn.Sequential(
            nn.Linear(hidden_dim, pose_dim),
            nn.Softplus()
        )

    def forward(self, x_t, time, image=None, context=None):
        """
        Args:
            x_t (torch.Tensor): Noisy pose at time t. Shape (B, 6).
            time (torch.Tensor): Flow Matching time step t in [0, 1]. Shape (B,).
            image (torch.Tensor): Camera observation. Shape (B, 3, H, W).
            
        Returns:
            v_pred (torch.Tensor): Predicted vector field (velocity). Shape (B, 6).
            kappa (torch.Tensor): Predicted field curvature. Shape (B, 6).
        """
        if context is None:
            with torch.no_grad():
                c = self.vision_encoder(image) 
                context = c.view(c.size(0), -1)

        t_emb = self.time_mlp(time)
        x_emb = self.pose_emb(x_t)
        c_emb = self.context_emb(context)
        
        fused_features = torch.cat([x_emb, c_emb, t_emb], dim=-1)
        hidden = self.joint_mlp(fused_features)
        
        v_pred = self.velocity_head(hidden)
        kappa = self.curvature_head(hidden)
        
        return v_pred, kappa, context
