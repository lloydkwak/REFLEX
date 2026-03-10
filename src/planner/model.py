import torch
import torch.nn as nn
import torchvision.models as models
import math

class SinusoidalPosEmb(nn.Module):
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
    REFLEX Brain: SE(3) Flow Matching with Action Chunking (Tp=16).
    Predicts trajectory fields from fused multi-view images.
    """
    def __init__(self, pred_horizon=16, context_dim=512, time_dim=256, hidden_dim=512):
        super().__init__()
        self.pose_dim = pred_horizon * 6 # 96 dimensions for 16-step trajectory
        
        # 1. 6-Channel Vision Encoder Initialization
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy ImageNet weights to both views for balanced initialization
            resnet.conv1.weight[:, :3] = original_conv.weight
            resnet.conv1.weight[:, 3:] = original_conv.weight
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Embedding Layers
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.pose_emb = nn.Linear(self.pose_dim, hidden_dim)
        self.context_emb = nn.Linear(context_dim, hidden_dim)
        
        # 3. Vector Field Head
        self.joint_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + time_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, self.pose_dim)
        )

    def forward(self, x_t, time, image=None, context=None):
        if context is None:
            with torch.no_grad():
                c = self.vision_encoder(image) 
                context = c.view(c.size(0), -1)

        t_emb = self.time_mlp(time)
        x_emb = self.pose_emb(x_t)
        c_emb = self.context_emb(context)
        
        h = torch.cat([x_emb, c_emb, t_emb], dim=-1)
        v_pred = self.joint_mlp(h)
        
        return v_pred, None, context
