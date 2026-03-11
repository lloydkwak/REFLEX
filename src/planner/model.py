import torch
import torch.nn as nn
import torchvision.models as models
import math

class SinusoidalPosEmb(nn.Module):
    """ Standard Sinusoidal Positional Embedding. """
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
    """ Flow Matching SE(3) + Gripper Network. """
    def __init__(self, pred_horizon=16, obs_horizon=2, num_cams=2, hidden_dim=256, time_dim=256):
        super().__init__()
        self.pred_horizon = pred_horizon
        # Updated to 7-DOF: 6D spatial velocity + 1D gripper action
        self.pose_dim = 7 
        
        self.vision_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_encoder.fc = nn.Identity()
        
        self.cam_proj = nn.Linear(512, hidden_dim)
        self.state_mlp = nn.Sequential(
            nn.Linear(obs_horizon * self.pose_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim), nn.GELU()
        )
        
        self.action_emb = nn.Linear(self.pose_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, pred_horizon, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, 
            dim_feedforward=hidden_dim * 4, 
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.out_proj = nn.Linear(hidden_dim, self.pose_dim)

    def forward(self, x_t, time, image=None, state=None, context_tokens=None):
        B = x_t.shape[0]
        
        if context_tokens is None:
            with torch.no_grad():
                N = image.shape[1] 
                img_flat = image.view(B * N, 3, 224, 224)
                features = self.vision_encoder(img_flat).view(B, N, 512)
                
            cam_tokens = self.cam_proj(features)
            state_token = self.state_mlp(state).unsqueeze(1)
            context_tokens = torch.cat([state_token, cam_tokens], dim=1)

        time_token = self.time_mlp(time).unsqueeze(1)
        cond_seq = torch.cat([time_token, context_tokens], dim=1)
        
        x_seq = self.action_emb(x_t) + self.pos_emb
        seq = torch.cat([cond_seq, x_seq], dim=1)
        
        out = self.transformer(seq)
        v_pred = self.out_proj(out[:, 6:, :])
        
        return v_pred, None, context_tokens
