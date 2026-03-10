import torch
import torch.nn as nn
import torchvision.models as models
import math

class SinusoidalPosEmb(nn.Module):
    """
    Standard Sinusoidal Positional Embedding for time conditioning.
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
    REFLEX Brain: SE(3) Flow Matching + Transformer Backbone.
    [DP Standard Fix] Replaced single bottleneck token with Independent Token Sequences 
    and added Proprioceptive State conditioning.
    """
    def __init__(self, pred_horizon=16, obs_horizon=2, num_cams=2, hidden_dim=256, time_dim=256):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.pose_dim = 6 
        
        # 1. Vision Backbone (Late Fusion)
        self.vision_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_encoder.fc = nn.Identity() # Outputs 512-dim feature per image
        
        # 2. Modality Encoders (Mapping everything to hidden_dim tokens)
        self.cam_proj = nn.Linear(512, hidden_dim) # Each image gets its own token
        self.state_mlp = nn.Sequential(
            nn.Linear(obs_horizon * self.pose_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim), nn.GELU()
        )
        
        # 3. Action Sequence
        self.action_emb = nn.Linear(self.pose_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, pred_horizon, hidden_dim))
        
        # 4. [DP Standard] Transformer Backbone (MinGPT / DiT style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, 
            dim_feedforward=hidden_dim * 4, 
            activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # 5. Output Regression Head
        self.out_proj = nn.Linear(hidden_dim, self.pose_dim)

    def forward(self, x_t, time, image=None, state=None, context_tokens=None):
        """
        Sequence layout: [Time, State, Cam1, Cam2, Cam3, Cam4, Action_0, ..., Action_15]
        """
        B = x_t.shape[0]
        
        # 1. Prepare Conditioning Tokens Sequence
        if context_tokens is None:
            with torch.no_grad():
                # N = obs_horizon * num_cams = 4
                N = image.shape[1] 
                img_flat = image.view(B * N, 3, 224, 224)
                features = self.vision_encoder(img_flat).view(B, N, 512)
                
            cam_tokens = self.cam_proj(features) # (B, 4, hidden_dim)
            state_token = self.state_mlp(state).unsqueeze(1) # (B, 1, hidden_dim)
            
            # Cache for ODE sampling
            context_tokens = torch.cat([state_token, cam_tokens], dim=1) # (B, 5, hidden_dim)

        time_token = self.time_mlp(time).unsqueeze(1) # (B, 1, hidden_dim)
        
        # Total conditioning sequence: [Time, State, Cam1, Cam2, Cam3, Cam4]
        cond_seq = torch.cat([time_token, context_tokens], dim=1) # (B, 6, hidden_dim)
        
        # 2. Action Sequence
        x_seq = self.action_emb(x_t) + self.pos_emb # (B, 16, hidden_dim)
        
        # 3. Transformer Processing
        seq = torch.cat([cond_seq, x_seq], dim=1) # (B, 22, hidden_dim)
        out = self.transformer(seq)
        
        # 4. Extract Vector Field Predictions (Discard the 6 cond_tokens output)
        v_pred = self.out_proj(out[:, 6:, :]) # (B, 16, 6)
        
        return v_pred, None, context_tokens
