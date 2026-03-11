import torch
import torch.nn as nn
import torchvision.models as models
import math

def replace_bn_with_gn(root_module):
    """ 
    Recursively replaces all BatchNorm2d layers with GroupNorm.
    Essential for stabilizing visual representation learning in robotics
    due to high correlation within episodic batches.
    """
    for name, module in root_module.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(root_module, name, nn.GroupNorm(8, module.num_features))
        else:
            replace_bn_with_gn(module)

class SinusoidalPosEmb(nn.Module):
    """ Standard Sinusoidal Positional Embedding for temporal encoding. """
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

class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block augmented with Adaptive Layer Normalization (FiLM).
    Conditions the forward pass on global temporal variables via scale and shift modulation.
    """
    def __init__(self, hidden_dim, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 4) 
        )

    def forward(self, x, cond):
        film_params = self.ada_lin(cond).unsqueeze(1) 
        scale1, shift1, scale2, shift2 = film_params.chunk(4, dim=-1)
        
        x_norm = self.norm1(x) * (1.0 + scale1) + shift1
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x) * (1.0 + scale2) + shift2
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x

class ReflexFMNetwork(nn.Module):
    """
    REFLEX Brain: DiT-style Flow Matching Network.
    Employs Prefix Tokens for spatial conditioning and AdaLN for temporal conditioning.
    Predicts 7-DOF targeted velocities (6D Spatial + 1D Gripper).
    """
    def __init__(self, pred_horizon=16, obs_horizon=2, num_cams=2, hidden_dim=256, time_dim=256):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.pose_dim = 7 
        self.hidden_dim = hidden_dim
        
        self.vision_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_encoder.fc = nn.Identity()
        replace_bn_with_gn(self.vision_encoder)
        
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
        
        # Positional Embedding for Context Tokens (1 State + 4 Image = 5 Context Tokens)
        num_context_tokens = 1 + (obs_horizon * num_cams)
        self.context_pos_emb = nn.Parameter(torch.zeros(1, num_context_tokens, hidden_dim))
        
        self.layers = nn.ModuleList([
            AdaLNTransformerBlock(hidden_dim, nhead=8, dropout=0.1) for _ in range(6)
        ])
        
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
            context_tokens = context_tokens + self.context_pos_emb

        time_token = self.time_mlp(time) 
        global_cond = time_token 
        
        x_seq = self.action_emb(x_t) + self.pos_emb 
        
        seq = torch.cat([context_tokens, x_seq], dim=1) 
        
        for layer in self.layers:
            seq = layer(seq, global_cond)
            
        num_ctx = context_tokens.shape[1]
        action_features = seq[:, num_ctx:, :] 
        
        v_pred = self.out_proj(action_features) 
        
        return v_pred, None, context_tokens
