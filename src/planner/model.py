import torch
import torch.nn as nn
import torchvision.models as models
import math

def replace_bn_with_gn(root_module):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm.
    Crucial for stable vision encoding in robot learning with highly correlated batches.
    """
    for name, module in root_module.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Standard GroupNorm with 8 groups
            setattr(root_module, name, nn.GroupNorm(8, module.num_features))
        else:
            replace_bn_with_gn(module)

class SinusoidalPosEmb(nn.Module):
    """ Standard Sinusoidal Positional Embedding for time conditioning. """
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
    Transformer block with Adaptive Layer Normalization (FiLM).
    Modulates the normalization layers directly using the global condition (Time + Context).
    """
    def __init__(self, hidden_dim, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # elementwise_affine=False because AdaLN dynamically generates the scale and shift
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # FiLM Generator: Maps global condition to scale and shift parameters
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 4) # 4 parameters: scale1, shift1, scale2, shift2
        )

    def forward(self, x, cond):
        # cond: (B, hidden_dim)
        film_params = self.ada_lin(cond).unsqueeze(1) # (B, 1, hidden_dim * 4)
        scale1, shift1, scale2, shift2 = film_params.chunk(4, dim=-1)
        
        # 1. Self-Attention with AdaLN
        x_norm = self.norm1(x) * (1.0 + scale1) + shift1
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # 2. MLP with AdaLN
        x_norm = self.norm2(x) * (1.0 + scale2) + shift2
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x

class ReflexFMNetwork(nn.Module):
    """
    REFLEX Brain: Flow Matching SE(3) + Gripper Network with AdaLN.
    Upgraded to DiT-style architecture using GroupNorm and Feature-wise Linear Modulation (FiLM)
    to prevent representation bottleneck and batch statistics collapse.
    """
    def __init__(self, pred_horizon=16, obs_horizon=2, num_cams=2, hidden_dim=256, time_dim=256):
        super().__init__()
        self.pred_horizon = pred_horizon
        # Updated to 7-DOF: 6D spatial velocity + 1D gripper action
        self.pose_dim = 7 
        self.hidden_dim = hidden_dim
        
        # 1. Vision Backbone
        self.vision_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_encoder.fc = nn.Identity()
        # Replace BatchNorm with GroupNorm for stable statistics
        replace_bn_with_gn(self.vision_encoder)
        
        # 2. Modality Encoders
        self.cam_proj = nn.Linear(512, hidden_dim)
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
        
        # 4. AdaLN Transformer Backbone
        self.layers = nn.ModuleList([
            AdaLNTransformerBlock(hidden_dim, nhead=8) for _ in range(6)
        ])
        
        # 5. Output Regression Head
        self.out_proj = nn.Linear(hidden_dim, self.pose_dim)

    def forward(self, x_t, time, image=None, state=None, context_tokens=None):
        B = x_t.shape[0]
        
        # 1. Prepare Conditioning Tokens
        if context_tokens is None:
            with torch.no_grad():
                N = image.shape[1] 
                img_flat = image.view(B * N, 3, 224, 224)
                features = self.vision_encoder(img_flat).view(B, N, 512)
                
            cam_tokens = self.cam_proj(features) # (B, N, hidden_dim)
            state_token = self.state_mlp(state).unsqueeze(1) # (B, 1, hidden_dim)
            
            # Combine all context into a single global condition vector (Mean Pooling)
            context_tokens = torch.cat([state_token, cam_tokens], dim=1).mean(dim=1) # (B, hidden_dim)

        time_token = self.time_mlp(time) # (B, hidden_dim)
        
        # Global Condition for AdaLN (Context + Time)
        global_cond = context_tokens + time_token # (B, hidden_dim)
        
        # 2. Action Sequence
        x = self.action_emb(x_t) + self.pos_emb # (B, 16, hidden_dim)
        
        # 3. Transformer Processing with AdaLN
        for layer in self.layers:
            x = layer(x, global_cond)
            
        # 4. Output Projection
        v_pred = self.out_proj(x) # (B, 16, 7)
        
        return v_pred, None, context_tokens
