import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HebbianPlasticityHead(nn.Module):
    """
    Implements a 'Living Weight' layer that adapts its weights based on 
    recent activation patterns (Hebbian Theory: cells that fire together, wire together).
    """
    def __init__(self, in_features, out_features, eta=0.01, decay=0.99):
        super().__init__()
        self.static_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Fast weights buffer (not updated via backprop)
        self.register_buffer('hebbian_trace', torch.zeros(out_features, in_features))
        self.eta = eta
        self.decay = decay

    def forward(self, x):
        # 1. Static Projection (Long-term memory)
        static_out = F.linear(x, self.static_weight, self.bias)
        
        # 2. Plastic Projection (Short-term working memory)
        plastic_out = F.linear(x, self.hebbian_trace)
        out = static_out + (0.1 * plastic_out) 
        
        # 3. Update Hebbian Trace (Only during training)
        if self.training:
            with torch.no_grad():
                x_mean = x.mean(dim=1) 
                y_mean = out.softmax(dim=-1).mean(dim=1) 
                # Outer product update rule
                update = torch.einsum('bi,bj->ij', y_mean, x_mean) 
                self.hebbian_trace = (self.decay * self.hebbian_trace) + (self.eta * update)
        return out

class SSMBlock(nn.Module):
    """
    Simplified State Space Model block using Gated Convolutional Recurrence.
    Provides O(N) complexity instead of O(N^2) Transformer attention.
    """
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x_proj = self.in_proj(x)
        x_val, x_gate = x_proj.chunk(2, dim=-1)
        
        # Reshape for Conv1D: [Batch, Length, Dim] -> [Batch, Dim, Length]
        x_val = rearrange(x_val, 'b l d -> b d l')
        x_val = self.conv(x_val)
        x_val = rearrange(x_val, 'b d l -> b l d')
        
        # Gating mechanism
        out = x_val * self.act(x_gate)
        return shortcut + self.out_proj(out)

class AdaptiveLayer(nn.Module):
    """
    A layer with a router that determines the 'thinking' score.
    Future implementations can use this score for early exiting.
    """
    def __init__(self, d_model):
        super().__init__()
        self.core = SSMBlock(d_model)
        self.router = nn.Linear(d_model, 1)
        
    def forward(self, x):
        think_score = torch.sigmoid(self.router(x))
        processed = self.core(x)
        # Soft mixing based on router confidence
        out = (think_score * processed) + ((1 - think_score) * x)
        return out, think_score.mean()