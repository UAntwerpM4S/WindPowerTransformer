import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with the module and isn't trainable
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # (1,T,D) broadcasts over batch

class TemporalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "Dimension must be divisible by number of heads."

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        # x: (batch, time, dim)
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # shape: (B, heads, T, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v  # (B, heads, T, head_dim)

        out = attn_output.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_mult),
            nn.GELU(),
            nn.Linear(dim * mlp_mult, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, n_heads=4, num_layers=4, mlp_mult=4,use_posenc=True, max_len=4096):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)

        self.use_posenc = use_posenc
        if use_posenc:
            self.posenc = SinusoidalPE(model_dim, max_len=max_len)

        self.blocks = nn.Sequential(*[
            TransformerBlock(model_dim, n_heads, mlp_mult) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(model_dim, 1)  # per time step output

    def forward(self, x):
        # x: (batch, lead_time, input_dim)
        x = self.input_proj(x)  # (B, T, D)
        if self.use_posenc:                 
            x = self.posenc(x)              # (B, T, D)
        x = self.blocks(x)      # (B, T, D)
        x = self.output_proj(x).squeeze(-1)  # (B, T)
        return x