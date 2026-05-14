"""
Transolver++ Model for Wave Prediction
========================================
Based on official Transolver (ICML 2024) Irregular Mesh architecture,
with Eidetic Physics-Attention from Transolver++ (ICML 2025).

Key differences from original Transolver:
  1. Single projection (no separate in_project_fx) — eidetic state design
  2. Input-dependent temperature via small MLP + Gumbel-Softmax
  3. Gradient checkpointing in training for memory savings

Stripped of DDP/distributed code for single-GPU use.

Reference:
  Luo et al., "Transolver++: An Accurate Neural Solver for PDEs on
  Million-Scale Geometries", ICML 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import trunc_normal_
from einops import rearrange
from torch.utils.checkpoint import checkpoint


ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU
}


def gumbel_softmax(logits, tau=1, hard=False):
    """Gumbel-Softmax for differentiable discrete slice assignment."""
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / tau
    y = F.softmax(y, dim=-1)
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


class Physics_Attention_Eidetic(nn.Module):
    """
    Eidetic Physics-Attention from Transolver++.

    vs original Physics_Attention_Irregular_Mesh:
      - Single projection (in_project_x only, no in_project_fx)
      - Input-dependent temperature via MLP + Gumbel noise
      - Uses F.scaled_dot_product_attention for efficiency
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Learnable bias for temperature
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # Input-dependent temperature (replaces fixed self.temperature)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        # Single projection for both slice weights and tokens (eidetic)
        self.in_project_x = nn.Linear(dim, inner_dim)

        # Slice assignment projection
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        # QKV for slice-level attention
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape

        # Single projection (eidetic: same features for weights & tokens)
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N D

        # Input-dependent temperature + Gumbel-Softmax slice assignment
        temperature = self.proj_temperature(x_mid) + self.bias  # B H N 1
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(
            self.in_project_slice(x_mid), temperature
        )  # B H N G

        # Aggregate mesh points → slice tokens
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # Attention among slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v)

        # Deslice: map back to mesh points
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        if act in ACTIVATION:
            act = ACTIVATION[act]
        else:
            raise NotImplementedError(f"Activation {act} not supported")
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_PP_Block(nn.Module):
    """Single Transolver++ block: LayerNorm → Eidetic Attention → FFN."""
    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu',
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 use_checkpoint=True):
        super().__init__()
        self.last_layer = last_layer
        self.use_checkpoint = use_checkpoint

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Eidetic(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        # Gradient checkpointing during training to save memory
        if self.training and self.use_checkpoint:
            fx = checkpoint(self.Attn, self.ln_1(fx), use_reentrant=False) + fx
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=False) + fx
        else:
            fx = self.Attn(self.ln_1(fx)) + fx
            fx = self.mlp(self.ln_2(fx)) + fx

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class TransolverPP(nn.Module):
    """
    Transolver++ for 2D time-stepping wave prediction.

    Input:  (x, z, alpha_t, Ux_t, Uz_t)  → shape (B, N, 5)
    Output: (alpha_{t+1}, Ux_{t+1}, Uz_{t+1}) → shape (B, N, 3)
    """
    def __init__(self,
                 space_dim=2,
                 window=5,
                 channels=5,
                 n_layers=4,
                 n_hidden=128,
                 dropout=0.0,
                 n_head=4,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=3,       # alpha, Ux, Uz
                 out_dim=3,       # alpha, Ux, Uz at t+1
                 slice_num=64,
                 use_checkpoint=True):
        super().__init__()
        self.__name__ = 'TransolverPP'
        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.window = window
        self.channels = channels

        pe = torch.zeros(window, channels)
        pos = torch.arange(window).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('time_pe', pe)

        # Input: coords (space_dim) + field values (fun_dim)
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                              n_layers=0, res=False, act=act)

        # Learnable placeholder (adds small signal to break symmetry)
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Transolver_PP_Block(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(i == n_layers - 1),
                use_checkpoint=use_checkpoint
            )
            for i in range(n_layers)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx=None):
        """
        Args:
            x:  (B, N, space_dim) — normalized coordinates
            fx: (B, N, fun_dim)   — field values at current timestep
        Returns:
            (B, N, out_dim) — predicted field values at next timestep
        """
        if fx is not None:
            fx = fx + self.time_pe[None, :, None, :]    # 加时间编码
            B, W, N, C = fx.shape
            fx = fx.permute(0, 2, 1, 3).reshape(B, N, W * C)
            inp = torch.cat((x, fx), dim=-1)  # (B, N, space_dim + fun_dim)
        else:
            inp = x
        fx = self.preprocess(inp)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)
        return fx


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


if __name__ == '__main__':
    # Quick smoke test
    model = TransolverPP(
        space_dim=2, n_layers=4, n_hidden=128, n_head=4,
        fun_dim=3, out_dim=3, slice_num=64, dropout=0.0
    )
    print(f"Parameters: {count_parameters(model):,}")

    B, N = 1, 1000
    x = torch.randn(B, N, 2)
    fx = torch.randn(B, N, 3)
    out = model(x, fx)
    print(f"Input:  x={x.shape}, fx={fx.shape}")
    print(f"Output: {out.shape}")