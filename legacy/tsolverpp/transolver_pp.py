"""
Transolver++ for 3D two-phase flow prediction.

Input:  (B, N, 3 + 6*W)  — 3 coords + 6 fields × window
Output: (B, N, 6)         — predicted fields at t+1 (residual)

Temporal feature extraction:
  - Macro history: learned weighted sum over window (exp-decay init)
  - 1st-order finite difference (velocity of change)
  - 2nd-order finite difference (acceleration of change)
  → Fixed 21-dim input to backbone regardless of window size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint


def gumbel_softmax(logits, tau=1, hard=False):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y


class EideticAttention(nn.Module):
    def __init__(self, dim, heads=8, slice_num=64, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dim_head = heads, dim // heads

        self.in_project = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.proj_temp = nn.Sequential(
            nn.Linear(self.dim_head, slice_num), nn.GELU(),
            nn.Linear(slice_num, 1), nn.GELU())
        self.bias = nn.Parameter(torch.full([1, heads, 1, 1], 0.5))

        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape
        h, d = self.heads, self.dim_head
        x_mid = self.in_project(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
        tau = torch.clamp(self.proj_temp(x_mid) + self.bias, min=0.01)
        sw = gumbel_softmax(self.in_project_slice(x_mid), tau)
        tokens = torch.einsum("bhnc,bhng->bhgc", x_mid, sw) / (sw.sum(2, keepdim=True).transpose(-1, -2) + 1e-5)
        out = F.scaled_dot_product_attention(self.to_q(tokens), self.to_k(tokens), self.to_v(tokens))
        out = rearrange(torch.einsum("bhgc,bhng->bhnc", out, sw), "b h n d -> b n (h d)")
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, slice_num, dropout, mlp_ratio, out_dim,
                 last_layer=False, use_ckpt=True):
        super().__init__()
        self.last_layer, self.use_ckpt = last_layer, use_ckpt
        self.ln1 = nn.LayerNorm(dim)
        self.attn = EideticAttention(dim, heads, slice_num, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        if last_layer:
            self.ln3, self.head = nn.LayerNorm(dim), nn.Linear(dim, out_dim)

    def forward(self, x):
        if self.training and self.use_ckpt:
            x = checkpoint(self.attn, self.ln1(x), use_reentrant=False) + x
            x = checkpoint(self.ffn, self.ln2(x), use_reentrant=False) + x
        else:
            x = self.attn(self.ln1(x)) + x
            x = self.ffn(self.ln2(x)) + x
        return self.head(self.ln3(x)) if self.last_layer else x


class TransolverPP(nn.Module):
    def __init__(self, space_dim=3, field_dim=6, out_dim=6, window=6,
                 n_hidden=128, n_layers=4, n_heads=8, slice_num=64,
                 mlp_ratio=1, dropout=0.0):
        super().__init__()
        assert window >= 3, "window must be >= 3 for finite differences"
        self.field_dim = field_dim
        self.window = window

        # Time aggregator: Linear(window, 1) with exponential decay init
        self.time_aggregator = nn.Linear(window, 1)
        self._init_time_aggregator()

        # Input: coords(3) + macro_history(6) + dt(6) + dt2(6) = 21
        self.preprocess = nn.Sequential(
            nn.Linear(space_dim + field_dim * 3, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, n_hidden))
        self.placeholder = nn.Parameter(torch.randn(n_hidden) / n_hidden)

        self.blocks = nn.ModuleList([
            Block(n_hidden, n_heads, slice_num, dropout, mlp_ratio, out_dim,
                  last_layer=(i == n_layers - 1))
            for i in range(n_layers)])
        self._init_weights()

    def _init_time_aggregator(self):
        """Exponential decay: recent frames get higher weight."""
        with torch.no_grad():
            decay = 0.5
            w = torch.tensor([decay ** (self.window - 1 - i) for i in range(self.window)])
            w = w / w.sum()  # normalize
            self.time_aggregator.weight.copy_(w.unsqueeze(0))
            nn.init.zeros_(self.time_aggregator.bias)

    def _init_weights(self):
        for m in self.modules():
            if m is self.time_aggregator:
                continue  # already initialized
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, _ = x.shape
        coords = x[..., :3]
        fields = x[..., 3:].reshape(B, N, self.window, self.field_dim)  # (B, N, W, 6)

        # Finite differences from last 3 frames
        dt = fields[:, :, -1] - fields[:, :, -2]
        dt2 = fields[:, :, -1] - 2 * fields[:, :, -2] + fields[:, :, -3]

        # Macro history: (B, N, 6, W) → Linear → (B, N, 6)
        macro = self.time_aggregator(fields.transpose(-1, -2)).squeeze(-1)

        # Combine: (B, N, 21)
        fx = self.preprocess(torch.cat([coords, macro, dt, dt2], -1)) + self.placeholder
        for block in self.blocks:
            fx = block(fx)
        return fields[:, :, -1] + fx  # residual from current frame