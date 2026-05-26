"""
HPM (Holistic Physics Mixer) adapted for time-stepping prediction.

Based on HPM_Irregular_Mesh.py from the original HPM paper (ICML 2025),
adapted for field-to-field temporal prediction with:
  - Temporal window (W=6) with exponential decay aggregator + finite differences
  - Residual learning (predict delta from most recent frame)
  - LBO eigenbasis from OpenFOAM graph Laplacian
  - Gradient checkpointing for large meshes

Input:  (B, N, 3 + 6*W)  — coordinates + W frames of 6 physical fields
Output: (B, N, 6)         — predicted delta (residual) for next timestep

Physical fields: [alpha.water, Ux, Uy, Uz, p_rgh, nut]
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers import trunc_normal_
from einops import rearrange


# ============================================================
# Calibrated Spectral Mixer (core HPM mechanism)
# ============================================================

class CalibratedSpectralMixer(nn.Module):
    """
    Point-Calibrated Spectral Transform for irregular meshes.

    Uses LBO eigenvectors as fixed spectral basis, with a learnable
    gate network that predicts per-point frequency preferences.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,
                 freq_num=64, spectral_embedding=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        # Projections
        self.in_project_x = nn.Linear(dim, inner_dim)    # for gate
        self.in_project_fx = nn.Linear(dim, inner_dim)   # for value

        # Spectral domain mixing weights
        self.mlp_trans_weights = nn.Parameter(torch.empty(dim_head, dim_head))
        nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))

        # Gate: predict frequency preference per point
        self.in_project_gates = nn.Linear(dim_head, freq_num)
        nn.init.orthogonal_(self.in_project_gates.weight)

        self.layernorm = nn.LayerNorm(dim_head)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Fixed LBO eigenvectors (frozen)
        # spectral_embedding: (N, K) numpy array
        emb = torch.from_numpy(spectral_embedding).float()
        emb = emb[:, :freq_num]                              # (N, freq_num)
        emb = emb.unsqueeze(0).expand(heads, -1, -1)         # (H, N, freq_num)
        emb = F.normalize(emb, p=2, dim=-1)                  # L2 normalize
        self.register_buffer('spectral_basis', emb)           # (H, N, freq_num)

    def forward(self, x):
        B, N, C = x.shape

        # Spectral basis: (1, H, N, G)
        basis = self.spectral_basis.unsqueeze(0)

        # Value projection: (B, H, N, C)
        fx = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
        fx = fx.permute(0, 2, 1, 3).contiguous()

        # Gate projection: (B, H, N, G)
        x_gate = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
        x_gate = x_gate.permute(0, 2, 1, 3).contiguous()
        gate = torch.softmax(self.in_project_gates(x_gate) / self.temperature, dim=-1)

        # Calibrated basis
        eigens = gate * basis                                 # (B, H, N, G)

        # Forward spectral transform: physical → spectral
        spectral = torch.einsum("bhnc,bhng->bhgc", fx, eigens)

        # Spectral domain processing: LayerNorm over channel dim only
        spectral = self.layernorm(spectral)  # normalizes last dim (dim_head)
        spectral = torch.einsum("bhgi,io->bhgo", spectral, self.mlp_trans_weights)

        # Inverse spectral transform: spectral → physical
        out = torch.einsum("bhgc,bhng->bhnc", spectral, eigens)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# ============================================================
# MLP and Mixer Block
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=1, act='gelu', res=True):
        super().__init__()
        act_fn = {'gelu': nn.GELU, 'silu': nn.SiLU, 'relu': nn.ReLU}[act]
        self.res = res
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden_dim), act_fn())
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act_fn())
            for _ in range(n_layers)
        ])
        self.post = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.pre(x)
        for layer in self.layers:
            x = layer(x) + x if self.res else layer(x)
        return self.post(x)


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, freq_num,
                 spectral_embedding, mlp_ratio=1, act='gelu',
                 last_layer=False, out_dim=6, use_ckpt=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_ckpt = use_ckpt

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mixer = CalibratedSpectralMixer(
            hidden_dim, heads=n_heads, dim_head=hidden_dim // n_heads,
            dropout=dropout, freq_num=freq_num,
            spectral_embedding=spectral_embedding
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)

        if last_layer:
            self.ln3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if self.training and self.use_ckpt:
            x = checkpoint(self.mixer, self.ln1(x), use_reentrant=False) + x
            x = checkpoint(self.ffn, self.ln2(x), use_reentrant=False) + x
        else:
            x = self.mixer(self.ln1(x)) + x
            x = self.ffn(self.ln2(x)) + x

        if self.last_layer:
            return self.head(self.ln3(x))
        return x


# ============================================================
# HPM Model (time-stepping variant)
# ============================================================

class HPM(nn.Module):
    """
    HPM adapted for field-to-field time-stepping prediction.

    Temporal features extracted from window of W frames:
      - macro_history: weighted average via exponential-decay aggregator (6 ch)
      - dt:  1st-order finite difference from last two frames (6 ch)
      - dt2: 2nd-order finite difference from last three frames (6 ch)
    Total temporal features: 18 channels

    Input to backbone: coords(3) + temporal_features(18) = 21 dims
    Positional embedding from LBO eigenvectors optionally appended.
    """

    def __init__(self,
                 space_dim=3,
                 field_dim=6,
                 out_dim=6,
                 window=6,
                 n_hidden=64,
                 n_layers=4,
                 n_heads=8,
                 freq_num=64,
                 dropout=0.0,
                 mlp_ratio=1,
                 act='gelu',
                 spectral_pos_dim=0,
                 spectral_embedding=None,
                 use_ckpt=True,
                 max_grad_norm=0.1):
        super().__init__()
        assert window >= 3, "Window must be >= 3 for finite differences"
        self.field_dim = field_dim
        self.window = window
        self.spectral_pos_dim = spectral_pos_dim
        self.max_grad_norm = max_grad_norm

        # --- Temporal feature extraction ---
        self.time_aggregator = nn.Linear(window, 1)

        # --- Preprocessing ---
        # Input: coords(3) + macro_history(6) + dt(6) + dt2(6) + spectral_pos
        input_dim = space_dim + field_dim * 3 + spectral_pos_dim
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, n_hidden * 2), nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden)
        )
        # --- Spectral positional embedding (optional, from LBO eigvecs) ---
        if spectral_pos_dim > 0 and spectral_embedding is not None:
            pos_emb = torch.from_numpy(spectral_embedding[:, :spectral_pos_dim]).float()
            self.register_buffer('spectral_pos_emb', pos_emb)  # (N, spectral_pos_dim)

        # --- Mixer blocks ---
        self.blocks = nn.ModuleList([
            MixerBlock(n_hidden, n_heads, dropout, freq_num,
                       spectral_embedding, mlp_ratio, act,
                       last_layer=(i == n_layers - 1),
                       out_dim=out_dim,
                       use_ckpt=use_ckpt)
            for i in range(n_layers)
        ])

        self._init_weights()
        # Must come AFTER _init_weights to avoid trunc_normal_ overwriting
        self._init_time_aggregator()

    def _init_time_aggregator(self):
        """Initialize with exponential decay: recent frames get higher weight."""
        with torch.no_grad():
            decay = 0.5
            w = torch.tensor([decay ** (self.window - 1 - i) for i in range(self.window)])
            w = w / w.sum()
            self.time_aggregator.weight.copy_(w.unsqueeze(0))
            self.time_aggregator.bias.zero_()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def extract_temporal_features(self, fields):
        """
        Extract temporal features from windowed field data.

        Args:
            fields: (B, N, W, F) — W frames of F=6 physical fields

        Returns:
            (B, N, 3*F) — macro_history(6) + dt(6) + dt2(6) = 18 channels
        """
        # Macro history: weighted average across time window
        # fields: (B, N, W, F) -> permute -> (B, N, F, W)
        f_perm = fields.permute(0, 1, 3, 2)
        macro = self.time_aggregator(f_perm).squeeze(-1)        # (B, N, F)

        # 1st-order finite difference (velocity-like)
        dt = fields[:, :, -1, :] - fields[:, :, -2, :]          # (B, N, F)

        # 2nd-order finite difference (acceleration-like)
        dt2 = (fields[:, :, -1, :] - 2 * fields[:, :, -2, :]
               + fields[:, :, -3, :])                            # (B, N, F)

        return torch.cat([macro, dt, dt2], dim=-1)               # (B, N, 3F)

    def forward(self, coords, fields):
        """
        Args:
            coords: (B, N, 3)     — spatial coordinates
            fields: (B, N, W*F)   — W frames × F fields, flattened

        Returns:
            delta: (B, N, out_dim) — predicted residual
        """
        B, N, _ = coords.shape

        # Reshape fields: (B, N, W*F) -> (B, N, W, F)
        fields_4d = fields.reshape(B, N, self.window, self.field_dim)

        # Extract temporal features
        temporal = self.extract_temporal_features(fields_4d)     # (B, N, 18)

        # Build input: coords + temporal + optional spectral pos
        if self.spectral_pos_dim > 0:
            pos = self.spectral_pos_emb.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([coords, temporal, pos], dim=-1)
        else:
            x = torch.cat([coords, temporal], dim=-1)

        # Preprocess
        x = self.preprocess(x)

        # Mixer blocks
        for block in self.blocks:
            x = block(x)

        return x  # (B, N, out_dim) — delta prediction