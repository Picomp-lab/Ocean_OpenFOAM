"""
HPM (Holistic Physics Mixer) — fwv variant.

Forked from hpm/hpm_model.py. ONLY difference (inside CalibratedSpectralMixer):
the LBO basis is stored ONCE as (N, G) instead of (H, N, G) per mixer, and
non-persistently. Bit-identical output; broadcasting replaces the materialized
head copies, and a single shared SpectralBasis module replaces the per-block ones.
Parent-era checkpoints need strip_legacy_basis() (bottom of file) before load.

Based on HPM_Irregular_Mesh.py from the original HPM paper (ICML 2025),
adapted for field-to-field temporal prediction with:
  - Temporal window (W frames) with exponential decay aggregator + finite differences
  - Residual learning (predict delta from most recent frame)
  - LBO eigenbasis from OpenFOAM graph Laplacian
  - Gradient checkpointing for large meshes

Channel-agnostic: field_dim / out_dim are REQUIRED constructor arguments with
no defaults — they must come from ChannelSchema (schema.py), never be assumed.
out_dim may be < field_dim (delta channels only; frozen channels have no head).

Input:  (B, N, 3 + field_dim*W)  — coordinates + W frames of field_dim fields
        window=0 -> (B, N, field_dim): single frame, no temporal features (1b mode)
Output: (B, N, out_dim)          — predicted delta for the DELTA channels
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

class SpectralBasis(nn.Module):
    """
    Sole owner of the LBO eigenbasis, shared by every mixer.

    Two redundancies removed vs the parent:
      (B) parent .expand(heads,...) BEFORE F.normalize; normalize is
          out-of-place, so H identical copies got materialized. normalize is
          invariant to that expansion, so (N, G) + broadcast is bit-identical.
      (C) parent built one buffer per mixer -> n_layers identical copies.
          Sharing must go through a shared nn.Module: assigning a bare tensor
          across modules breaks on .to(), where _apply reallocates per module.

    persistent=False -> stays out of the checkpoint. The basis is a fixed
    on-disk artifact; if the LBO decomposition is ever re-run, eigenvector SIGN
    AMBIGUITY will silently degrade old checkpoints without raising.
    """

    def __init__(self, spectral_embedding, freq_num):
        super().__init__()
        assert spectral_embedding is not None, "SpectralBasis needs spectral_embedding"
        emb = torch.from_numpy(spectral_embedding).float()[:, :freq_num]
        assert emb.shape[1] == freq_num, \
            f"spectral_embedding has only {emb.shape[1]} modes < freq_num={freq_num}"
        self.register_buffer('basis', F.normalize(emb, p=2, dim=-1),
                             persistent=False)                    # (N, freq_num)


class CalibratedSpectralMixer(nn.Module):
    """
    Point-Calibrated Spectral Transform for irregular meshes.

    Uses LBO eigenvectors as fixed spectral basis, with a learnable
    gate network that predicts per-point frequency preferences.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,
                 freq_num=64, spectral_basis=None):
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

        # Fixed LBO eigenvectors (frozen) — SHARED, not owned. Assigning an
        # nn.Module registers it as a submodule, so .to()/dtype casts reach the
        # single underlying tensor and every mixer keeps seeing the same one.
        assert spectral_basis is not None, "mixer needs a shared SpectralBasis"
        self.spectral_basis = spectral_basis

    def forward(self, x):
        B, N, C = x.shape

        # Spectral basis: (N, G) — broadcasts against (B, H, N, G)
        basis = self.spectral_basis.basis

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
                 spectral_basis, mlp_ratio=1, act='gelu',
                 last_layer=False, out_dim=None, use_ckpt=False):
        super().__init__()
        self.last_layer = last_layer
        self.use_ckpt = use_ckpt

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mixer = CalibratedSpectralMixer(
            hidden_dim, heads=n_heads, dim_head=hidden_dim // n_heads,
            dropout=dropout, freq_num=freq_num,
            spectral_basis=spectral_basis
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)

        if last_layer:
            assert out_dim is not None, "last_layer=True requires out_dim"
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

    Temporal features extracted from window of W frames (F = field_dim):
      - macro_history: weighted average via exponential-decay aggregator (F ch)
      - dt:  1st-order finite difference from last two frames (F ch)
      - dt2: 2nd-order finite difference from last three frames (F ch)
    Total temporal features: 3*F channels

    Input to backbone: coords(3) + temporal_features(3*F) [+ spectral_pos]
    Head outputs out_dim channels (= number of DELTA channels in the schema;
    frozen / non-predicted channels have no head — fork-1b).
    """

    def __init__(self,
                 field_dim,          # REQUIRED — from ChannelSchema, no default
                 out_dim,            # REQUIRED — from ChannelSchema, no default
                 space_dim=3,
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
        # window == 0 : 无时间特征 (1b / prior-only 模式)。输入就是单帧场本身,
        #   没有历史、没有自反馈 -> 无 rollout 漂移、无 cold start。
        # window >= 3 : 原有时间窗模式 (macro/dt/dt2 需要至少 3 帧)。
        self.no_temporal = (window == 0)
        assert self.no_temporal or window >= 3, \
            "window must be 0 (no temporal features) or >= 3 (finite differences)"
        assert 0 < out_dim <= field_dim, \
            f"out_dim={out_dim} must be in (0, field_dim={field_dim}]"
        self.field_dim = field_dim
        self.window = window
        self.spectral_pos_dim = spectral_pos_dim
        self.max_grad_norm = max_grad_norm

        # --- Temporal feature extraction ---
        if not self.no_temporal:
            self.time_aggregator = nn.Linear(window, 1)

        # --- Preprocessing ---
        # window>0: coords(3) + macro_history(F) + dt(F) + dt2(F) + spectral_pos
        # window=0: coords(3) + fields(F) + spectral_pos
        n_feat = field_dim if self.no_temporal else field_dim * 3
        input_dim = space_dim + n_feat + spectral_pos_dim
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, n_hidden * 2), nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden)
        )
        # --- Spectral positional embedding (optional, from LBO eigvecs) ---
        if spectral_pos_dim > 0 and spectral_embedding is not None:
            pos_emb = torch.from_numpy(spectral_embedding[:, :spectral_pos_dim]).float()
            self.register_buffer('spectral_pos_emb', pos_emb, persistent=False)

        # --- Shared LBO basis: one tensor for the whole model ---
        self.spectral = SpectralBasis(spectral_embedding, freq_num)

        # --- Mixer blocks ---
        self.blocks = nn.ModuleList([
            MixerBlock(n_hidden, n_heads, dropout, freq_num,
                       self.spectral, mlp_ratio, act,
                       last_layer=(i == n_layers - 1),
                       out_dim=out_dim,
                       use_ckpt=use_ckpt)
            for i in range(n_layers)
        ])

        self._init_weights()
        # Must come AFTER _init_weights to avoid trunc_normal_ overwriting
        if not self.no_temporal:
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
            fields: (B, N, W, F) — W frames of F=field_dim physical fields

        Returns:
            (B, N, 3*F) — macro_history(F) + dt(F) + dt2(F)
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
            delta: (B, N, out_dim) — predicted residual for DELTA channels
        """
        B, N, _ = coords.shape

        if self.no_temporal:
            # 1b: fields 就是单帧 (B, N, F), 直接作为特征, 不做任何时间处理
            assert fields.shape[-1] == self.field_dim, (
                f"window=0 expects fields (B,N,{self.field_dim}), "
                f"got {tuple(fields.shape)}")
            temporal = fields                                    # (B, N, F)
        else:
            # Reshape fields: (B, N, W*F) -> (B, N, W, F)
            fields_4d = fields.reshape(B, N, self.window, self.field_dim)
            # Extract temporal features
            temporal = self.extract_temporal_features(fields_4d)  # (B, N, 3F)

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

        delta = x  # (B, N, out_dim) — raw delta prediction

        return delta

def strip_legacy_basis(state_dict, verbose=True):
    """Drop parent-era persistent basis keys so strict=True load still works."""
    drop = [k for k in state_dict
            if k.endswith('spectral_basis') or k.endswith('spectral_pos_emb')]
    if verbose and drop:
        freed = sum(state_dict[k].numel() * state_dict[k].element_size() for k in drop)
        print(f"[strip_legacy_basis] dropped {len(drop)} keys ({freed/1e9:.2f} GB)")
    return {k: v for k, v in state_dict.items() if k not in drop}