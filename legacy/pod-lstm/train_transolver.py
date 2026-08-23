"""
Transolver for Wave Flow Field Prediction
==========================================
Adapts the Transolver (ICML 2024) to predict next-timestep flow fields
for two-phase VOF wave simulation data.

Task: Given (x, z, alpha_t, Ux_t, Uz_t) at each mesh point,
      predict (alpha_{t+1}, Ux_{t+1}, Uz_{t+1}).

Usage:
    python train_transolver.py \
        --data_dir $OCEAN_DATA/transolver_data \
        --output   $OCEAN_DATA/transolver_results \
        --gpu 0

Key differences from standard Transolver benchmarks:
    - Temporal prediction (field at t -> field at t+1) instead of
      parameter-to-field mapping
    - Two-phase flow with discontinuous alpha field
    - 149,758 unstructured mesh points per sample
    - Uses Irregular Mesh variant (no grid structure assumed)
"""

import os
import sys
import argparse
import json
import time as timer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import wandb

# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser('Transolver Wave Training')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Directory with coords.npy, fields.npy, times.npy')
parser.add_argument('--output', type=str, default='./transolver_results')
parser.add_argument('--gpu', type=str, default='0')

# Data params
parser.add_argument('--transient_time', type=float, default=10.0,
                    help='Discard timesteps before this time (transient)')
parser.add_argument('--test_time', type=float, default=45.0,
                    help='Timesteps after this are test set')

# Model params
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--slice_num', type=int, default=64,
                    help='Number of physics-aware slices')
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--unified_pos', type=int, default=0)

# Training params
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size (1 recommended for 150k points)')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--grad_accum', type=int, default=4,
                    help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
parser.add_argument('--max_grad_norm', type=float, default=1.0)

# Evaluation
parser.add_argument('--ar_steps', type=int, default=50,
                    help='Number of autoregressive rollout steps for evaluation')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# ============================================================
# Physics Attention (from official Transolver repo)
# ============================================================

class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics-aware attention for irregular meshes."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)\
            .permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)\
            .permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[..., None])

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


# ============================================================
# Transolver Model
# ============================================================

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        act_fn = nn.GELU
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class Transolver_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=2, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, mlp_ratio=1, fun_dim=3, out_dim=3,
                 slice_num=32, ref=8, unified_pos=False):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.n_hidden = n_hidden

        if unified_pos:
            in_dim = fun_dim + ref * ref
        else:
            in_dim = fun_dim + space_dim

        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False)
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))

        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num,
                last_layer=(i == n_layers - 1)
            ) for i in range(n_layers)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        gridx = torch.linspace(0, 1, self.ref).float()
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat(batchsize, 1, self.ref, 1)
        gridy = torch.linspace(0, 1, self.ref).float()
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat(batchsize, self.ref, 1, 1)
        grid_ref = torch.cat((gridx, gridy), dim=-1).to(x.device)
        grid_ref = grid_ref.reshape(batchsize, self.ref * self.ref, 2)
        pos = torch.sqrt(torch.sum(
            (x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1
        )).reshape(batchsize, x.shape[1], self.ref * self.ref)
        return pos

    def forward(self, x, fx):
        """
        x:  (B, N, 2)  — spatial coordinates (x, z)
        fx: (B, N, 3)  — input fields (alpha, Ux, Uz)
        Returns: (B, N, 3) — predicted next-step fields
        """
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])
        if fx is not None:
            fx = torch.cat((x, fx), dim=-1)  # (B, N, 5)
        else:
            fx = x
        fx = self.preprocess(fx)  # (B, N, n_hidden)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)
        return fx  # (B, N, 3)


# ============================================================
# Dataset
# ============================================================

class WaveFieldDataset(Dataset):
    """Dataset of consecutive timestep pairs for next-step prediction."""

    def __init__(self, coords, fields, indices):
        """
        coords:  (N, 2)     — shared spatial coordinates
        fields:  (T, N, 3)  — all timestep fields
        indices: list of int — which timestep indices to use as input
                               (target is index+1)
        """
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.fields = torch.tensor(fields, dtype=torch.float32)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        fx_in = self.fields[t]      # (N, 3) input fields at time t
        fx_out = self.fields[t + 1]  # (N, 3) target fields at time t+1
        return self.coords, fx_in, fx_out


# ============================================================
# Normalizer (from official Transolver repo)
# ============================================================

class FieldNormalizer:
    """Per-channel normalization for multi-variable fields."""
    def __init__(self, data):
        """data: (T, N, C) or (T*N, C)"""
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])
        self.mean = data.mean(dim=0, keepdim=True)  # (1, C)
        self.std = data.std(dim=0, keepdim=True) + 1e-8

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# ============================================================
# Relative L2 loss
# ============================================================

class RelativeLoss:
    """Relative L2 loss: ||pred - true||_2 / ||true||_2"""
    def __call__(self, pred, true):
        batch = pred.shape[0]
        diff = torch.norm(pred.reshape(batch, -1) - true.reshape(batch, -1), p=2, dim=1)
        ref = torch.norm(true.reshape(batch, -1), p=2, dim=1)
        return (diff / ref).mean()


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, loss_fn,
                normalizer, grad_accum, max_grad_norm, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    optimizer.zero_grad()

    for i, (pos, fx_in, fx_out) in enumerate(loader):
        pos = pos.to(device)
        fx_in = fx_in.to(device)
        fx_out = fx_out.to(device)

        # Normalize input and target
        fx_in_norm = normalizer.encode(fx_in)
        fx_out_norm = normalizer.encode(fx_out)

        pred = model(pos, fx_in_norm)

        # Compute loss in normalized space
        loss = loss_fn(pred, fx_out_norm) / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum * pos.shape[0]
        n_samples += pos.shape[0]

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, loss_fn, normalizer, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    per_var_mse = torch.zeros(3, device=device)

    for pos, fx_in, fx_out in loader:
        pos = pos.to(device)
        fx_in = fx_in.to(device)
        fx_out = fx_out.to(device)

        fx_in_norm = normalizer.encode(fx_in)
        pred_norm = model(pos, fx_in_norm)
        pred = normalizer.decode(pred_norm)

        # Relative L2 loss
        rel_loss = loss_fn(pred, fx_out)
        total_loss += rel_loss.item() * pos.shape[0]
        n_samples += pos.shape[0]

        # Per-variable MSE (in physical space)
        for j in range(3):
            mse_j = ((pred[:, :, j] - fx_out[:, :, j]) ** 2).mean()
            per_var_mse[j] += mse_j * pos.shape[0]

    per_var_mse /= n_samples
    return total_loss / n_samples, per_var_mse.cpu().numpy()


@torch.no_grad()
def autoregressive_rollout(model, coords, fields, start_idx, n_steps,
                           normalizer, device):
    """Autoregressive rollout: feed predictions back as input."""
    model.eval()
    coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
    current = torch.tensor(fields[start_idx], dtype=torch.float32).unsqueeze(0).to(device)

    preds = [current.cpu().numpy()[0]]  # store initial state
    targets = [fields[start_idx]]

    for step in range(n_steps):
        t_idx = start_idx + step + 1
        if t_idx >= len(fields):
            break

        current_norm = normalizer.encode(current)
        pred_norm = model(coords_t, current_norm)
        pred = normalizer.decode(pred_norm)

        preds.append(pred.cpu().numpy()[0])
        targets.append(fields[t_idx])

        current = pred  # feed prediction back

    return np.array(preds), np.array(targets)


# ============================================================
# Plotting
# ============================================================

def plot_training_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative L2 Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Transolver Training Curves')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_field_comparison(coords, pred, true, timestep, save_path, var_names=None):
    """Plot predicted vs true fields for one timestep."""
    if var_names is None:
        var_names = ['alpha.water', 'Ux', 'Uz']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for j, vname in enumerate(var_names):
        vmin = min(true[:, j].min(), pred[:, j].min())
        vmax = max(true[:, j].max(), pred[:, j].max())

        # Ground truth
        sc0 = axes[j, 0].scatter(coords[:, 0], coords[:, 1],
                                  c=true[:, j], s=0.1, cmap='coolwarm',
                                  vmin=vmin, vmax=vmax)
        axes[j, 0].set_title(f'{vname} — Ground Truth')
        plt.colorbar(sc0, ax=axes[j, 0])

        # Prediction
        sc1 = axes[j, 1].scatter(coords[:, 0], coords[:, 1],
                                  c=pred[:, j], s=0.1, cmap='coolwarm',
                                  vmin=vmin, vmax=vmax)
        axes[j, 1].set_title(f'{vname} — Prediction')
        plt.colorbar(sc1, ax=axes[j, 1])

        # Error
        err = pred[:, j] - true[:, j]
        emax = max(abs(err.min()), abs(err.max()))
        sc2 = axes[j, 2].scatter(coords[:, 0], coords[:, 1],
                                  c=err, s=0.1, cmap='RdBu_r',
                                  vmin=-emax, vmax=emax)
        axes[j, 2].set_title(f'{vname} — Error')
        plt.colorbar(sc2, ax=axes[j, 2])

    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('z')

    plt.suptitle(f'Transolver Prediction (step {timestep})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    os.makedirs(args.output, exist_ok=True)

    # ---- Load data ----
    print("\nLoading data...")
    coords = np.load(os.path.join(args.data_dir, 'coords.npy'))   # (N, 2)
    fields = np.load(os.path.join(args.data_dir, 'fields.npy'))   # (T, N, 3)
    times = np.load(os.path.join(args.data_dir, 'times.npy'))     # (T,)

    N = coords.shape[0]
    T = fields.shape[0]
    print(f"  N_points={N}, N_timesteps={T}")
    print(f"  Time range: [{times[0]:.2f}, {times[-1]:.2f}]")

    # ---- Normalize coordinates to [0,1] ----
    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    coords_norm[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())

    # ---- Split data ----
    # Discard transient, split into train/test
    transient_mask = times >= args.transient_time
    train_mask = (times >= args.transient_time) & (times < args.test_time)
    test_mask = times >= args.test_time

    # Indices for consecutive pairs (input at t, target at t+1)
    all_indices = np.where(transient_mask)[0]
    train_indices = [i for i in all_indices if train_mask[i] and i + 1 < T]
    test_indices = [i for i in all_indices if test_mask[i] and i + 1 < T]

    # Remove last test index if its target would go out of bounds
    train_indices = [i for i in train_indices if i + 1 < T]
    test_indices = [i for i in test_indices if i + 1 < T]

    print(f"  Train samples: {len(train_indices)} (t={times[train_indices[0]]:.2f} to {times[train_indices[-1]]:.2f})")
    print(f"  Test samples:  {len(test_indices)} (t={times[test_indices[0]]:.2f} to {times[test_indices[-1]]:.2f})")

    # ---- Datasets ----
    train_dataset = WaveFieldDataset(coords_norm, fields, train_indices)
    test_dataset = WaveFieldDataset(coords_norm, fields, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ---- Normalizer (fit on training data only) ----
    train_fields = torch.tensor(fields[train_indices], dtype=torch.float32)
    normalizer = FieldNormalizer(train_fields)
    normalizer.cuda()

    print(f"\n  Field normalizer stats:")
    var_names = ['alpha', 'Ux', 'Uz']
    for j, vn in enumerate(var_names):
        print(f"    {vn}: mean={normalizer.mean[0,j].item():.4f}, std={normalizer.std[0,j].item():.4f}")

    # ---- Model ----
    model = Transolver(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        n_head=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        fun_dim=3,       # alpha, Ux, Uz
        out_dim=3,       # predict alpha, Ux, Uz
        slice_num=args.slice_num,
        ref=args.ref,
        unified_pos=bool(args.unified_pos),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: Transolver (Irregular Mesh)")
    print(f"  Parameters: {n_params:,}")
    print(f"  Layers={args.n_layers}, Hidden={args.n_hidden}, Heads={args.n_heads}")
    print(f"  Slices={args.slice_num}, MLP_ratio={args.mlp_ratio}")

    # ---- Wandb ----
    wandb.init(
        project="ocean-wave-surrogate",
        name=f"transolver_h{args.n_hidden}_l{args.n_layers}_s{args.slice_num}",
        config={
            'model': 'Transolver_Irregular_Mesh',
            'n_params': n_params,
            'n_layers': args.n_layers,
            'n_hidden': args.n_hidden,
            'n_heads': args.n_heads,
            'slice_num': args.slice_num,
            'mlp_ratio': args.mlp_ratio,
            'dropout': args.dropout,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'grad_accum': args.grad_accum,
            'epochs': args.epochs,
            'patience': args.patience,
            'transient_time': args.transient_time,
            'test_time': args.test_time,
            'n_points': N,
            'n_train': len(train_indices),
            'n_test': len(test_indices),
        }
    )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader) // args.grad_accum + 1
    )

    loss_fn = RelativeLoss()

    # ---- Training loop ----
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        t0 = timer.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 loss_fn, normalizer, args.grad_accum,
                                 args.max_grad_norm, device)
        val_loss, val_mse = evaluate(model, test_loader, loss_fn,
                                     normalizer, device)

        elapsed = timer.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
            'epoch_time': elapsed,
            'val_rmse_alpha': float(np.sqrt(val_mse[0])),
            'val_rmse_Ux': float(np.sqrt(val_mse[1])),
            'val_rmse_Uz': float(np.sqrt(val_mse[2])),
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output, 'best_model.pt'))
            print(f"  -> New best! Saved model.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ---- Plot training curves ----
    plot_training_curves(train_losses, val_losses,
                         os.path.join(args.output, 'training_curves.png'))

    # ---- Final evaluation ----
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(args.output, 'best_model.pt')))
    model.eval()

    # Single-step evaluation
    ss_loss, ss_mse = evaluate(model, test_loader, loss_fn, normalizer, device)
    print(f"\nSingle-step relative L2: {ss_loss:.6f}")
    for j, vn in enumerate(var_names):
        print(f"  {vn} RMSE: {np.sqrt(ss_mse[j]):.6f}")

    # Autoregressive rollout
    ar_start = test_indices[0]
    ar_steps = min(args.ar_steps, len(test_indices) - 1)
    print(f"\nAutoregressive rollout: {ar_steps} steps from t={times[ar_start]:.2f}")

    preds_ar, targets_ar = autoregressive_rollout(
        model, coords_norm, fields, ar_start, ar_steps, normalizer, device
    )

    # AR errors over time
    ar_rel_errors = []
    for step in range(1, len(preds_ar)):
        diff = np.linalg.norm(preds_ar[step] - targets_ar[step])
        ref = np.linalg.norm(targets_ar[step])
        ar_rel_errors.append(diff / ref if ref > 0 else 0)

    print(f"  AR relative L2 at step 1:  {ar_rel_errors[0]:.6f}")
    print(f"  AR relative L2 at step 10: {ar_rel_errors[min(9, len(ar_rel_errors)-1)]:.6f}")
    print(f"  AR relative L2 at step {ar_steps}: {ar_rel_errors[-1]:.6f}")

    # Plot field comparison at a few timesteps
    for step_idx in [1, 10, min(ar_steps, len(preds_ar)-1)]:
        if step_idx < len(preds_ar):
            plot_field_comparison(
                coords, preds_ar[step_idx], targets_ar[step_idx],
                step_idx,
                os.path.join(args.output, f'field_comparison_step{step_idx}.png')
            )

    # Plot AR error growth
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(ar_rel_errors) + 1), ar_rel_errors, 'b-o', markersize=2)
    ax.set_xlabel('Autoregressive Step')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title('Autoregressive Error Growth')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'ar_error_growth.png'), dpi=150)
    plt.close()

    # ---- Save results summary ----
    results = {
        'model': 'Transolver_Irregular_Mesh',
        'n_params': n_params,
        'n_layers': args.n_layers,
        'n_hidden': args.n_hidden,
        'n_heads': args.n_heads,
        'slice_num': args.slice_num,
        'n_points': N,
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'best_epoch': len(train_losses) - patience_counter,
        'best_val_loss': float(best_val_loss),
        'ss_relative_l2': float(ss_loss),
        'ss_rmse_per_var': {vn: float(np.sqrt(ss_mse[j])) for j, vn in enumerate(var_names)},
        'ar_steps': ar_steps,
        'ar_final_error': float(ar_rel_errors[-1]),
        'ar_errors': [float(e) for e in ar_rel_errors],
    }

    with open(os.path.join(args.output, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Log final results to wandb ----
    wandb.log({
        'ss_relative_l2': float(ss_loss),
        'ar_final_error': float(ar_rel_errors[-1]),
    })
    for j, vn in enumerate(var_names):
        wandb.log({
            f'ss_rmse_{vn}': float(np.sqrt(ss_mse[j])),
        })

    # Upload plots to wandb
    for img_name in ['training_curves.png', 'ar_error_growth.png',
                     'field_comparison_step1.png', 'field_comparison_step10.png']:
        img_path = os.path.join(args.output, img_name)
        if os.path.exists(img_path):
            wandb.log({img_name.replace('.png', ''): wandb.Image(img_path)})

    wandb.finish()

    print(f"\nResults saved to {args.output}/")
    print("Done!")


if __name__ == '__main__':
    main()