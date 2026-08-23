"""
Training script for HPM time-stepping model (Hydra config, schema-driven).

Features:
  - Channel schema from config (single source of truth — see schema.py)
  - Per-channel weighted MSE loss, weights from schema (delta channels only)
  - Shared rollout closure (schema.advance_window) — identical to inference
  - Gradient clipping
  - wandb logging
  - Checkpoint save/resume (latest.pt + best.pt)
  - No AMP (causes zero gradients in this setup)
  - No DDP (NVLink unavailable)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from hpm_model import HPM
from schema import ChannelSchema, advance_window, register_autoname_resolver
from dataset import WaveDataset, load_coords, resolve_stats, expand_range

# Must run at import time, BEFORE @hydra.main resolves hydra.run.dir
# (which interpolates ${wandb.name} -> ${autoname:}).
register_autoname_resolver()


# ============================================================
# Loss
# ============================================================

class WeightedMSELoss(nn.Module):
    """Per-channel weighted MSE over DELTA channels only.

    Weights come from schema.delta_loss_weights() — declared in config.yaml,
    never hardcoded here. Weights act in z-normalized delta space (relative
    importance, not physical scale).
    """

    def __init__(self, weights):
        super().__init__()
        assert len(weights) > 0, "empty loss weights"
        self.register_buffer('weights',
                             torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        mse = (pred - target) ** 2                     # (B, N, out_dim)
        w = self.weights[None, None, :]                # (1, 1, out_dim)
        return (mse * w).mean()


# ============================================================
# Train / Validate
# ============================================================

def multistep_rollout_loss(model, coords, window_fields, future_frames,
                           criterion, schema):
    """
    Multi-step autoregressive rollout loss (full BPTT, no detach).

    Loss is computed on delta channels only (schema.delta_indices); frozen
    channels are carried through the window unchanged by advance_window.

    Args:
        model:         HPM model
        coords:        (B, N, 3) — spatial coordinates
        window_fields: (B, N, W*F) — initial window, flattened
        future_frames: (B, R, N, F) — R ground truth future frames
        criterion:     loss over (B, N, out_dim) delta tensors
        schema:        ChannelSchema

    Returns:
        mean of per-step losses
    """
    B, R, N, F = future_frames.shape
    assert F == schema.field_dim, (
        f"future_frames has {F} channels, schema expects {schema.field_dim}")
    delta_idx = torch.as_tensor(schema.delta_indices,
                                device=future_frames.device)

    current_window = window_fields                              # (B, N, W*F)
    total_loss = 0.0

    for step in range(R):
        # Predict delta for delta channels only: (B, N, out_dim)
        delta_pred = model(coords, current_window)

        # Ground truth delta, restricted to delta channels
        current_frame = current_window[..., -schema.field_dim:]
        gt_delta = future_frames[:, step] - current_frame       # (B, N, F)
        gt_delta_sel = gt_delta.index_select(-1, delta_idx)     # (B, N, out_dim)

        total_loss = total_loss + criterion(delta_pred, gt_delta_sel)

        # Shared closure: scatter delta, freeze non-delta, shift window.
        _, current_window = advance_window(current_window, delta_pred, schema)

    return total_loss / R


def train_one_epoch(model, loader, optimizer, criterion, device,
                    max_grad_norm, coords, schema, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for fields, future in loader:
        fields = fields.to(device)                             # (B, N, W*F)
        future = future.to(device)                             # (B, R, N, F)
        coords_batch = coords.unsqueeze(0).expand(fields.shape[0], -1, -1)

        optimizer.zero_grad()
        loss = multistep_rollout_loss(model, coords_batch, fields, future,
                                      criterion, schema)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, coords, schema):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for fields, future in loader:
        fields = fields.to(device)
        future = future.to(device)
        coords_batch = coords.unsqueeze(0).expand(fields.shape[0], -1, -1)

        loss = multistep_rollout_loss(model, coords_batch, fields, future,
                                      criterion, schema)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# Main
# ============================================================

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg.save.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Print environment ----
    print("=" * 60)
    print("HPM Training")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    print("=" * 60)

    # ---- Channel schema (single source of truth) ----
    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())

    # ---- Load LBO eigenvectors ----
    lbo_dir = Path(cfg.data.dir) / "lbo"
    eigvec_path = lbo_dir / "lbo_eigenvectors.npy"
    assert eigvec_path.exists(), f"LBO eigenvectors not found: {eigvec_path}"
    spectral_embedding = np.load(eigvec_path)
    print(f"Loaded LBO eigenvectors: {spectral_embedding.shape}")

    # ---- Stats (versioned by chunk set + channel signature) ----
    train_chunks = expand_range(cfg.data.train_chunk_range)
    val_chunks = expand_range(cfg.data.val_chunk_range)
    print(f"Train chunks: {train_chunks}, Val chunks: {val_chunks}")
    stats = resolve_stats(cfg.data.dir, train_chunks, schema)

    # ---- Dataset ----
    rollout_steps = cfg.train.get('rollout_steps', 4)
    train_set = WaveDataset(cfg.data.dir, train_chunks, cfg.data.window,
                            schema, stats=stats, rollout_steps=rollout_steps)
    val_set = WaveDataset(cfg.data.dir, val_chunks, cfg.data.window,
                          schema, stats=stats, rollout_steps=rollout_steps)
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"Rollout steps: {rollout_steps}")

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=True)

    # ---- Load coords once ----
    coords = load_coords(cfg.data.dir).to(device)
    print(f"Coords loaded: {coords.shape}, on {coords.device}")

    # ---- Model (dims derived from schema) ----
    model = HPM(
        space_dim=3,
        field_dim=schema.field_dim,
        out_dim=schema.out_dim,
        window=cfg.data.window,
        n_hidden=cfg.model.n_hidden,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        freq_num=cfg.model.freq_num,
        dropout=cfg.model.dropout,
        mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.spectral_pos_dim,
        spectral_embedding=spectral_embedding,
        use_ckpt=cfg.model.use_ckpt,
        max_grad_norm=cfg.train.max_grad_norm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Optimizer / Scheduler / Loss ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.train.lr, epochs=cfg.train.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    loss_weights = schema.delta_loss_weights()
    print(f"Loss weights (delta channels "
          f"{[schema.names[i] for i in schema.delta_indices]}): {loss_weights}")
    criterion = WeightedMSELoss(loss_weights).to(device)

    # ---- Resume ----
    start_epoch = 0
    best_val = float('inf')
    latest_path = save_dir / "latest.pt"

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    # ---- wandb ----
    # 运行名 = autoname (schema diff) + override_dirname (超参 diff, 可为空)
    if HAS_WANDB and cfg.wandb.enabled:
        from hydra.core.hydra_config import HydraConfig
        od = HydraConfig.get().job.override_dirname
        wandb_run_name = cfg.wandb.name + (f"_{od}" if od else "")
        print(f"wandb run name: {wandb_run_name}")
        wandb.init(project=cfg.wandb.project, name=wandb_run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # ---- Training loop ----
    print(f"\nStarting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, cfg.train.max_grad_norm, coords,
                                     schema, scheduler)
        val_loss = validate(model, val_loader, criterion, device, coords, schema)

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Peak GPU memory (cumulative since start; epoch-1 steady state is
        # usually THE peak). alloc = tensors; reserved = allocator footprint
        # (what nvidia-smi shows) — use reserved for OOM headroom judgement.
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" mem={mem_alloc:.1f}/{mem_reserved:.1f}GB"
        else:
            mem_alloc = mem_reserved = 0.0
            mem_str = ""

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} val={val_loss:.6f} "
              f"lr={lr:.2e} ({elapsed:.1f}s){mem_str}")

        # wandb
        if HAS_WANDB and cfg.wandb.enabled:
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "lr": lr,
                       "epoch": epoch,
                       "gpu_mem_alloc_gb": mem_alloc,
                       "gpu_mem_reserved_gb": mem_reserved,
                       })

        # Update best_val first
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        # Save latest (always has current best_val)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        }
        torch.save(ckpt, latest_path)

        # Save best
        if is_best:
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  → New best: {best_val:.6f}")

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    if HAS_WANDB and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()