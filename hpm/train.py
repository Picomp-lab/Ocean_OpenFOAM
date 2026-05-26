"""
Training script for HPM time-stepping model (Hydra config).

Features:
  - Weighted MSE loss (alpha+U at 1.0, p_rgh+nut at 0.1)
  - Gradient clipping
  - wandb logging
  - Checkpoint save/resume (latest.pt + best.pt)
  - No AMP (causes zero gradients in this setup)
  - No DDP (NVLink unavailable)
"""

import os
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
from dataset import WaveDataset, compute_stats, load_coords


# ============================================================
# Loss
# ============================================================

class WeightedMSELoss(nn.Module):
    """Per-channel weighted MSE. Channels: [alpha, Ux, Uy, Uz, p_rgh, nut]"""

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1]
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weighted = mse * self.weights[None, None, :]
        return weighted.mean()


# ============================================================
# Train / Validate
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    max_grad_norm, coords, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for fields, target in loader:
        fields = fields.to(device)
        target = target.to(device)
        coords_batch = coords.unsqueeze(0).expand(fields.shape[0], -1, -1)

        optimizer.zero_grad()
        pred = model(coords_batch, fields)
        loss = criterion(pred, target)
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
def validate(model, loader, criterion, device, coords):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for fields, target in loader:
        fields = fields.to(device)
        target = target.to(device)
        coords_batch = coords.unsqueeze(0).expand(fields.shape[0], -1, -1)

        pred = model(coords_batch, fields)
        loss = criterion(pred, target)
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

    # ---- Load LBO eigenvectors ----
    lbo_dir = Path(cfg.data.dir) / "lbo"
    eigvec_path = lbo_dir / "lbo_eigenvectors.npy"
    assert eigvec_path.exists(), f"LBO eigenvectors not found: {eigvec_path}"
    spectral_embedding = np.load(eigvec_path)
    print(f"Loaded LBO eigenvectors: {spectral_embedding.shape}")

    # ---- Stats ----
    stats_path = Path(cfg.data.dir) / "stats.npy"
    if not stats_path.exists():
        print("Computing dataset statistics from training chunks...")
        stats = compute_stats(cfg.data.dir, cfg.data.train_chunks)
    else:
        stats = np.load(stats_path)
        print(f"Loaded stats: mean={stats[0]}, std={stats[1]}")

    # ---- Dataset ----
    train_set = WaveDataset(cfg.data.dir, cfg.data.train_chunks, cfg.data.window, stats)
    val_set = WaveDataset(cfg.data.dir, cfg.data.val_chunks, cfg.data.window, stats)
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=True)

    # ---- Load coords once ----
    coords = load_coords(cfg.data.dir).to(device)
    print(f"Coords loaded: {coords.shape}, on {coords.device}")

    # ---- Model ----
    model = HPM(
        space_dim=3,
        field_dim=6,
        out_dim=6,
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
    criterion = WeightedMSELoss().to(device)

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
    if HAS_WANDB and cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # ---- Training loop ----
    print(f"\nStarting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, cfg.train.max_grad_norm, coords,
                                     scheduler)
        val_loss = validate(model, val_loader, criterion, device, coords)

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} val={val_loss:.6f} "
              f"lr={lr:.2e} ({elapsed:.1f}s)")

        # wandb
        if HAS_WANDB and cfg.wandb.enabled:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                        "lr": lr, "epoch": epoch})

        # Save latest
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        }
        torch.save(ckpt, latest_path)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  → New best: {best_val:.6f}")

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    if HAS_WANDB and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()