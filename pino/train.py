"""
train.py — Training loop for FNO / PINO.

Features:
    - Single-step and multi-step rollout loss
    - Optional physics loss (continuity equation)
    - Checkpoint save/resume
    - Optional wandb logging

Usage:
    python train.py                                    # FNO baseline
    python train.py train.physics_weight=0.1           # PINO
    python train.py train.rollout_steps=4              # multi-step rollout
    python train.py train.resume=checkpoints/epoch_100.pt  # resume
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import WaveDataset
from model import FNO2d


# ─── Physics Loss ────────────────────────────────────────────

def compute_continuity_residual(pred, mask, dx, dz):
    """
    Compute incompressibility residual: ∂Ux/∂x + ∂Uz/∂z = 0

    Args:
        pred: (B, C, nx, nz) predicted fields, C=[alpha, Ux, Uz]
        mask: (nx, nz) terrain mask
        dx, dz: grid spacings

    Returns:
        scalar: mean squared residual over fluid domain
    """
    ux = pred[:, 1]   # (B, nx, nz)
    uz = pred[:, 2]   # (B, nx, nz)

    # Central differences
    dux_dx = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / (2.0 * dx)
    duz_dz = (uz[:, 1:-1, 2:] - uz[:, 1:-1, :-2]) / (2.0 * dz)

    residual = dux_dx + duz_dz  # (B, nx-2, nz-2)

    # Mask: only penalize inside fluid domain
    m = mask[1:-1, 1:-1].unsqueeze(0)  # (1, nx-2, nz-2)
    masked_residual = residual * m

    # Mean squared residual
    n_fluid = m.sum().clamp(min=1.0)
    return (masked_residual ** 2).sum() / (pred.shape[0] * n_fluid)


# ─── Training ────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, cfg, mask_tensor, dx, dz):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_phys_loss = 0.0
    n_batches = 0

    for x, target in loader:
        x = x.to(cfg.hardware.device)
        target = target.to(cfg.hardware.device)

        if cfg.train.rollout_steps == 1:
            # ── Single-step ──
            pred = model(x)
            data_loss = criterion(pred * mask_tensor, target * mask_tensor)
        else:
            # ── Multi-step rollout ──
            data_loss = 0.0
            current_input = x  # (B, n_input*C + 1, nx, nz)
            mask_ch = x[:, -1:, :, :]  # (B, 1, nx, nz) — mask channel
            n_fields = cfg.model.n_out_ch

            for s in range(cfg.train.rollout_steps):
                pred = model(current_input)
                step_target = target[:, s]  # (B, C, nx, nz)
                data_loss += criterion(pred * mask_tensor, step_target * mask_tensor)

                # Shift window: drop oldest frame, append prediction
                # current_input without mask: (B, n_input*C, nx, nz)
                old_frames = current_input[:, n_fields:-1, :, :]  # drop first frame, keep rest
                current_input = torch.cat([old_frames, pred, mask_ch], dim=1)

            data_loss /= cfg.train.rollout_steps

        # Physics loss
        phys_loss = torch.tensor(0.0, device=x.device)
        if cfg.train.physics_weight > 0:
            # Denormalize pred for physical residual computation
            pred_denorm = pred  # TODO: denormalize when physics loss is enabled
            phys_loss = compute_continuity_residual(pred_denorm, mask_tensor[0, 0], dx, dz)

        loss = cfg.train.data_weight * data_loss + cfg.train.physics_weight * phys_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_phys_loss += phys_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "data_loss": total_data_loss / n_batches,
        "phys_loss": total_phys_loss / n_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, cfg, mask_tensor):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, target in loader:
        x = x.to(cfg.hardware.device)
        target = target.to(cfg.hardware.device)

        if cfg.train.rollout_steps == 1:
            pred = model(x)
            loss = criterion(pred * mask_tensor, target * mask_tensor)
        else:
            loss = 0.0
            current_input = x
            mask_ch = x[:, -1:, :, :]
            n_fields = cfg.model.n_out_ch
            for s in range(cfg.train.rollout_steps):
                pred = model(current_input)
                step_target = target[:, s]
                loss += criterion(pred * mask_tensor, step_target * mask_tensor)
                old_frames = current_input[:, n_fields:-1, :, :]
                current_input = torch.cat([old_frames, pred, mask_ch], dim=1)
            loss /= cfg.train.rollout_steps

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def save_checkpoint(model, optimizer, scheduler, epoch, cfg, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "cfg": OmegaConf.to_container(cfg),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from {path}, starting at epoch {start_epoch}")
    return start_epoch


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = cfg.hardware.device

    # ── Data ──
    proc = cfg.data.processed_dir
    train_ds = WaveDataset(
        os.path.join(proc, "train_data.npy"),
        os.path.join(proc, "terrain_mask.npy"),
        n_input_steps=cfg.data.n_input_steps,
        rollout_steps=cfg.train.rollout_steps,
    )
    test_ds = WaveDataset(
        os.path.join(proc, "test_data.npy"),
        os.path.join(proc, "terrain_mask.npy"),
        n_input_steps=cfg.data.n_input_steps,
        rollout_steps=cfg.train.rollout_steps,
    )
    # Share normalization stats from training set
    test_ds.mean = train_ds.mean
    test_ds.std = train_ds.std

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory,
    )

    # Terrain mask as tensor for masking losses
    mask_np = np.load(os.path.join(proc, "terrain_mask.npy"))
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,nx,nz)

    # Grid spacings for physics loss
    grid_x = np.load(os.path.join(proc, "grid_x.npy"))
    grid_z = np.load(os.path.join(proc, "grid_z.npy"))
    dx = grid_x[1] - grid_x[0]
    dz = grid_z[1] - grid_z[0]

    # ── Model ──
    model = FNO2d(
        n_in_ch=cfg.data.n_input_ch,
        n_out_ch=cfg.model.n_out_ch,
        modes1=cfg.model.modes1,
        modes2=cfg.model.modes2,
        width=cfg.model.width,
        n_layers=cfg.model.n_layers,
    ).to(device)

    print(f"\nModel parameters: {model.count_params():,}")

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    scheduler = None
    if cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs
        )
    elif cfg.train.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma
        )

    criterion = nn.MSELoss()

    # ── Resume ──
    start_epoch = 0
    if cfg.train.resume:
        start_epoch = load_checkpoint(cfg.train.resume, model, optimizer, scheduler)

    # ── Wandb ──
    if cfg.logging.use_wandb:
        import wandb
        run_name = cfg.logging.run_name or f"fno_m{cfg.model.modes1}_w{cfg.model.width}_pw{cfg.train.physics_weight}"
        wandb.init(project=cfg.logging.project, name=run_name, config=OmegaConf.to_container(cfg))

    # ── Training loop ──
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nStarting training: epochs {start_epoch} → {cfg.train.epochs}")
    print(f"  physics_weight = {cfg.train.physics_weight} "
          f"({'PINO' if cfg.train.physics_weight > 0 else 'FNO'})")
    print(f"  rollout_steps = {cfg.train.rollout_steps}")
    print()

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, cfg, mask_tensor, dx, dz
        )
        val_loss = validate(model, test_loader, criterion, cfg, mask_tensor)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Logging
        log_str = (
            f"Epoch {epoch:4d} | "
            f"train {train_metrics['loss']:.6f} "
            f"(data {train_metrics['data_loss']:.6f}, phys {train_metrics['phys_loss']:.6f}) | "
            f"val {val_loss:.6f} | "
            f"lr {lr:.2e} | {elapsed:.1f}s"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, cfg,
                            os.path.join(cfg.train.checkpoint_dir, "best.pt"))
            log_str += " ★"
        print(log_str)

        if cfg.logging.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/data_loss": train_metrics["data_loss"],
                "train/phys_loss": train_metrics["phys_loss"],
                "val/loss": val_loss,
                "lr": lr,
            })

        # Periodic checkpoint
        if (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg,
                            os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch+1}.pt"))

    # Final save
    save_checkpoint(model, optimizer, scheduler, cfg.train.epochs - 1, cfg,
                    os.path.join(cfg.train.checkpoint_dir, "final.pt"))
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")

    if cfg.logging.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()