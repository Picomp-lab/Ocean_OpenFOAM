"""
Training script — Hydra config, single GPU, wandb.

Usage:
    python -u train.py
    python -u train.py model.n_hidden=256 train.lr=5e-5
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from transolver_pp import TransolverPP
from dataset import WaveDataset


def weighted_mse(pred, target, w_aux=0.1):
    w = torch.ones(6, device=pred.device)
    w[4:] = w_aux
    return ((pred - target) ** 2 * w).mean()


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    total, F = 0.0, 6
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        if cfg.train.rollout_steps == 1:
            loss = weighted_mse(model(inp), tgt, cfg.train.loss_weight_aux)
        else:
            coords, fields_w = inp[..., :3], inp[..., 3:]
            loss = 0.0
            for s in range(cfg.train.rollout_steps):
                pred = model(torch.cat([coords, fields_w], -1))
                loss = loss + weighted_mse(pred, tgt[:, s], cfg.train.loss_weight_aux)
                p = pred.detach() if s < cfg.train.rollout_steps - 1 else pred
                fields_w = torch.cat([fields_w[..., F:], p], -1)
            loss = loss / cfg.train.rollout_steps
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, cfg, device):
    model.eval()
    total, F = 0.0, 6
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        if cfg.train.rollout_steps == 1:
            loss = weighted_mse(model(inp), tgt, cfg.train.loss_weight_aux)
        else:
            coords, fields_w = inp[..., :3], inp[..., 3:]
            loss = 0.0
            for s in range(cfg.train.rollout_steps):
                pred = model(torch.cat([coords, fields_w], -1))
                loss = loss + weighted_mse(pred, tgt[:, s], cfg.train.loss_weight_aux)
                fields_w = torch.cat([fields_w[..., F:], pred], -1)
            loss = loss / cfg.train.rollout_steps
        total += loss.item()
    return total / len(loader)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda")
    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg))

    # Data
    W = cfg.train.window
    train_set = WaveDataset(cfg.data.dir, cfg.data.train_chunks, W, cfg.train.rollout_steps)
    val_set = WaveDataset(cfg.data.dir, cfg.data.val_chunks, W, cfg.train.rollout_steps)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = TransolverPP(window=W, **cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n:,} | Train: {len(train_set)} | Val: {len(val_set)}")

    # Resume
    start_epoch, best_val = 0, float("inf")
    latest = save_dir / "latest.pt"
    if cfg.train.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_val = ckpt["epoch"] + 1, ckpt["best_val"]
        print(f"Resumed epoch {start_epoch}, best_val={best_val:.6f}")

    # Train
    for epoch in range(start_epoch, cfg.train.epochs):
        t_loss = train_epoch(model, train_loader, optimizer, cfg, device)
        v_loss = val_epoch(model, val_loader, cfg, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"E{epoch:03d} train={t_loss:.6f} val={v_loss:.6f} lr={lr:.2e}")
        if cfg.wandb.enabled:
            wandb.log({"train_loss": t_loss, "val_loss": v_loss, "lr": lr, "epoch": epoch})

        ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch, best_val=best_val)
        torch.save(ckpt, latest)
        if v_loss < best_val:
            best_val = v_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ↑ best={best_val:.6f}")

    print(f"Done. best_val={best_val:.6f}")
    if cfg.wandb.enabled: wandb.finish()


if __name__ == "__main__":
    main()