"""
Training script — Hydra config, DDP, wandb, AMP.

Usage:
    torchrun --nproc_per_node=2 train.py
    torchrun --nproc_per_node=2 train.py model.n_hidden=256 train.lr=5e-5
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def train_epoch(model, loader, optimizer, scaler, cfg, device):
    model.train()
    total = 0.0
    W, F = cfg.train.window, 6
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.amp.autocast("cuda"):
            if cfg.train.rollout_steps == 1:
                loss = weighted_mse(model(inp), tgt, cfg.train.loss_weight_aux)
            else:
                coords = inp[..., :3]                          # (B, N, 3)
                fields_w = inp[..., 3:]                        # (B, N, 6*W)
                loss = 0.0
                for s in range(cfg.train.rollout_steps):
                    pred = model(torch.cat([coords, fields_w], -1))  # (B, N, 6)
                    loss = loss + weighted_mse(pred, tgt[:, s], cfg.train.loss_weight_aux)
                    # Shift window: drop oldest frame, append prediction
                    p = pred.detach() if s < cfg.train.rollout_steps - 1 else pred
                    fields_w = torch.cat([fields_w[..., F:], p], -1)
                loss = loss / cfg.train.rollout_steps
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, cfg, device):
    model.eval()
    total = 0.0
    W, F = cfg.train.window, 6
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.amp.autocast("cuda"):
            if cfg.train.rollout_steps == 1:
                loss = weighted_mse(model(inp), tgt, cfg.train.loss_weight_aux)
            else:
                coords = inp[..., :3]
                fields_w = inp[..., 3:]
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
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = (rank == 0)

    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if is_main and cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg))

    # Data
    W = cfg.train.window
    train_set = WaveDataset(cfg.data.dir, cfg.data.train_chunks, W, cfg.train.rollout_steps)
    val_set = WaveDataset(cfg.data.dir, cfg.data.val_chunks, W, cfg.train.rollout_steps)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              sampler=DistributedSampler(train_set, shuffle=True),
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size,
                            sampler=DistributedSampler(val_set, shuffle=False),
                            num_workers=4, pin_memory=True)

    # Model
    model = TransolverPP(window=W, **cfg.model).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda")

    if is_main:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Params: {n:,} | Train: {len(train_set)} | Val: {len(val_set)}")

    # Resume
    start_epoch, best_val = 0, float("inf")
    latest = save_dir / "latest.pt"
    if cfg.train.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_val = ckpt["epoch"] + 1, ckpt["best_val"]
        if is_main: print(f"Resumed epoch {start_epoch}, best_val={best_val:.6f}")

    # Train
    for epoch in range(start_epoch, cfg.train.epochs):
        train_loader.sampler.set_epoch(epoch)
        t_loss = train_epoch(model, train_loader, optimizer, scaler, cfg, device)
        v_loss = val_epoch(model, val_loader, cfg, device)
        scheduler.step()

        losses = torch.tensor([t_loss, v_loss], device=device)
        dist.all_reduce(losses, op=dist.ReduceOp.AVG)
        t_loss, v_loss = losses.tolist()

        if is_main:
            lr = optimizer.param_groups[0]["lr"]
            print(f"E{epoch:03d} train={t_loss:.6f} val={v_loss:.6f} lr={lr:.2e}")
            if cfg.wandb.enabled:
                wandb.log({"train_loss": t_loss, "val_loss": v_loss, "lr": lr, "epoch": epoch})

            ckpt = dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, best_val=best_val)
            torch.save(ckpt, latest)
            if v_loss < best_val:
                best_val = v_loss
                ckpt["best_val"] = best_val
                torch.save(ckpt, save_dir / "best.pt")
                print(f"  ↑ best={best_val:.6f}")

    if is_main:
        print(f"Done. best_val={best_val:.6f}")
        if cfg.wandb.enabled: wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
