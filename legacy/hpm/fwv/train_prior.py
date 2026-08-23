"""
train_prior.py — 1b capability check: prior 场 -> CFD 场 的单步映射。

    输入 = prior(t)      无时间窗、无自反馈
    base = prior(t)
    预测 = prior(t) + Δ
    R = 1                无反馈回路, R>1 等价扩大 batch

与 train.py 完全隔离: 不共用 rollout 路径, 现有 E0/E1 实验不受影响。
唯一共用的是 schema / stats / 模型定义。

核心判据 —— 不是 "优于现行 baseline" (通道数与时间建模都变了, 不可比),
而是 **优于 Δ=0**, 即 prior 本身。每个 epoch 打印:

    per-channel nRMSE:  模型  vs  Δ=0 基线

归一化空间里 Δ_norm = (GT - prior)/std(GT), 故其 RMS 就等于 diag_prior.py
报的 nRMSE —— 两边数字可直接对照。diag 给出的 Δ=0 基线 (raw, chunk 均值):
    alpha 0.42   Ux 0.92   Uz 0.96   p_rgh 0.69
模型把这些压到多少, 就是残差可学性的直接度量。
"""

import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dataset import expand_range, load_coords, resolve_stats
from hpm_model import HPM
from prior_ext import PriorPairDataset, assert_prior_compatible
from schema import ChannelSchema


class WeightedMSELoss(nn.Module):
    """与 train.py 同式: delta 通道上的逐通道加权 MSE。"""

    def __init__(self, weights):
        super().__init__()
        assert len(weights) > 0, "empty loss weights"
        self.register_buffer('weights',
                             torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        return (((pred - target) ** 2) * self.weights[None, None, :]).mean()


def step_metrics(model, coords, prior, gt, criterion, delta_idx):
    """单步前向 + 指标。

    Returns (loss, se_model, se_base, n)
      se_model: (out_dim,) 模型残差平方和
      se_base : (out_dim,) Δ=0 残差平方和 (即 prior 自身的误差)
    两者除以 n 开方即 nRMSE (归一化空间下 std=1, 故 RMS == nRMSE)。
    """
    delta_pred = model(coords, prior)                    # (B, N, out_dim)
    gt_delta = (gt - prior).index_select(-1, delta_idx)  # (B, N, out_dim)
    loss = criterion(delta_pred, gt_delta)

    with torch.no_grad():
        se_model = ((delta_pred - gt_delta) ** 2).sum(dim=(0, 1))
        se_base = (gt_delta ** 2).sum(dim=(0, 1))
        n = gt_delta.shape[0] * gt_delta.shape[1]
    return loss, se_model, se_base, n


def run_epoch(model, loader, coords, criterion, delta_idx, device,
              optimizer=None, scheduler=None, max_grad_norm=0.0):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, nb = 0.0, 0
    se_m = se_b = None
    ntot = 0

    for prior, gt in loader:
        prior = prior.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        cb = coords.unsqueeze(0).expand(prior.shape[0], -1, -1)

        if train:
            optimizer.zero_grad()
            loss, sm, sb, n = step_metrics(model, cb, prior, gt,
                                           criterion, delta_idx)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                loss, sm, sb, n = step_metrics(model, cb, prior, gt,
                                               criterion, delta_idx)

        tot += loss.item(); nb += 1
        se_m = sm if se_m is None else se_m + sm
        se_b = sb if se_b is None else se_b + sb
        ntot += n

    nrmse_model = torch.sqrt(se_m / max(ntot, 1)).cpu().numpy()
    nrmse_base = torch.sqrt(se_b / max(ntot, 1)).cpu().numpy()
    return tot / max(nb, 1), nrmse_model, nrmse_base


@hydra.main(config_path=".", config_name="config_prior", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg.save.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("HPM capability check — 1b: prior-only, single step")
    print("=" * 68)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    assert_prior_compatible(schema)
    assert cfg.data.window == 0, \
        f"1b 模式要求 data.window=0 (无时间特征), 当前 {cfg.data.window}"

    spectral_embedding = np.load(Path(cfg.data.dir) / "lbo" / "lbo_eigenvectors.npy")
    print(f"LBO eigenvectors: {spectral_embedding.shape}")

    train_chunks = expand_range(cfg.data.train_chunk_range)
    val_chunks = expand_range(cfg.data.val_chunk_range)
    print(f"Train chunks: {train_chunks}   Val chunks: {val_chunks}")

    # stats 从 GT 训练 chunk 计算; prior 用同一套 (同物理空间, 必须同尺度)
    stats = resolve_stats(cfg.data.dir, train_chunks, schema)

    print("Loading train set...")
    train_set = PriorPairDataset(cfg.data.dir, cfg.data.prior_dir,
                                 train_chunks, schema, stats)
    print("Loading val set...")
    val_set = PriorPairDataset(cfg.data.dir, cfg.data.prior_dir,
                               val_chunks, schema, stats)
    print(f"Train samples: {len(train_set)}   Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers,
                            pin_memory=True)

    coords = load_coords(cfg.data.dir).to(device)

    model = HPM(
        space_dim=3, field_dim=schema.field_dim, out_dim=schema.out_dim,
        window=0,                                   # 1b: 无时间特征
        n_hidden=cfg.model.n_hidden, n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads, freq_num=cfg.model.freq_num,
        dropout=cfg.model.dropout, mlp_ratio=cfg.model.mlp_ratio,
        spectral_pos_dim=cfg.model.spectral_pos_dim,
        spectral_embedding=spectral_embedding,
        use_ckpt=cfg.model.use_ckpt,
        max_grad_norm=cfg.train.max_grad_norm,
    ).to(device)
    print(f"Model parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.train.lr, epochs=cfg.train.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1)
    criterion = WeightedMSELoss(schema.delta_loss_weights()).to(device)
    delta_idx = torch.as_tensor(schema.delta_indices, device=device)
    dnames = [schema.names[i] for i in schema.delta_indices]

    start_epoch, best_val = 0, float('inf')
    latest_path = save_dir / "latest.pt"
    if latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1; best_val = ck.get('best_val', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    if HAS_WANDB and cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    print("\n判据: val nRMSE 是否低于 Δ=0 基线 (prior 本身)。低于 = 残差可学。\n")
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        tr_loss, tr_nr, tr_nb = run_epoch(
            model, train_loader, coords, criterion, delta_idx, device,
            optimizer, scheduler, cfg.train.max_grad_norm)
        va_loss, va_nr, va_nb = run_epoch(
            model, val_loader, coords, criterion, delta_idx, device)

        lr = optimizer.param_groups[0]['lr']
        # 显存: alloc = 张量本身; reserved = 分配器占用 (nvidia-smi 看到的值,
        # 判断 OOM 余量应看它)。max_* 是自启动以来的累计峰值。
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" mem={mem_alloc:.1f}/{mem_reserved:.1f}GB"
        else:
            mem_alloc = mem_reserved = 0.0
            mem_str = ""

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} val={va_loss:.6f} "
              f"lr={lr:.2e} ({time.time()-t0:.1f}s){mem_str}")
        print("   val nRMSE  " + "  ".join(
            f"{n}: {m:.3f}/{b:.3f}{'✓' if m < b else '✗'}"
            for n, m, b in zip(dnames, va_nr, va_nb)) + "   (模型/基线)")

        if HAS_WANDB and cfg.wandb.enabled:
            log = {"train_loss": tr_loss, "val_loss": va_loss,
                   "lr": lr, "epoch": epoch,
                   "gpu_mem_alloc_gb": mem_alloc,
                   "gpu_mem_reserved_gb": mem_reserved}
            for n, m, b in zip(dnames, va_nr, va_nb):
                log[f"val_nrmse/{n}"] = float(m)
                log[f"val_nrmse_base/{n}"] = float(b)
                log[f"val_gain/{n}"] = float(b / max(m, 1e-12))
            wandb.log(log)

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss

        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(), "epoch": epoch,
              "best_val": best_val}
        torch.save(ck, latest_path)
        if is_best:
            torch.save(ck, save_dir / "best.pt")
            print(f"  → New best: {best_val:.6f}")

    print(f"\nDone. Best val loss: {best_val:.6f}")
    if HAS_WANDB and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()