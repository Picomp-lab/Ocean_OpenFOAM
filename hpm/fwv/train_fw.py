"""
train_fw.py — fw: prior + self-feedback 输入分支, 单步 (teacher forcing)。

    输入 = [prior(t) | x_f * m (| m)]   x_f = GT(t-1)   (teacher forcing)
           model.mask_channel=false -> arm1, 2F 通道  (最小 CFG)
           model.mask_channel=true  -> arm2, 2F+1 通道 (+ 显式告知)
    base = prior(t)                       feedback 是输入分支, 不是残差基座
    预测 = prior(t) + Δ
    R = 1                                 尚未 rollout, 先验证支路本身有没有用

相对 1b (train_prior.py) 只多了一个变量: 自反馈支路 + conditioning dropout。
prior / schema / stats / 划分 / 模型超参全部不变 —— 单变量对照。

判据 (三层, 缺一不可)
--------------------
1. 是否优于 Δ=0        每 epoch 打印 val nRMSE vs 基线, 与 1b 同口径
2. 支路有没有被用上    ‖W_f‖ / ‖W_p‖ (及 arm2 的 ‖w_m‖) 第一层权重范数
3. 支路有没有转化成行为  同一 checkpoint 跑两遍 val:
                          full = m 取真实值 (有 GT(t-1) 就用)
                          none = m 恒 0    (prior-only, 模拟 cold start)
                        gap ≈ 0 -> 支路是死重 (p 太高, 或它本就无信息)
                        gap 显著 -> 两种 regime 确实分化了

已知不覆盖的失效模式
------------------
conditioning dropout 治的是"槽位缺失"(cold start), **不治**"槽位错误"
(exposure bias)。训练喂 GT(t-1)、推理装 X̂(t-1), 分布不匹配依然存在,
rollout 照样可能崩。那是下一个变量, 等这一轮基线稳了再动。
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
from prior_ext import assert_prior_compatible
from schema import ChannelSchema

from data_fw import (PriorFeedbackDataset, assemble, branch_norms,
                     input_dim, sample_mask)


class WeightedMSELoss(nn.Module):
    """与 train.py / train_prior.py 同式: delta 通道上的逐通道加权 MSE。"""

    def __init__(self, weights):
        super().__init__()
        assert len(weights) > 0, "empty loss weights"
        self.register_buffer('weights',
                             torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        return (((pred - target) ** 2) * self.weights[None, None, :]).mean()


def step_metrics(model, coords, prior, x_f, m, gt, criterion, delta_idx,
                 mask_channel=True):
    """单步前向 + 指标。base 是 prior, 与 1b 完全一致 —— 数字可直接对照。

    Returns (loss, se_model, se_base, n)
      se_model: (out_dim,) 模型残差平方和
      se_base : (out_dim,) Δ=0 残差平方和 (即 prior 自身的误差)
    """
    x = assemble(prior, x_f, m, mask_channel)            # (B, N, 2F 或 2F+1)
    delta_pred = model(coords, x)                        # (B, N, out_dim)
    gt_delta = (gt - prior).index_select(-1, delta_idx)
    loss = criterion(delta_pred, gt_delta)

    with torch.no_grad():
        se_model = ((delta_pred - gt_delta) ** 2).sum(dim=(0, 1))
        se_base = (gt_delta ** 2).sum(dim=(0, 1))
        n = gt_delta.shape[0] * gt_delta.shape[1]
    return loss, se_model, se_base, n


def run_epoch(model, loader, coords, criterion, delta_idx, device,
              p_drop=0.0, m_mode="train", mask_channel=True,
              optimizer=None, scheduler=None, max_grad_norm=0.0):
    """m_mode:
        train  按 p_drop 随机 drop (真 t=0 恒 0)
        full   m = has_prev, 不 drop     —— 正常推理条件
        none   m = 0                     —— prior-only, 模拟 cold start
    """
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, nb, ntot = 0.0, 0, 0
    se_m = se_b = None

    for prior, gt_prev, gt, has_prev in loader:
        prior = prior.to(device, non_blocking=True)
        gt_prev = gt_prev.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        has_prev = has_prev.to(device, non_blocking=True)
        cb = coords.unsqueeze(0).expand(prior.shape[0], -1, -1)

        if m_mode == "train":
            m = sample_mask(has_prev, p_drop)
        elif m_mode == "full":
            m = has_prev.reshape(-1, 1, 1).float()
        elif m_mode == "none":
            m = torch.zeros(prior.shape[0], 1, 1, device=device)
        else:
            raise ValueError(f"unknown m_mode {m_mode}")

        if train:
            optimizer.zero_grad()
            loss, sm, sb, n = step_metrics(model, cb, prior, gt_prev, m, gt,
                                           criterion, delta_idx, mask_channel)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                loss, sm, sb, n = step_metrics(model, cb, prior, gt_prev, m, gt,
                                               criterion, delta_idx, mask_channel)

        tot += loss.item(); nb += 1
        se_m = sm if se_m is None else se_m + sm
        se_b = sb if se_b is None else se_b + sb
        ntot += n

    nrmse_model = torch.sqrt(se_m / max(ntot, 1)).cpu().numpy()
    nrmse_base = torch.sqrt(se_b / max(ntot, 1)).cpu().numpy()
    return tot / max(nb, 1), nrmse_model, nrmse_base


@hydra.main(config_path=".", config_name="config_fw", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg.save.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("HPM fw — prior + self-feedback (teacher forcing, single step)")
    print("=" * 68)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    assert_prior_compatible(schema)
    assert cfg.data.window == 0, \
        f"fw 仍走 window=0 路径 (无时间窗), 当前 {cfg.data.window}"

    F = schema.field_dim
    mask_ch = bool(cfg.model.mask_channel)
    in_dim = input_dim(F, mask_ch)
    p_drop = float(cfg.train.cond_dropout)
    arm = "arm2 (+mask channel)" if mask_ch else "arm1 (只清零)"
    print(f"{arm}   输入宽度: prior({F}) + feedback({F})"
          f"{' + mask(1)' if mask_ch else ''} = {in_dim}")
    print(f"conditioning dropout p = {p_drop}  "
          f"(真 t=0 恒 m=0, 故实际 m=0 比例略高于 p)")

    spectral_embedding = np.load(Path(cfg.data.dir) / "lbo" / "lbo_eigenvectors.npy")
    print(f"LBO eigenvectors: {spectral_embedding.shape}")

    train_chunks = expand_range(cfg.data.train_chunk_range)
    val_chunks = expand_range(cfg.data.val_chunk_range)
    print(f"Train chunks: {train_chunks}   Val chunks: {val_chunks}")

    stats = resolve_stats(cfg.data.dir, train_chunks, schema)

    print("Loading train set...")
    train_set = PriorFeedbackDataset(cfg.data.dir, cfg.data.prior_dir,
                                     train_chunks, schema, stats)
    print("Loading val set...")
    val_set = PriorFeedbackDataset(cfg.data.dir, cfg.data.prior_dir,
                                   val_chunks, schema, stats)
    print(f"Train samples: {len(train_set)}   Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers,
                            pin_memory=True)

    coords = load_coords(cfg.data.dir).to(device)

    # window=0 下 field_dim 就是"输入特征宽度", 模型对这些通道是什么不知情 ——
    # 故 2F+1 是合法用法, hpm_model.py 一行不用改。
    model = HPM(
        space_dim=3, field_dim=in_dim, out_dim=schema.out_dim,
        window=0,
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

    print("\n判据: (1) val nRMSE < Δ=0 基线  (2) ‖w_m‖ 非死  "
          "(3) full vs none 有 gap\n")
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        tr_loss, tr_nr, tr_nb = run_epoch(
            model, train_loader, coords, criterion, delta_idx, device,
            p_drop=p_drop, m_mode="train", mask_channel=mask_ch,
            optimizer=optimizer, scheduler=scheduler,
            max_grad_norm=cfg.train.max_grad_norm)

        # 两遍 val: 正常条件 vs 强制 cold start。差值就是支路的行为增益。
        va_loss, va_nr, va_nb = run_epoch(
            model, val_loader, coords, criterion, delta_idx, device,
            m_mode="full", mask_channel=mask_ch)
        vn_loss, vn_nr, _ = run_epoch(
            model, val_loader, coords, criterion, delta_idx, device,
            m_mode="none", mask_channel=mask_ch)

        lr = optimizer.param_groups[0]['lr']
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" mem={mem_alloc:.1f}/{mem_reserved:.1f}GB"
        else:
            mem_alloc = mem_reserved = 0.0
            mem_str = ""

        bn = branch_norms(model, F, mask_ch)

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} val={va_loss:.6f} "
              f"lr={lr:.2e} ({time.time()-t0:.1f}s){mem_str}")
        print("   val nRMSE  " + "  ".join(
            f"{n}: {m:.3f}/{b:.3f}{'✓' if m < b else '✗'}"
            for n, m, b in zip(dnames, va_nr, va_nb)) + "   (模型/基线)")
        print("   feedback gap (none-full)  " + "  ".join(
            f"{n}: {a-f:+.4f}" for n, f, a in zip(dnames, va_nr, vn_nr))
            + "   >0 = 支路在起作用")
        print("   ‖W‖ " + "  ".join(f"{k}={v:.3f}" for k, v in bn.items()))

        if HAS_WANDB and cfg.wandb.enabled:
            log = {"train_loss": tr_loss, "val_loss": va_loss,
                   "val_loss_prior_only": vn_loss,
                   "lr": lr, "epoch": epoch,
                   "gpu_mem_alloc_gb": mem_alloc,
                   "gpu_mem_reserved_gb": mem_reserved}
            for k, v in bn.items():
                log[f"wnorm/{k}"] = v
            for n, m, b, a in zip(dnames, va_nr, va_nb, vn_nr):
                log[f"val_nrmse/{n}"] = float(m)
                log[f"val_nrmse_base/{n}"] = float(b)
                log[f"val_nrmse_prior_only/{n}"] = float(a)
                log[f"val_gain/{n}"] = float(b / max(m, 1e-12))
                log[f"feedback_gap/{n}"] = float(a - m)
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