"""
train_fw_ss.py — fw: prior + self-feedback, R 步真 BPTT + scheduled sampling。

相对 train_fw.py (单步 teacher forcing) 的变化: 训练时展开 R 步 rollout,
每步用 scheduled sampling 决定 feedback 槽喂 GT 还是喂模型自己上一步的 pred。
治的是 exposure bias (部署 rollout 近岸误差积累), 不动 arch, cold start 从头训。

每步机制 (序列内 r = 0..R-1, 预测帧 t+r)
----------------------------------------
    r = 0    x_f = 0, m = 0            冷启动 (与部署一致, 不喂任何历史)
    r >= 1   掷骰子 (per-sample):
               prob p     -> x_f = gt(t+r-1)    常数, 无梯度 -> **断链**
               prob 1-p   -> x_f = pred(t+r-1)  带梯度 -> **保链** (真 BPTT)
    base = prior(t+r);  pred = prior + Δ;  loss 用 gt(t+r) 算。
    R 步等权平均, backward 沿"保链"的步回传。

p 退火 (cold start 必须)
------------------------
    p = 喂 GT 的概率。从 p_start 线性退到 p_end, 占前 anneal_frac 的 epochs,
    之后恒 p_end。默认 1.0 -> 0.1, 前 60%。初期几乎全喂 GT 稳住, 末期 90%
    的步喂自己的预测, 贴部署。

"喂 GT 的步为什么断链": 梯度只沿"张量的来路"回传; GT 是数据/常数, 没有来路,
所以那一步的输入与上一步 forward 无计算图连接, 梯度到此为止。torch.where
(pred, gt) 天然实现: 选 pred 的样本有梯度, 选 gt 的样本无 —— per-sample。

超参 (config_fw 里没有的, 用 .get 默认值; CLI 用 +train.R=4 覆盖)
    train.R              rollout 步数           默认 4
    train.ss_p_start     退火起点 (喂 GT 概率)  默认 1.0
    train.ss_p_end       退火终点               默认 0.1
    train.ss_anneal_frac 退火占 epochs 比例      默认 0.6
"""

import contextlib
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

from dataset import expand_range, load_coords, resolve_stats    # 父 hpm/
from hpm_model import HPM                                       # 父 hpm/
from prior_ext import assert_prior_compatible                  # fwv/
from schema import ChannelSchema                               # 父 hpm/

from data_fw import assemble, branch_norms, input_dim          # fwv/  (复用拼接)
from data_fw_ss import PriorSeqDataset                         # fwv/


class WeightedMSELoss(nn.Module):
    """与 train_fw 同式: delta 通道上的逐通道加权 MSE。"""

    def __init__(self, weights):
        super().__init__()
        assert len(weights) > 0, "empty loss weights"
        self.register_buffer('weights',
                             torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        return (((pred - target) ** 2) * self.weights[None, None, :]).mean()


def reconstruct(prior_r, delta, delta_idx):
    """pred = prior + Δ, Δ 只加在 delta_idx 通道 (frozen 通道保持 = prior)。
    out-of-place index_add: 对 delta 可导, prior_r 无需 grad。"""
    return prior_r.index_add(-1, delta_idx, delta)


def p_at(epoch, epochs, p_start, p_end, anneal_frac):
    """线性退火: 前 anneal_frac*epochs 从 p_start 退到 p_end, 之后恒 p_end。"""
    a = max(anneal_frac * epochs, 1e-9)
    if epoch >= a:
        return p_end
    return p_start + (p_end - p_start) * (epoch / a)


def run_epoch(model, loader, coords, criterion, delta_idx, device, R,
              mask_channel, p=0.0, train=True,
              optimizer=None, scheduler=None, max_grad_norm=0.0):
    """R 步 rollout。

    train=True:  scheduled sampling (每步 per-sample 掷骰子), 真 BPTT, 回传。
    train=False: 纯 rollout (恒喂 pred, 无骰子) —— 部署条件, 出 val nRMSE。

    Returns (mean_loss, nrmse_model(out_dim,), nrmse_base(out_dim,))
      nrmse 在 R 步 + 所有序列 + 所有 cell 上聚合 (normalized delta 空间)。
    """
    model.train() if train else model.eval()
    tot, nb, ntot = 0.0, 0, 0
    se_m = se_b = None

    # val (train=False) 全程 no_grad —— 否则 R 步 rollout 会建完整计算图且不释放,
    # 显存爆得比 train 还狠 (无 backward 清图)。这是 val 阶段 OOM 的根因。
    grad_ctx = contextlib.nullcontext() if train else torch.no_grad()

    for prior_seq, gt_seq in loader:
        prior_seq = prior_seq.to(device, non_blocking=True)   # (B,R,N,F)
        gt_seq = gt_seq.to(device, non_blocking=True)
        B, _, N, F = prior_seq.shape
        cb = coords.unsqueeze(0).expand(B, -1, -1)

        if train:
            optimizer.zero_grad()

        x_f = torch.zeros(B, N, F, device=device)   # 冷启动初值
        step_losses = []

        with grad_ctx:
            for r in range(R):
                prior_r = prior_seq[:, r]                # (B,N,F)
                m_val = 0.0 if r == 0 else 1.0
                m = torch.full((B, 1, 1), m_val, device=device)

                x = assemble(prior_r, x_f, m, mask_channel)   # 复用拼接, 与推理一致
                delta = model(cb, x)                          # (B,N,out)
                pred_r = reconstruct(prior_r, delta, delta_idx)   # (B,N,F) normalized

                gt_r = gt_seq[:, r]
                gt_delta = (gt_r - prior_r).index_select(-1, delta_idx)
                step_losses.append(criterion(delta, gt_delta))

                with torch.no_grad():
                    sm = ((delta - gt_delta) ** 2).sum(dim=(0, 1))
                    sb = (gt_delta ** 2).sum(dim=(0, 1))
                    se_m = sm if se_m is None else se_m + sm
                    se_b = sb if se_b is None else se_b + sb
                    ntot += B * N

                # 准备下一步 (r+1) 的 x_f = 帧 (t+r) 的值
                if r < R - 1:
                    if train:
                        # per-sample 掷骰子: prob p 喂 GT(断链), 否则喂 pred(保链)
                        use_pred = (torch.rand(B, 1, 1, device=device) >= p)
                        x_f = torch.where(use_pred, pred_r, gt_r)
                    else:
                        x_f = pred_r                 # val: 纯 rollout, 恒喂 pred

        loss = torch.stack(step_losses).mean()       # R 步等权

        if train:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        tot += loss.item(); nb += 1

    nrmse_model = torch.sqrt(se_m / max(ntot, 1)).cpu().numpy()
    nrmse_base = torch.sqrt(se_b / max(ntot, 1)).cpu().numpy()
    return tot / max(nb, 1), nrmse_model, nrmse_base


@hydra.main(config_path=".", config_name="config_fw", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)   # 允许 .get 读 config 里没有的 SS 键
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg.save.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    R = int(cfg.train.get('R', 4))
    p_start = float(cfg.train.get('ss_p_start', 1.0))
    p_end = float(cfg.train.get('ss_p_end', 0.1))
    anneal_frac = float(cfg.train.get('ss_anneal_frac', 0.6))

    print("=" * 68)
    print(f"HPM fw — scheduled sampling + 真 BPTT (R={R}, cold start)")
    print("=" * 68)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())
    assert_prior_compatible(schema)
    assert cfg.data.window == 0, \
        f"fw 走 window=0, 当前 {cfg.data.window}"

    F = schema.field_dim
    mask_ch = bool(cfg.model.mask_channel)
    in_dim = input_dim(F, mask_ch)
    print(f"{'arm2(+mask)' if mask_ch else 'arm1'}  输入宽度={in_dim} "
          f"(F={F}, out_dim={schema.out_dim})")
    print(f"scheduled sampling: p {p_start} -> {p_end}, 退火占前 "
          f"{anneal_frac*100:.0f}% epochs  (p = 喂 GT 概率)")
    print(f"每样本 = 连续 {R} 帧; 序列第 0 步恒冷启动 m=0; 真 BPTT (喂 pred 保链)")
    print("注意: ss 版**不使用** cond_dropout —— cold start 由每序列 r=0 强制 "
          "m=0 结构处理 (约 1/R 的 step 为冷启动)。config 里的 cond_dropout "
          "是 train_fw.py(单步版)的活参数, 在此为死参数, 不参与训练。")

    spectral_embedding = np.load(Path(cfg.data.dir) / "lbo" / "lbo_eigenvectors.npy")
    print(f"LBO eigenvectors: {spectral_embedding.shape}")

    train_chunks = expand_range(cfg.data.train_chunk_range)
    val_chunks = expand_range(cfg.data.val_chunk_range)
    print(f"Train chunks: {train_chunks}   Val chunks: {val_chunks}")

    stats = resolve_stats(cfg.data.dir, train_chunks, schema)

    print("Loading train set...")
    train_set = PriorSeqDataset(cfg.data.dir, cfg.data.prior_dir,
                                train_chunks, schema, stats, R)
    print("Loading val set...")
    val_set = PriorSeqDataset(cfg.data.dir, cfg.data.prior_dir,
                              val_chunks, schema, stats, R)
    print(f"Train seq: {len(train_set)}   Val seq: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers,
                            pin_memory=True)

    coords = load_coords(cfg.data.dir).to(device)

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

    print("\n判据: val = R 步纯 rollout nRMSE (部署条件); 目标是压近岸积累\n")
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        p = p_at(epoch, cfg.train.epochs, p_start, p_end, anneal_frac)

        tr_loss, tr_nr, tr_nb = run_epoch(
            model, train_loader, coords, criterion, delta_idx, device, R,
            mask_ch, p=p, train=True,
            optimizer=optimizer, scheduler=scheduler,
            max_grad_norm=cfg.train.max_grad_norm)

        va_loss, va_nr, va_nb = run_epoch(
            model, val_loader, coords, criterion, delta_idx, device, R,
            mask_ch, train=False)

        lr = optimizer.param_groups[0]['lr']
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" mem={mem_alloc:.1f}/{mem_reserved:.1f}GB"
        else:
            mem_alloc = mem_reserved = 0.0; mem_str = ""

        bn = branch_norms(model, F, mask_ch)

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} val_rollout={va_loss:.6f} "
              f"p={p:.2f} lr={lr:.2e} ({time.time()-t0:.1f}s){mem_str}")
        print("   val rollout nRMSE  " + "  ".join(
            f"{n}: {m:.3f}/{b:.3f}{'✓' if m < b else '✗'}"
            for n, m, b in zip(dnames, va_nr, va_nb)) + "   (模型/基线)")
        print("   ‖W‖ " + "  ".join(f"{k}={v:.3f}" for k, v in bn.items()))

        if HAS_WANDB and cfg.wandb.enabled:
            log = {"train_loss": tr_loss, "val_rollout_loss": va_loss,
                   "ss_p": p, "lr": lr, "epoch": epoch,
                   "gpu_mem_alloc_gb": mem_alloc,
                   "gpu_mem_reserved_gb": mem_reserved}
            for k, v in bn.items():
                log[f"wnorm/{k}"] = v
            for n, m, b in zip(dnames, va_nr, va_nb):
                log[f"val_rollout_nrmse/{n}"] = float(m)
                log[f"val_rollout_nrmse_base/{n}"] = float(b)
                log[f"val_rollout_gain/{n}"] = float(b / max(m, 1e-12))
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

    print(f"\nDone. Best val rollout loss: {best_val:.6f}")
    if HAS_WANDB and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()