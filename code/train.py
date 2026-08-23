"""
train.py — 唯一训练入口 (single training entry point)。

合并了原先的 train.py (纯 HPM) / train_prior.py / train_fw_ss.py。三者是同一个
rollout 循环上的三个参数点。

两条模型线, 由 data.window 单参数区分
------------------------------------
    data.window >= 3   纯 HPM (pure HPM)         残差加在**自己上一帧**
                       输入 = W 帧滑窗            无外部先验
    data.window == 0   HPM + FUNWAVE (fwv)       残差加在 **prior(t)** 上
                       输入 = prior [| 反馈槽]    prior_dir 必填

这两者不是自由组合: window>0 意味着有历史帧可作基座, window=0 意味着没有 ——
所以基座由 window 推导, 不设独立开关。若将来真要试 "时间窗 + 外部先验",
那是通道语义未定义的新格子, 届时才把基座拆成独立参数。

四个轴
------
    data.window      基座 + 输入形状 (见上)
    rollout.feedback none | self      自反馈槽, 仅 fwv 线
    rollout.R        rollout 展开步数
    rollout.ss       scheduled sampling 总开关; false -> p 恒 0, p_* 全忽略
    rollout.p_*      SS 的退火参数, 喂 GT 的概率 (仅 ss=true 时生效)

关键观察: 纯 HPM 的"恒喂 pred"就是 p=0, teacher forcing 就是 p=1 ——
三种 schedule 是**同一个参数的三个取值**, 不是三种机制。这是合并成立的根据。

统一循环
--------
    for r in range(R):
        base  = policy.base_at(batch, state, r)
        x     = policy.make_input(batch, state, r)
        delta = model(coords, x)
        loss_r = criterion(delta, (gt[r] - base)[delta_idx])
        pred  = base + scatter(delta)
        state = policy.advance(state, pred, gt[r], p, train)

"喂 GT 的步为什么断链": 梯度只沿张量的来路回传; GT 是数据/常数, 没有来路,
所以那一步的输入与上一步 forward 无计算图连接。torch.where(pred, gt) 天然实现
per-sample 的断链/保链。

数据层 (拼接 assemble / 重建 reconstruct / 两个 dataset) 全在 dataset.py,
训练与推理共用同一份 —— 两边拼法必须逐字节一致, 分开写迟早漂。

复现命令见 config.yaml 头部的复现表。
"""

import contextlib
import os
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


def wandb_ready(cfg) -> bool:
    """wandb.enabled=true 只是「想用」, 这里判断「能不能用」。

    没装 / 没登录时 wandb.init 会停在交互式提示上等输入 —— 在 SLURM 里就是一个
    白占着 GPU 熬到 --time 超时的作业, 日志里什么都看不出来。所以开跑前先判掉,
    判不过就只在 log 里写清楚原因, 训练照常跑, 后面所有 wandb 调用全部跳过。

    WANDB_MODE=offline 不需要 key (落本地 wandb/ 目录, 事后 wandb sync 补传)。
    """
    if not cfg.wandb.enabled:
        return False

    if not HAS_WANDB:
        print("[wandb] 环境里没装 wandb —— 本次不记录, 训练照常。")
        print("[wandb] 要用: pip install wandb  (或在仓库根目录跑 ./setup.sh)")
        return False

    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if mode in ("offline", "dryrun"):
        print(f"[wandb] WANDB_MODE={mode} —— 只落本地, 事后 wandb sync 补传。")
        return True
    if mode == "disabled":
        print("[wandb] WANDB_MODE=disabled —— 本次不记录, 训练照常。")
        return False

    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            key = wandb.api.api_key          # 读 ~/.netrc / wandb settings
        except Exception:
            key = None
    if not key:
        print("[wandb] 没找到 API key (环境变量 WANDB_API_KEY 和 ~/.netrc 里都没有)"
              " —— 本次不记录, 训练照常。")
        print("[wandb] 要记录的话三选一, 然后重投:")
        print("[wandb]   wandb login                 # 在登录节点做, 计算节点不一定通外网")
        print("[wandb]   export WANDB_MODE=offline   # 先离线记, 事后 wandb sync")
        print("[wandb]   sbatch run.sh wandb.enabled=false")
        return False

    return True


from dataset import (PriorSeqDataset, WindowSeqDataset, assemble, branch_norms,
                     expand_range, input_dim, load_coords, reconstruct,
                     resolve_stats)
from hpm_model import HPM
from schema import ChannelSchema, advance_window, auto_run_name


# ============================================================
# 运行名 —— 本身也是派生量
# ============================================================

def run_name(cfg):
    """两条线各自的命名空间。名字只编码**相对默认配置的 diff**。

    纯 HPM   走 schema.auto_run_name (编码 schema diff): hpm_bl_h128 / hpm_no-nut_h128
    fwv 线   hpm_fw[_nofb][_aU|_aUx|_aUz]_h{n}

    规则:
      nofb      关掉自反馈槽 (默认 self, 故只在关掉时出现)
      aU 系列   **列出被 alpha 加权的速度通道** —— 与纯 HPM 线同向 (加权了才显示):
                  Ux+Uz -> aU     只 Ux -> aUx     只 Uz -> aUz     都没有 -> 省略
      h{n}      模型容量常驻 (不是 diff, 但与 auto_run_name 形状对齐, 且它是
                将来要横向比较的一列, 埋进 override_dirname 不好认)

    R / lr 等超参**不在这里编码** —— 走 CLI 时 hydra 的 override_dirname 已经
    把它们编进目录名与 run 名了, 再编一遍会变成 hpm_fw_R8_rollout.R-8。
    """
    if int(cfg.data.window) > 0:
        return auto_run_name(cfg)

    schema = ChannelSchema.from_cfg(cfg, verbose=False)
    parts = ["hpm", "fw"]
    if str(cfg.rollout.feedback) != "self":
        parts.append("nofb")

    # αU 标记: 被加权的速度通道。两个都加 -> 'aU'; 只加一个 -> 'aUx' / 'aUz'
    wU = [n for n, w in zip(schema.names, schema.alpha_weighted)
          if w and n in ("Ux", "Uy", "Uz")]
    if len(wU) >= 2:
        parts.append("aU")
    elif len(wU) == 1:
        parts.append(f"a{wU[0]}")

    parts.append(f"h{cfg.model.n_hidden}")
    name = "_".join(parts)
    suffix = str(cfg.wandb.get("name_suffix", "") or "")
    return f"{name}_{suffix}" if suffix else name


def register_resolvers():
    repo_root = Path(__file__).resolve().parent.parent

    if not OmegaConf.has_resolver("repo"):
        OmegaConf.register_new_resolver(
            "repo", lambda: str(repo_root))

    if not OmegaConf.has_resolver("runname"):
        OmegaConf.register_new_resolver(
            "runname", lambda *, _root_: run_name(_root_))


register_resolvers()


# ============================================================
# Loss
# ============================================================

class WeightedMSELoss(nn.Module):
    """delta 通道上的逐通道加权 MSE。权重来自 schema, 从不在这里硬编码。

    权重作用在 z-normalized delta 空间 (相对重要性, 非物理量纲)。
    """

    def __init__(self, weights):
        super().__init__()
        assert len(weights) > 0, "empty loss weights"
        self.register_buffer('weights',
                             torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        return (((pred - target) ** 2) * self.weights[None, None, :]).mean()


# ============================================================
# scheduled sampling
# ============================================================

def p_at(epoch, epochs, p_start, p_end, anneal_frac):
    """线性退火: 前 anneal_frac*epochs 从 p_start 退到 p_end, 之后恒 p_end。

    p = 喂 GT 的概率。p=1 全 teacher forcing, p=0 全自回归 (部署条件)。
    anneal_frac<=0 -> 恒 p_end。

    本函数不看 rollout.ss —— 开关在 main 里一次性把 p_* 归零 (见那里的注释),
    所以这里永远只需要处理 "p_* 已经是最终值" 这一种情况。
    """
    a = max(anneal_frac * epochs, 1e-9)
    if epoch >= a:
        return p_end
    return p_start + (p_end - p_start) * (epoch / a)


def _pick(pred, gt_r, p, train):
    """per-sample 掷骰子: prob p 喂 GT (断链), 否则喂 pred (保链, 真 BPTT)。

    val 恒喂 pred —— 纯 rollout 就是部署条件。
    p<=0 时短路成恒喂 pred: 省一次 rand, 且让纯 HPM 线的计算图与合并前的
    multistep_rollout_loss 逐算子一致。
    """
    if not train or p <= 0.0:
        return pred
    if p >= 1.0:
        return gt_r
    use_pred = (torch.rand(pred.shape[0], 1, 1, device=pred.device) >= p)
    return torch.where(use_pred, pred, gt_r)


# ============================================================
# 两条线的策略
# ============================================================

class SelfStatePolicy:
    """纯 HPM: 基座 = 窗口末帧 (自身状态), 状态 = W 帧滑动窗口。

    基座与状态在这条线上是**耦合**的 —— 窗口本身既是模型输入也是残差基座。
    窗口移位在 advance() 内联 (reconstruct + torch.cat); 推理侧 (vis.py) 用
    schema.advance_window 做**等价**操作。两者数值等价但**不是同一实现** —— 改
    任一处需手动对齐另一处 (schema.advance_window 因需在 pred/移位间插 SS 的
    _pick 而无法被训练直接复用)。
    """

    def __init__(self, schema, window):
        assert window >= 3, \
            f"纯 HPM 需要 window>=3 (macro/dt/dt2 有限差分), 当前 {window}"
        self.schema = schema
        self.window = window
        self.F = schema.field_dim

    def model_kwargs(self):
        return dict(field_dim=self.F, window=self.window)

    def init_state(self, batch, device):
        return batch["window"]                          # (B, N, W*F)

    def base_at(self, batch, state, r):
        return state[..., -self.F:]

    def make_input(self, batch, state, r):
        return state

    def advance(self, state, pred, gt_r, p, train):
        frame = _pick(pred, gt_r, p, train)
        return torch.cat([state[..., self.F:], frame], dim=-1)

    def diagnostics(self, model):
        return {}


class PriorPolicy:
    """HPM + FUNWAVE: 基座 = prior(t+r) (外部先验), 状态 = 单帧反馈槽 x_f。

    基座与状态在这条线上是**解耦**的: prior 从 t=0 全程可用, 所以 cold start
    缺口只落在反馈这一条输入支路上。

    feedback:
        none  输入 = [prior],          F 通道     (无反馈基线)
        self  输入 = [prior | x_f*m],  2F 通道

    屏蔽由 x_f*m 完成 —— 算术恒等 (w·0=0), 不需要学。m 是 per-frame 的 (B,1,1)
    而非 per-point: 上一帧要么算过要么没算过, 不存在一帧里有些点有有些点没有。

    序列第 0 步恒 m=0 —— 与部署一致, 不喂任何历史。约 1/R 的 step 是冷启动,
    这是结构性处理, 不需要 conditioning dropout。
    """

    def __init__(self, schema, feedback="self"):
        assert feedback in ("none", "self"), f"未知 feedback={feedback}"
        self.schema = schema
        self.feedback = feedback
        self.F = schema.field_dim

    @property
    def in_dim(self):
        return self.F if self.feedback == "none" else input_dim(self.F)

    def model_kwargs(self):
        # window=0: 模型对这些通道是什么不知情, 只当作 in_dim 宽的特征向量,
        # 所以 2F / 2F+1 都是合法用法, hpm_model.py 一行不用改。
        return dict(field_dim=self.in_dim, window=0)

    def init_state(self, batch, device):
        if self.feedback == "none":
            return None
        p0 = batch["prior_seq"]
        return torch.zeros(p0.shape[0], p0.shape[2], self.F, device=device)

    def base_at(self, batch, state, r):
        return batch["prior_seq"][:, r]

    def make_input(self, batch, state, r):
        prior_r = batch["prior_seq"][:, r]
        if self.feedback == "none":
            return prior_r
        m_val = 0.0 if r == 0 else 1.0                  # r=0 恒冷启动
        m = torch.full((prior_r.shape[0], 1, 1), m_val, device=prior_r.device)
        return assemble(prior_r, state, m)

    def advance(self, state, pred, gt_r, p, train):
        if self.feedback == "none":
            return None
        return _pick(pred, gt_r, p, train)

    def diagnostics(self, model):
        """第一层权重按输入分支切开取范数 —— 支路有没有被用上。"""
        if self.feedback == "none":
            return {}
        return branch_norms(model, self.F)


def build_policy(cfg, schema):
    """基座由 data.window 推导 —— 见文件头。"""
    w = int(cfg.data.window)
    if w > 0:
        assert str(cfg.rollout.feedback) == "none", (
            "纯 HPM (window>0) 不支持自反馈槽: 窗口本身已是状态, 再加反馈槽等于"
            "两个冗余状态源, 从未训练验证过。设 rollout.feedback=none。")
        # SS 是 fwv 线的默认值 (p 1.0->0.1)。纯 HPM 线的历史行为是恒喂 pred,
        # 若默认值漏过来, GT 帧会被静默混进滑窗 —— 不报错, 只是结果不再是那条
        # 基线。这里挡住: 只写 data.window=6 而忘了关 SS 就直接失败。
        assert not bool(cfg.rollout.ss), (
            "纯 HPM (window>0) 必须 rollout.ss=false: 这条线的历史行为是恒喂 "
            "pred (p 恒 0), SS 会把 GT 帧混进滑窗, 静默偏离基线。"
            "用 `./run.sh pure` 已代劳。")
        return SelfStatePolicy(schema, w)
    return PriorPolicy(schema, feedback=str(cfg.rollout.feedback))


# ============================================================
# 统一 epoch
# ============================================================

def run_epoch(model, loader, coords, criterion, delta_idx, device, policy, R,
              p=0.0, train=True, optimizer=None, scheduler=None,
              max_grad_norm=0.0):
    """R 步 rollout。

    train=True   按 p 掷骰子 (scheduled sampling), 真 BPTT, 回传
    train=False  恒喂 pred (纯 rollout = 部署条件), 出 val nRMSE

    Returns (mean_loss, nrmse_model (out_dim,), nrmse_base (out_dim,))
      nrmse_base = Δ=0 基线, 即基座自身的误差:
        fwv 线   -> prior 本身的 nRMSE (与 diag_prior.py 同口径)
        纯 HPM   -> persistence 基线
      两者都在归一化 delta 空间聚合 (std=1, 故 RMS == nRMSE)。
    """
    model.train() if train else model.eval()
    tot, nb, ntot = 0.0, 0, 0
    se_m = se_b = None

    # val 全程 no_grad —— 否则 R 步 rollout 会建完整计算图且不释放 (无 backward
    # 清图), 显存爆得比 train 还狠。这是历史上 val 阶段 OOM 的根因。
    grad_ctx = contextlib.nullcontext() if train else torch.no_grad()

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        gt_seq = batch["gt_seq"]                         # (B, R, N, F)
        B, _, N, _ = gt_seq.shape
        cb = coords.unsqueeze(0).expand(B, -1, -1)

        if train:
            optimizer.zero_grad()

        state = policy.init_state(batch, device)
        step_losses = []

        with grad_ctx:
            for r in range(R):
                base = policy.base_at(batch, state, r)        # (B, N, F)
                x = policy.make_input(batch, state, r)        # (B, N, in_dim)
                delta = model(cb, x)                          # (B, N, out_dim)

                gt_r = gt_seq[:, r]
                gt_delta = (gt_r - base).index_select(-1, delta_idx)
                step_losses.append(criterion(delta, gt_delta))

                with torch.no_grad():
                    sm = ((delta - gt_delta) ** 2).sum(dim=(0, 1))
                    sb = (gt_delta ** 2).sum(dim=(0, 1))
                    se_m = sm if se_m is None else se_m + sm
                    se_b = sb if se_b is None else se_b + sb
                    ntot += B * N

                if r < R - 1:
                    pred = reconstruct(base, delta, delta_idx)
                    state = policy.advance(state, pred, gt_r, p, train)

        loss = torch.stack(step_losses).mean()               # R 步等权

        if train:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        tot += loss.item()
        nb += 1

    nrmse_model = torch.sqrt(se_m / max(ntot, 1)).cpu().numpy()
    nrmse_base = torch.sqrt(se_b / max(ntot, 1)).cpu().numpy()
    return tot / max(nb, 1), nrmse_model, nrmse_base


# ============================================================
# 数据
# ============================================================

def build_datasets(cfg, schema, stats, R):
    """按 data.window 选数据层。两者都产出 dict batch (见 dataset.py)。"""
    train_chunks = expand_range(cfg.data.train_chunk_range)
    val_chunks = expand_range(cfg.data.val_chunk_range)
    print(f"Train chunks: {train_chunks}   Val chunks: {val_chunks}")

    if int(cfg.data.window) > 0:
        def mk(cs):
            return WindowSeqDataset(cfg.data.dir, cs, cfg.data.window,
                                    schema, stats=stats, rollout_steps=R)
    else:
        assert cfg.data.get("prior_dir", None), \
            "fwv 线 (window=0) 需要 data.prior_dir"

        def mk(cs):
            return PriorSeqDataset(cfg.data.dir, cfg.data.prior_dir,
                                   cs, schema, stats, R)

    print("Loading train set...")
    train_set = mk(train_chunks)
    print("Loading val set...")
    val_set = mk(val_chunks)
    print(f"Train samples: {len(train_set)}   Val samples: {len(val_set)}")
    return train_set, val_set


# ============================================================
# Main
# ============================================================

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg.save.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    R = int(cfg.rollout.R)
    # SS 开关**只在这一处**生效: 关掉就把 p_* 全部归零, 下游 (打印 / R=1 警告 /
    # p_at / _pick 的 p<=0 短路) 因此自动一致, 不必各自再判一次 ss。
    ss = bool(cfg.rollout.ss)
    p_start = float(cfg.rollout.p_start) if ss else 0.0
    p_end = float(cfg.rollout.p_end) if ss else 0.0
    anneal_frac = float(cfg.rollout.anneal_frac) if ss else 0.0
    line = "纯 HPM (pure HPM)" if int(cfg.data.window) > 0 \
        else "HPM + FUNWAVE (fwv)"

    sched = (f"SS p {p_start}->{p_end} (前 {anneal_frac*100:.0f}% epochs)"
             if ss else "SS off (p 恒 0, 全自回归)")
    print("=" * 68)
    print(f"{line}  |  window={cfg.data.window}  feedback={cfg.rollout.feedback}"
          f"  R={R}  {sched}")
    print("=" * 68)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    schema = ChannelSchema.from_cfg(cfg)
    print(schema.describe())

    policy = build_policy(cfg, schema)
    mk = policy.model_kwargs()
    print(f"policy={type(policy).__name__}  模型输入宽度={mk['field_dim']}  "
          f"window={mk['window']}  out_dim={schema.out_dim}")
    if R == 1 and p_start != p_end:
        print("[warn] R=1 时没有 r>=1 的步, p 退火不产生任何作用 —— "
              "这组 p_* 是惰性参数。")

    eig_path = Path(cfg.data.dir) / "lbo" / "lbo_eigenvectors.npy"
    assert eig_path.exists(), \
        f"LBO eigenvectors not found: {eig_path} (先跑 lbo.sh —— 训练的前置步骤)"
    spectral_embedding = np.load(eig_path)
    print(f"LBO eigenvectors: {spectral_embedding.shape}")

    train_chunks = expand_range(cfg.data.train_chunk_range)
    stats = resolve_stats(cfg.data.dir, train_chunks, schema)
    train_set, val_set = build_datasets(cfg, schema, stats, R)

    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers,
                            pin_memory=True)

    coords = load_coords(cfg.data.dir).to(device)

    model = HPM(
        space_dim=3, out_dim=schema.out_dim, **mk,
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

    # ---- resume ----
    start_epoch, best_val = 0, float('inf')
    latest_path = save_dir / "latest.pt"
    if latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        best_val = ck.get('best_val', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")

    use_wandb = wandb_ready(cfg)
    if use_wandb:
        from hydra.core.hydra_config import HydraConfig
        od = HydraConfig.get().job.override_dirname
        wname = cfg.wandb.name + (f"_{od}" if od else "")
        print(f"wandb run name: {wname}")
        try:
            wandb.init(project=cfg.wandb.project, name=wname,
                       config=OmegaConf.to_container(cfg, resolve=True))
        except Exception as e:
            # init 还是可能挂 (计算节点不通外网 / 服务端 5xx)。别让它带走整个训练。
            use_wandb = False
            print(f"[wandb] init 失败, 本次不记录, 训练照常: {type(e).__name__}: {e}")

    print("\n判据: val = R 步纯 rollout nRMSE (部署条件) vs Δ=0 基线\n")
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        p = p_at(epoch, cfg.train.epochs, p_start, p_end, anneal_frac)

        tr_loss, _, _ = run_epoch(
            model, train_loader, coords, criterion, delta_idx, device,
            policy, R, p=p, train=True,
            optimizer=optimizer, scheduler=scheduler,
            max_grad_norm=cfg.train.max_grad_norm)
        va_loss, va_nr, va_nb = run_epoch(
            model, val_loader, coords, criterion, delta_idx, device,
            policy, R, train=False)

        lr = optimizer.param_groups[0]['lr']
        # alloc = 张量本身; reserved = 分配器占用 (nvidia-smi 看到的值,
        # 判断 OOM 余量应看它)。max_* 是自启动以来的累计峰值。
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" mem={mem_alloc:.1f}/{mem_reserved:.1f}GB"
        else:
            mem_alloc = mem_reserved = 0.0
            mem_str = ""

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} val_rollout={va_loss:.6f} "
              f"p={p:.2f} lr={lr:.2e} ({time.time()-t0:.1f}s){mem_str}")
        print("   val nRMSE  " + "  ".join(
            f"{n}: {m:.3f}/{b:.3f}{'✓' if m < b else '✗'}"
            for n, m, b in zip(dnames, va_nr, va_nb)) + "   (模型/基线)")

        bn = policy.diagnostics(model)
        if bn:
            print("   ‖W‖ " + "  ".join(f"{k}={v:.3f}" for k, v in bn.items()))

        if use_wandb:
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
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()