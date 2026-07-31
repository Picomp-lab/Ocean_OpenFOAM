"""
data_fw.py — self-feedback 分支的数据层。

相对 prior_ext.py (1b) 的唯一变化: 样本多带一个 X̂(t-1) 槽位的内容。

    1b        输入 = [prior(t)]                        F 通道
    fw arm1   输入 = [prior(t) | x_f * m]              2F 通道
    fw arm2   输入 = [prior(t) | x_f * m | m]          2F+1 通道

其中 x_f 是自反馈槽位:
    训练 (teacher forcing)  x_f = GT(t-1)
    推理                    x_f = 模型上一步的输出 X̂(t-1)

设计决定 (前几轮定的, 记在这里免得漂):
  - x_f 是**输入分支**, 不是残差基座。base 仍然是 prior ——
    prior 从 t=0 全程可用, 所以 cold start 缺口只落在这一条输入支路上,
    这正是 conditioning dropout 定义良好的唯一情形。
  - 屏蔽由 `x_f * m` 完成 (算术, 不用学); mask 通道只负责**告知**模型
    "这次是屏蔽, 不是巧合", 让它有能力对两种 regime 分化行为。
  - m 是 per-frame 抽的 (B,1,1), 不是 per-point —— 上一帧要么算过要么没算过,
    不存在"一帧里有些点有、有些点没有"。
  - 真 t=0 强制 m=0: 那是免费的、分布最真实的 cold start 样本。
    因此 m=0 的实际比例略高于 cond_dropout, 扫参时记住这点。

拼接写成函数、训练与推理共用 —— 两边拼法必须逐字节一致, 分开写迟早漂。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import apply_alpha_weighting          # 父目录 hpm/dataset.py
from prior_ext import (PRIOR_CHANNELS, assert_prior_compatible,   # noqa: F401
                       load_prior, load_prior_valid, prior_indices)


# ------------------------------------------------------------ 输入拼接 ------

def input_dim(field_dim, mask_channel):
    """assemble 的输出宽度。构造模型时用它, 避免两处各算一遍。"""
    return 2 * field_dim + (1 if mask_channel else 0)


def assemble(prior, x_f, m, mask_channel=True):
    """拼输入。训练和推理都调它 —— 两边拼法必须一致。

    Args:
        prior: (B, N, F)   已归一化
        x_f:   (B, N, F)   自反馈槽位内容 (GT(t-1) 或 X̂(t-1)), 已归一化
        m:     (B, 1, 1) 或 (B, N, 1)   槽位有效性, 0/1
        mask_channel: 是否附加 mask 通道 (见下)

    两个 arm:
        arm 1  mask_channel=False -> [prior | x_f*m]      2F 通道
        arm 2  mask_channel=True  -> [prior | x_f*m | m]  2F+1 通道

    arm 1 已经是完整的 conditioning dropout —— 屏蔽由 `x_f * m` 完成, 是算术
    恒等 (w·0=0), 不需要学。模型也能自己发现"整帧 feedback 精确全零"这个
    签名 (真实数据里概率约等于零, HPM 有 LBO 全局混合, 够得着这个全局特征)。
    arm 2 的 mask 通道只是把这件事**直接告知**, 省下模型推断它的容量 ——
    是效率, 不是必要性。先跑 arm 1, 不够再开 arm 2。
    """
    parts = [prior, x_f * m]
    if mask_channel:
        if m.shape[1] == 1:
            m = m.expand(-1, prior.shape[1], -1)
        parts.append(m)
    return torch.cat(parts, dim=-1)


def sample_mask(has_prev, p_drop, generator=None):
    """训练用的 m: 真 t=0 恒为 0, 其余以 1-p_drop 的概率为 1。

    Args:
        has_prev: (B,) 或 (B,1,1)  1.0 = 该样本存在 GT(t-1)
        p_drop:   conditioning dropout 概率
    Returns:
        (B, 1, 1) float
    """
    h = has_prev.reshape(-1, 1, 1).float()
    if p_drop <= 0:
        return h
    keep = (torch.rand(h.shape, device=h.device, generator=generator)
            >= p_drop).float()
    return h * keep


def branch_norms(model, field_dim, mask_channel=True):
    """第一层权重按输入分支切开取范数 —— 诊断各支路有没有被用上。

    x = [coords(3) | prior(F) | x_f(F) | m(1)?],  故 preprocess[0].weight 的列:
        [0:3]           coords
        [3:3+F]         prior
        [3+F:3+2F]      feedback
        [3+2F]          mask (仅 arm 2)
    读法:
      ‖W_f‖ 比 ‖W_p‖ 小一两个数量级 -> feedback 支路是死重 (p 太高 / 无信息)
      ‖w_m‖ 与 ‖W_f‖ 量级可比       -> mask 被接收; 否则是死通道
    """
    W = model.preprocess[0].weight.detach()
    F = field_dim
    out = dict(
        coords=W[:, 0:3].norm().item(),
        prior=W[:, 3:3 + F].norm().item(),
        feedback=W[:, 3 + F:3 + 2 * F].norm().item(),
    )
    if mask_channel:
        out["mask"] = W[:, 3 + 2 * F].norm().item()
    return out


# ------------------------------------------------------------ Dataset ------

class PriorFeedbackDataset(Dataset):
    """样本 = (prior(t), gt(t-1), gt(t), has_prev)。

    与 PriorPairDataset 的唯一差别是多返回 gt(t-1) 和 has_prev。
    t=0 时 gt_prev 填 0 且 has_prev=0 —— 那一帧就是真实的 cold start。
    prior / gt 用同一套 stats 归一化 (同一物理空间, 必须同尺度)。

    Returns per sample:
        prior_t:  (N, F) float32  已归一化
        gt_prev:  (N, F) float32  已归一化; t=0 时全 0 (被 m=0 屏蔽, 值无所谓)
        gt_t:     (N, F) float32  已归一化
        has_prev: float           1.0 / 0.0
    """

    def __init__(self, data_dir, prior_dir, chunk_ids, schema, stats,
                 verbose=True):
        super().__init__()
        assert_prior_compatible(schema)
        from dataset import load_chunk           # 局部导入避免循环依赖

        self.schema = schema
        mean, std = stats[0], stats[1]

        self.priors, self.gts, self.samples = [], [], []
        for cid in chunk_ids:
            gt = load_chunk(data_dir, cid, schema)               # (T, N, F)
            pr = load_prior(prior_dir, cid, schema)              # (T, N, F)
            assert gt.shape == pr.shape, (
                f"chunk {cid}: GT {gt.shape} vs prior {pr.shape} 形状不符 —— "
                f"prior 是否用同一份 coords.npy 生成?")

            gt_n = ((gt - mean) / std).astype(np.float32)
            pr_n = ((pr - mean) / std).astype(np.float32)
            self.gts.append(torch.from_numpy(gt_n).clone().share_memory_())
            self.priors.append(torch.from_numpy(pr_n).clone().share_memory_())

            ci = len(self.gts) - 1
            for t in range(gt.shape[0]):
                self.samples.append((ci, t))
            if verbose:
                d = gt_n - pr_n
                rms = np.sqrt((d ** 2).mean(axis=(0, 1)))        # (F,)
                print(f"  chunk {cid}: T={gt.shape[0]}  "
                      f"Δ=0 基线 nRMSE per-ch = "
                      + " ".join(f"{n}={v:.3f}"
                                 for n, v in zip(schema.names, rms)))

        n_cold = sum(1 for _, t in self.samples if t == 0)
        if verbose:
            print(f"  样本 {len(self.samples)}, 其中真 cold start (t=0) {n_cold} "
                  f"({n_cold/max(len(self.samples),1)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, t = self.samples[idx]
        prior_t = self.priors[ci][t]
        gt_t = self.gts[ci][t]
        if t == 0:
            return prior_t, torch.zeros_like(prior_t), gt_t, 0.0
        return prior_t, self.gts[ci][t - 1], gt_t, 1.0