"""
Prior extension — FUNWAVE lift prior 的载入与单步 dataset。

隔离原则: 本模块只 *读* schema.py / dataset.py 的公开接口, 不修改它们。
现有 E0/E1 训练路径 (train.py + WaveDataset) 完全不受影响。

1b 模式 (capability check):
    输入 = prior(t)          无时间窗、无自反馈
    base = prior(t)
    预测 = prior(t) + Δ
    R = 1                    没有反馈回路, R>1 等价于扩大 batch, 不提供额外信息

结果: 无 rollout 漂移、无 cold start (t=0 即可预测)。
这是最纯粹的 "lift 场 -> CFD 场" 映射测试, 对应 progress report 第 5 节。

判据不是 "优于现行 baseline" (通道数与时间建模都变了, 不可比), 而是
优于 Δ=0 —— 即 prior 本身。diag_prior.py 已给出该基线 (raw 空间 nRMSE):
    alpha 0.42   Ux 0.92   Uz 0.96   p_rgh 0.69
train_prior.py 每个 epoch 都会打印这个基线, 直接对照。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import apply_alpha_weighting

# gen_prior.py 的输出通道顺序 (无 nut —— FUNWAVE 给不出湍流粘度)
PRIOR_CHANNELS = ["alpha", "Ux", "Uy", "Uz", "p_rgh"]


def prior_indices(schema):
    """schema 通道 -> prior 磁盘布局的列索引。"""
    return [PRIOR_CHANNELS.index(n) for n in schema.names]


def assert_prior_compatible(schema):
    """schema 的每个通道都必须能从 prior 拿到。

    nut 无 prior (FUNWAVE 是无粘 Boussinesq), Uy 的 prior 经诊断无效
    (十个 chunk nRMSE 1.000-1.079, corr ~= 0) —— 两者在 1b 下都应 disabled。
    """
    missing = [n for n in schema.names if n not in PRIOR_CHANNELS]
    assert not missing, (
        f"通道 {missing} 在 prior 中不存在 (prior 只有 {PRIOR_CHANNELS})。\n"
        f"1b 模式下所有通道都以 prior 为 base, 故这些通道必须 enabled: false。\n"
        f"  nut  —— FUNWAVE 无湍流模型, 结构上给不出\n"
        f"  Uy   —— 有 prior 但诊断显示无效 (nRMSE>1), 留着只会加噪")


def load_prior(prior_dir, cid, schema):
    """载入一个 chunk 的 prior, 按名选列 + alpha 加权。

    与 GT 走同一个 apply_alpha_weighting, 保证两者在同一物理空间。
    prior 的 alpha 是 sharp 0/1, 加权对它近乎幂等, 但保持代码路径一致 ——
    将来若换 fractional alpha 无需改动此处。

    Returns: (T, N, field_dim)
    """
    prior_dir = Path(prior_dir)
    data = np.load(prior_dir / f"prior_{cid:03d}_data.npy")     # (T, N, 5)
    assert data.shape[-1] == len(PRIOR_CHANNELS), (
        f"prior_{cid:03d}_data.npy 有 {data.shape[-1]} 通道, "
        f"期望 {len(PRIOR_CHANNELS)} ({PRIOR_CHANNELS})")
    data = data[..., prior_indices(schema)]
    return apply_alpha_weighting(data, schema)


def load_prior_valid(prior_dir, cid):
    """载入 valid 掩码 (T, N) bool; 不存在则返回 None。

    invalid 处 prior 被填 0 (干单元/梯度扩散/域外)。诊断显示这类 cell 占
    0.3-11%, 且 GT 在那里 alpha 均值 0.0001 —— CFD 也是空气, 填 0 与 GT 一致,
    故默认不排除。保留接口以备需要。
    """
    p = Path(prior_dir) / f"prior_{cid:03d}_valid.npy"
    return np.load(p) if p.exists() else None


class PriorPairDataset(Dataset):
    """1b: 样本 = (prior(t), gt(t)) 同帧配对, 无时间窗、无 rollout。

    两者用同一套 stats 归一化 —— 它们在同一物理空间, 必须同尺度, 否则
    `prior_norm + Δ_norm` 加的不是同一量纲的东西。

    Returns per sample:
        prior_t: (N, F) float32  已归一化
        gt_t:    (N, F) float32  已归一化
    """

    def __init__(self, data_dir, prior_dir, chunk_ids, schema, stats,
                 verbose=True):
        super().__init__()
        assert_prior_compatible(schema)
        from dataset import load_chunk           # 局部导入避免循环依赖

        self.schema = schema
        mean, std = stats[0], stats[1]

        self.priors, self.gts = [], []
        self.samples = []
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, t = self.samples[idx]
        return self.priors[ci][t], self.gts[ci][t]