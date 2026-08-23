"""
data_fw_ss.py — scheduled sampling + 真 BPTT 的数据层。

相对 data_fw.py (单步 teacher forcing) 的唯一变化: 样本从"单帧"变成
"从随机起点起的连续 R 帧序列"。

    单步 (data_fw)   样本 = (prior(t), gt(t-1), gt(t))            1 帧
    R 步 (本文件)     样本 = (prior[t:t+R], gt[t:t+R])            R 帧连续

关键设计 (前几轮定的, 记在这里免得漂):
  - 起点 t 在 chunk 内**随机**抽, t ∈ [0, T-R]; DataLoader shuffle 照旧。
    "起点随机跳, 起点定了之后那 R 帧顺序推" —— 序列内部严格顺序 (BPTT 要求),
    序列之间随机 (泛化)。这两件事不矛盾。
  - **不返回 gt(t-1)**。序列第一步 (r=0) 恒 m=0 冷启动, 不喂任何 feedback,
    所以 t 之前的帧与本样本无关。r≥1 的 feedback 全在 [t, t+R-1] 内自足
    (要么模型自己上一步的 pred, 要么切片内的 gt)。这就是为什么起点前一帧
    "拿不到"其实是"故意不用" —— 每个序列自带一次冷启动, 分布贴部署。
  - prior / gt 用同一套 stats 归一化 (同一物理空间, 必须同尺度)。
  - 最后 R-1 帧不能当起点 (会越界), 但它们仍作为别的序列的中间帧被训练到。

拼接 (assemble) 不在这里 —— 复用 data_fw.assemble, 训练/推理逐字节一致。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from prior_ext import assert_prior_compatible, load_prior     # fwv/


class PriorSeqDataset(Dataset):
    """样本 = (prior_seq, gt_seq), 各 (R, N, F), 已归一化。

    prior_seq[r] = prior(t+r);  gt_seq[r] = gt(t+r),  r = 0..R-1。
    起点 t ∈ [0, T-R] 随机 (合法起点在 __init__ 枚举, shuffle 交给 DataLoader)。
    """

    def __init__(self, data_dir, prior_dir, chunk_ids, schema, stats, R,
                 verbose=True):
        super().__init__()
        assert R >= 1, f"R 必须 >=1, 当前 {R}"
        assert_prior_compatible(schema)
        from dataset import load_chunk           # 父 hpm/; 局部导入避免循环依赖

        self.schema = schema
        self.R = R
        mean, std = stats[0], stats[1]

        self.priors, self.gts, self.samples = [], [], []
        for cid in chunk_ids:
            gt = load_chunk(data_dir, cid, schema)               # (T, N, F)
            pr = load_prior(prior_dir, cid, schema)              # (T, N, F)
            assert gt.shape == pr.shape, (
                f"chunk {cid}: GT {gt.shape} vs prior {pr.shape} 形状不符 —— "
                f"prior 是否用同一份 coords.npy 生成?")
            T = gt.shape[0]
            if T < R:
                if verbose:
                    print(f"  chunk {cid}: T={T} < R={R}, 跳过")
                continue

            gt_n = ((gt - mean) / std).astype(np.float32)
            pr_n = ((pr - mean) / std).astype(np.float32)
            self.gts.append(torch.from_numpy(gt_n).clone().share_memory_())
            self.priors.append(torch.from_numpy(pr_n).clone().share_memory_())

            ci = len(self.gts) - 1
            for t in range(0, T - R + 1):        # 合法起点 [0, T-R]
                self.samples.append((ci, t))
            if verbose:
                d = gt_n - pr_n
                rms = np.sqrt((d ** 2).mean(axis=(0, 1)))        # (F,)
                print(f"  chunk {cid}: T={T}  起点数={T - R + 1}  "
                      f"Δ=0 基线 nRMSE per-ch = "
                      + " ".join(f"{n}={v:.3f}"
                                 for n, v in zip(schema.names, rms)))

        if verbose:
            print(f"  序列样本总数 {len(self.samples)}  (R={R})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, t = self.samples[idx]
        R = self.R
        prior_seq = self.priors[ci][t:t + R]     # (R, N, F)
        gt_seq = self.gts[ci][t:t + R]           # (R, N, F)
        return prior_seq, gt_seq
