"""
dataset.py — 唯一数据层 (single data layer)。

合并了原先的 dataset.py / prior_ext.py / data_fw.py / data_seq.py。拆开时它们
互相 import 且出现循环依赖 (prior_ext 调 dataset.apply_alpha_weighting,
data_fw 调 prior_ext, data_seq 调两者), 靠"局部 import"打补丁; 合成一个文件后
那个循环消失。

Schema 驱动: 通道选择、alpha 加权、stats 命名全部来自 ChannelSchema (schema.py)。
磁盘文件恒为 6 通道, 按**名字**在加载时选列 —— 消融不需要重新生成数据。

分节
----
    1. 通道锚点          DISK_CHANNELS (schema.py) / PRIOR_CHANNELS
    2. chunk 区间工具    expand_range, chunk_tag
    3. GT 侧 I/O         load_chunk, apply_alpha_weighting
    4. stats 版本化      stats_filename, compute_stats, resolve_stats
    5. coords            load_coords
    6. prior 侧 I/O      assert_prior_compatible, load_prior, load_prior_valid
    7. 输入拼接 / 重建   input_dim, assemble, reconstruct, branch_norms
    8. 序列 dataset      WaveDataset, WindowSeqDataset, PriorSeqDataset

磁盘布局
--------
    data_dir/
        coords.npy                    # (N, 3) float32
        chunk_000_data.npy            # (T, N, 6) float32 — DISK_CHANNELS 顺序
        chunk_000_times.npy           # (T,) float64
        stats_{tag}_{signature}.npy   # (2, field_dim) — [mean, std]
        lbo/lbo_eigenvectors.npy      # (N, K) float32
    prior_dir/
        prior_000_data.npy            # (T, N, 5) float32 — PRIOR_CHANNELS 顺序
        prior_000_valid.npy           # (T, N) bool  (可选)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from schema import ChannelSchema, DISK_CHANNELS
# lift.CH_NAMES = prior 生成侧的通道序真相源; 仅供 §1 的护栏比对 (lift 只依赖 numpy)。
from lift import CH_NAMES as _LIFT_CH_NAMES


# ============================================================
# 1. 通道锚点
# ============================================================

# gen_prior.py 的输出通道顺序。无 nut —— FUNWAVE 是无粘 Boussinesq, 给不出
# 湍流粘度。Uy 有 prior 但诊断显示无效 (nRMSE>1, corr≈0), 见 config.yaml。
PRIOR_CHANNELS = ["alpha", "Ux", "Uy", "Uz", "p_rgh"]

# 护栏 (channel-order guard): prior 磁盘列序由 lift.CH_NAMES 决定 (gen_prior 照它
# stack + 落盘), 而 dataset 按 PRIOR_CHANNELS 的**名字**去磁盘取列。两份是同一事实
# 的两处定义, 必须逐字一致 —— 否则按名取列会错位, 且列数不变
# (assert pr.shape==gt.shape 抓不到), 会静默拿错位数据训练。这里让"两份漂了"变成
# import 时即崩, 并指出哪不一致。
# 注: 触发条件是"两份不相等", 不是"prior 是否含 Uy"; 改动任一份使其失配都会触发。
# 彻底消除重复的做法是令 PRIOR_CHANNELS = list(_LIFT_CH_NAMES) (单一真相源, 连
# assert 都不需要); 此处暂留双份 + 护栏, 改动最小、不改变现有行为。
assert list(_LIFT_CH_NAMES) == PRIOR_CHANNELS, (
    f"通道序漂移: lift.CH_NAMES={list(_LIFT_CH_NAMES)} != "
    f"PRIOR_CHANNELS={PRIOR_CHANNELS} —— prior 磁盘列序(由 lift 决定)与 dataset "
    f"读取假定已不一致, 必须同步二者。")


# ============================================================
# 2. chunk 区间工具
# ============================================================

def expand_range(r):
    """chunk 区间 -> 显式列表。
    [8]   -> [8]            单个
    [1,7] -> [1,2,...,7]    连续闭区间
    3 个及以上元素 -> 报错 (强制单个或区间, 不允许离散列表)。
    """
    if len(r) == 1:
        return [int(r[0])]
    elif len(r) == 2:
        return list(range(int(r[0]), int(r[1]) + 1))
    else:
        raise ValueError(
            f"chunk range must be [single] or [start,end], got {list(r)} "
            f"(discrete lists not allowed; use continuous interval)")


def chunk_tag(chunk_ids):
    """chunk 列表 -> 文件名用的紧凑标签。单个 -> 'c8'; 区间 -> 'c1-7'。"""
    ids = sorted(int(c) for c in chunk_ids)
    if len(ids) == 1:
        return f"c{ids[0]}"
    return f"c{ids[0]}-{ids[-1]}"


# ============================================================
# 3. GT 侧 I/O
# ============================================================

def load_chunk(data_dir, cid, schema):
    """载入一个 chunk, 按名选 schema 通道, 应用 alpha 加权。

    返回 (T, N, field_dim), 与模型训练所在的物理空间一致。
    """
    data_dir = Path(data_dir)
    data = np.load(data_dir / f"chunk_{cid:03d}_data.npy")   # (T, N, 6)
    assert data.shape[-1] == len(DISK_CHANNELS), (
        f"chunk_{cid:03d}_data.npy has {data.shape[-1]} channels, expected "
        f"{len(DISK_CHANNELS)} ({DISK_CHANNELS}) — disk layout mismatch")
    data = data[..., schema.disk_indices]                     # 按名选列
    return apply_alpha_weighting(data, schema)


def apply_alpha_weighting(data, schema):
    """把 alpha_weighted 通道在**物理空间**(归一化之前) 乘以 alpha。
    alpha 通道按名字定位 (schema.alpha_idx), 不用魔法索引。

    Args:
        data: (..., field_dim), schema 通道顺序, alpha ∈ [0,1]
    Returns:
        新数组 (不修改入参)
    """
    assert data.shape[-1] == schema.field_dim, (
        f"data has {data.shape[-1]} channels, schema expects "
        f"{schema.field_dim} ({schema.names})")
    out = data.copy()
    alpha = data[..., schema.alpha_idx:schema.alpha_idx + 1]  # (..., 1)
    for i, w in enumerate(schema.alpha_weighted):
        if w:
            out[..., i:i + 1] = data[..., i:i + 1] * alpha
    return out


# ============================================================
# 4. stats —— 按 chunk 集合 + 通道签名版本化
# ============================================================

def stats_filename(chunk_ids, schema):
    """stats_{tag}_{signature}.npy —— 编码训练 chunk 集合 + 通道集合 + alpha
    加权, 不同设置永远不会撞名。"""
    return f"stats_{chunk_tag(chunk_ids)}_{schema.signature()}.npy"


def _legacy_stats_filename(chunk_ids, schema):
    """旧命名 (schema 之前): stats_{tag}_u{0|1}_nut{0|1}.npy。
    只对那套 6 通道布局有意义。"""
    wu = int(schema.alpha_weighted[schema.names.index("Ux")])
    wn = int(schema.alpha_weighted[schema.names.index("nut")])
    return f"stats_{chunk_tag(chunk_ids)}_u{wu}_nut{wn}.npy"


def compute_stats(data_dir, chunk_ids, schema):
    """从指定 chunk (只能是训练集) 算逐通道 mean/std, 在 alpha 加权后、
    schema 选列后的场上算 —— 与喂给模型的东西一致。存成版本化文件名并返回。"""
    data_dir = Path(data_dir)
    running_sum = None
    running_sq_sum = None
    total_count = 0

    for cid in chunk_ids:
        data = load_chunk(data_dir, cid, schema)              # (T, N, C)
        T, N, C = data.shape
        flat = data.reshape(-1, C).astype(np.float64)
        if running_sum is None:
            running_sum = np.zeros(C, dtype=np.float64)
            running_sq_sum = np.zeros(C, dtype=np.float64)
        running_sum += flat.sum(axis=0)
        running_sq_sum += (flat ** 2).sum(axis=0)
        total_count += T * N

    mean = running_sum / total_count
    std = np.sqrt(running_sq_sum / total_count - mean ** 2)
    std = np.maximum(std, 1e-8)

    stats = np.stack([mean, std], axis=0).astype(np.float32)  # (2, C)
    fname = stats_filename(chunk_ids, schema)
    np.save(data_dir / fname, stats)
    print(f"Saved {fname}: mean={mean}, std={std}")
    return stats


def resolve_stats(data_dir, chunk_ids, schema, verbose=True):
    """有就读, 没有就算。解析顺序:
      1. 新版本化名  stats_{tag}_{signature}.npy
      2. 旧名        stats_{tag}_u{0|1}_nut{0|1}.npy  (只读回退, 数值等价)
      3. 从 chunk 现算
    总是对照 schema.field_dim 校验形状 (抓过期文件)。"""
    data_dir = Path(data_dir)

    new_path = data_dir / stats_filename(chunk_ids, schema)
    if new_path.exists():
        stats = np.load(new_path)
        if verbose:
            print(f"Loaded {new_path.name}: mean={stats[0]}, std={stats[1]}")
    elif schema.is_legacy_layout() and \
            (data_dir / _legacy_stats_filename(chunk_ids, schema)).exists():
        legacy_path = data_dir / _legacy_stats_filename(chunk_ids, schema)
        stats = np.load(legacy_path)
        if verbose:
            print(f"Loaded legacy stats {legacy_path.name} "
                  f"(identical computation, old naming)")
    else:
        if verbose:
            print(f"Computing dataset statistics -> {new_path.name}")
        stats = compute_stats(data_dir, chunk_ids, schema)

    assert stats.shape == (2, schema.field_dim), (
        f"stats shape {stats.shape} != (2, {schema.field_dim}) — stale stats "
        f"file for a different channel set? Delete and recompute.")
    return stats


# ============================================================
# 5. coords
# ============================================================

def load_coords(data_dir):
    """载入并归一化坐标。调用一次, 在循环外送上 GPU。"""
    data_dir = Path(data_dir)
    coords = np.load(data_dir / "coords.npy").astype(np.float32)
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    coord_range = np.maximum(coord_max - coord_min, 1e-8)
    coords_norm = (coords - coord_min) / coord_range
    return torch.from_numpy(coords_norm)  # (N, 3)


# ============================================================
# 6. prior 侧 I/O
# ============================================================

def prior_indices(schema):
    """schema 通道 -> prior 磁盘布局的列索引。"""
    return [PRIOR_CHANNELS.index(n) for n in schema.names]


def assert_prior_compatible(schema):
    """schema 的每个通道都必须能从 prior 拿到。

    nut 无 prior (FUNWAVE 是无粘 Boussinesq); Uy 的 prior 经诊断无效
    (十个 chunk nRMSE 1.000-1.079, corr≈0) —— 两者在 fwv 线都应 disabled。
    """
    missing = [n for n in schema.names if n not in PRIOR_CHANNELS]
    assert not missing, (
        f"通道 {missing} 在 prior 中不存在 (prior 只有 {PRIOR_CHANNELS})。\n"
        f"fwv 线所有通道都以 prior 为 base, 故这些通道必须 enabled: false。\n"
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


# ============================================================
# 7. 输入拼接 / 重建
#    训练与推理**共用**这几个函数 —— 两边拼法必须逐字节一致, 分开写迟早漂。
# ============================================================

def input_dim(field_dim):
    """自反馈臂的模型输入宽度 = [prior | x_f*m] = 2F。

    构造模型时用它, 避免"2*F"这个魔数散落在训练与推理两处。
    """
    return 2 * field_dim


def assemble(prior, x_f, m):
    """拼输入: [prior | x_f * m], 2F 通道。

    Args:
        prior: (B, N, F)              已归一化
        x_f:   (B, N, F)              自反馈槽位内容 (GT 或 X̂(t-1)), 已归一化
        m:     (B, 1, 1) 或 (B, N, 1) 槽位有效性, 0/1

    屏蔽由 `x_f * m` 完成 —— 算术恒等 (w·0=0), 不需要学。模型也能自己发现
    "整帧 feedback 精确全零"这个签名 (真实数据里概率约等于零, HPM 有 LBO
    全局混合, 够得着这个全局特征)。

    m 是 per-frame 的 (B,1,1) 而非 per-point: 上一帧要么算过要么没算过,
    不存在一帧里有些点有、有些点没有。
    """
    return torch.cat([prior, x_f * m], dim=-1)


def reconstruct(base, delta, delta_idx):
    """pred = base + Δ, Δ 只加在 delta_idx 通道 (frozen 通道保持 = base)。

    base: (B, N, F);  delta: (B, N, out_dim);  返回 (B, N, F)。
    out-of-place index_add: 对 delta 可导, base 无需 grad。
    训练与推理共用 —— 与 assemble 同一个理由。
    """
    return base.index_add(-1, delta_idx, delta)


def branch_norms(model, field_dim):
    """第一层权重按输入分支切开取范数 —— 诊断各支路有没有被用上。

    x = [coords(3) | prior(F) | x_f(F)], 故 preprocess[0].weight 的列:
        [0:3]        coords
        [3:3+F]      prior
        [3+F:3+2F]   feedback
    读法: ‖W_f‖ 比 ‖W_p‖ 小一两个数量级 -> feedback 支路是死重 (无信息)。
    """
    W = model.preprocess[0].weight.detach()
    F = field_dim
    return dict(
        coords=W[:, 0:3].norm().item(),
        prior=W[:, 3:3 + F].norm().item(),
        feedback=W[:, 3 + F:3 + 2 * F].norm().item(),
    )


# ============================================================
# 8. 序列 dataset
#    两条线都产出 **dict batch** —— 训练循环按键取用, 不按形状分派。
# ============================================================

class WaveDataset(Dataset):
    """纯 HPM 线的基础 dataset: (window_fields, future_frames) 元组。

    每个样本:
        window_fields: (N, W*F) float32 — W 帧连续窗口, 压平
        future_frames: (R, N, F) float32 — R 帧未来 GT (归一化, 绝对值)

    rollout / delta 计算 / 逐步累积由训练循环处理 (见 train.py)。
    coords **不**逐样本返回 —— 用 load_coords() 单独取。
    数据存成 torch.Tensor 以便 DataLoader worker 共享内存。
    """

    def __init__(self, data_dir, chunk_ids, window, schema, stats=None,
                 rollout_steps=4):
        super().__init__()
        assert isinstance(schema, ChannelSchema), \
            "WaveDataset requires a ChannelSchema (see schema.py)"
        self.data_dir = Path(data_dir)
        self.window = window
        self.schema = schema
        self.rollout_steps = rollout_steps
        self.N = np.load(self.data_dir / "coords.npy").shape[0]

        if stats is None:
            stats = resolve_stats(self.data_dir, chunk_ids, schema)
        self.stats = stats
        self.mean = self.stats[0]
        self.std = self.stats[1]

        self.chunks = []
        self.samples = []
        self._build_samples(chunk_ids)

    def _build_samples(self, chunk_ids):
        for cid in chunk_ids:
            data = load_chunk(self.data_dir, cid, self.schema)  # (T, N, F)
            data_norm = ((data - self.mean) / self.std).astype(np.float32)
            # .clone() 脱离 numpy 内存, .share_memory_() 让多 worker 安全
            # (避免 fork 后的 copy-on-write)
            tensor = torch.from_numpy(data_norm).clone().share_memory_()
            self.chunks.append(tensor)

            T = data.shape[0]
            # 输入需要 W 帧 + 目标需要 R 帧
            for t in range(self.window, T - self.rollout_steps + 1):
                self.samples.append((len(self.chunks) - 1, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_idx, t = self.samples[idx]
        chunk = self.chunks[chunk_idx]
        window_frames = chunk[t - self.window:t]        # (W, N, F)
        future = chunk[t:t + self.rollout_steps]        # (R, N, F) 绝对值
        window_flat = window_frames.permute(1, 0, 2).reshape(self.N, -1)
        return window_flat, future


class WindowSeqDataset(WaveDataset):
    """纯 HPM 线: {"window": (N, W*F), "gt_seq": (R, N, F)}。

    只改返回形状为 dict —— 采样逻辑、归一化、share_memory_ 全部沿用父类。
    """

    def __getitem__(self, idx):
        window_flat, future = super().__getitem__(idx)
        return {"window": window_flat, "gt_seq": future}


class PriorSeqDataset(Dataset):
    """fwv 线: {"prior_seq": (R, N, F), "gt_seq": (R, N, F)}, 已归一化。

    prior_seq[r] = prior(t+r);  gt_seq[r] = gt(t+r),  r = 0..R-1。
    合法起点 t ∈ [0, T-R] 在 __init__ 枚举, 打乱交给 DataLoader shuffle。

    设计决定 (记在这里免得漂):
      - 起点随机跳, 起点定了之后那 R 帧严格顺序推。序列**内**顺序是 BPTT 的
        要求, 序列**间**随机是泛化的要求 —— 两者不矛盾。
      - **不返回 gt(t-1)**。每个序列的 r=0 恒冷启动 (m=0, 不喂任何 feedback),
        所以起点之前的帧与本样本无关。r>=1 的 feedback 全在 [t, t+R-1] 内自足。
        起点前一帧"拿不到"其实是"故意不用" —— 每个序列自带一次冷启动, 分布贴
        部署条件。
      - prior 与 gt 用**同一套 stats** 归一化: 它们在同一物理空间, 必须同尺度,
        否则 `base + Δ` 加的不是同一量纲的东西。
      - R=1 时退化为单帧样本 (无反馈基线用的就是这个)。
      - 最后 R-1 帧不能当起点 (越界), 但它们仍作为别的序列的中间帧被训练到。
    """

    def __init__(self, data_dir, prior_dir, chunk_ids, schema, stats, R,
                 verbose=True):
        super().__init__()
        assert R >= 1, f"R 必须 >=1, 当前 {R}"
        assert_prior_compatible(schema)

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
        return {"prior_seq": self.priors[ci][t:t + R],   # (R, N, F)
                "gt_seq": self.gts[ci][t:t + R]}         # (R, N, F)