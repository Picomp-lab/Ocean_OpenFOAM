"""
Channel schema —— 通道配置的唯一真相源 (single source of truth)。

config.yaml 声明 `data.channels` (list of {name, enabled, update_rule,
loss_weight, alpha_weighted}); 其余全部在这里**派生** —— field_dim, out_dim,
delta 索引, alpha 索引, 磁盘列选择, stats 文件签名。任何其他文件都不许硬编码
通道数或通道索引。

本模块同时拥有 `advance_window()`: 滑窗前进的实现, 由**推理 (vis.py)** 调用。
注: 训练侧 (train.py SelfStatePolicy) 走 dataset.reconstruct + 内联移位, **不经**
此函数 —— 两者当前数值等价, 但不是同一实现, 改任一处需手动对齐另一处
(advance_window 因需在 pred/移位之间插 SS 的 _pick 而无法被训练直接复用)。
fwv 线不走滑窗 (状态是单帧反馈槽), 不调这个函数。

update_rule 语义:
    delta    — 由 delta head 预测 (残差学习)
    frozen   — 只作输入; rollout 中原样携带, 不进 loss (nut 消融的 Method B)
    flux_div — 预留给 E2 flux head (alpha 由面通量散度更新)。作为合法枚举值
               校验, 但在 flux head 落地前直接抛 NotImplementedError。
"""

from dataclasses import dataclass

import torch

# chunk_*_data.npy 在磁盘上的固定列序。数据文件**从不**为消融重新生成;
# 通道在载入时按名字对着这个锚点选列。
DISK_CHANNELS = ["alpha", "Ux", "Uy", "Uz", "p_rgh", "nut"]

UPDATE_RULES = ("delta", "frozen", "flux_div")


@dataclass(frozen=True)
class ChannelSchema:
    names: tuple            # 通道名, 顺序 == 张量通道顺序
    update_rules: tuple     # 逐通道 update rule (见模块 docstring)
    loss_weights: tuple     # 逐通道 loss 权重 (非 delta 通道忽略)
    alpha_weighted: tuple   # 逐通道 bool: 物理空间内乘 alpha

    # ---------------- 派生量 (从不声明, 总是算出来) ----------------

    @property
    def field_dim(self):
        return len(self.names)

    @property
    def delta_indices(self):
        """由 delta head 预测的通道索引。"""
        return [i for i, r in enumerate(self.update_rules) if r == "delta"]

    @property
    def out_dim(self):
        """模型头宽度 = delta 通道数。"""
        return len(self.delta_indices)

    @property
    def alpha_idx(self):
        """alpha 通道的索引 —— 按名字查, 不用魔法数字。"""
        return self.names.index("alpha")

    @property
    def disk_indices(self):
        """到磁盘 6 通道布局的列索引。"""
        return [DISK_CHANNELS.index(n) for n in self.names]

    def delta_loss_weights(self):
        """限制到 delta 通道的 loss 权重 (loss 只覆盖这些通道)。"""
        return [self.loss_weights[i] for i in self.delta_indices]

    def display_names(self):
        """可读名; alpha 加权的通道加 'α' 前缀。"""
        return [("α" + n) if w else n
                for n, w in zip(self.names, self.alpha_weighted)]

    def signature(self):
        """stats 文件名用的通道签名。alpha 加权的通道带 'w' 后缀, 例如
            alpha.Uxw.Uzw.p_rgh
        保证不同通道集 / 不同加权方式在磁盘上永不撞名 —— 这也是为什么 αU 与
        非 αU 的 nRMSE 不在同一个空间 (归一化的量本身换了), 不能直接比大小。"""
        parts = [(n + "w") if w else n
                 for n, w in zip(self.names, self.alpha_weighted)]
        return ".".join(parts)

    def is_legacy_layout(self):
        """True 当且仅当这个 schema 正好是历史上的 6 通道全 delta 布局
        (仅用于回落到旧的 stats_*_u{0|1}_nut{0|1}.npy)。"""
        return (list(self.names) == DISK_CHANNELS
                and all(r == "delta" for r in self.update_rules))

    def describe(self):
        lines = ["ChannelSchema:"]
        for i, (n, r, lw, aw) in enumerate(zip(
                self.names, self.update_rules, self.loss_weights,
                self.alpha_weighted)):
            lines.append(f"  [{i}] {n:8s} rule={r:8s} loss_w={lw:<4g} "
                         f"alpha_weighted={aw}")
        lines.append(f"  field_dim={self.field_dim}  out_dim={self.out_dim}  "
                     f"delta_indices={self.delta_indices}  "
                     f"alpha_idx={self.alpha_idx}")
        return "\n".join(lines)

    # ---------------- 构造 + 校验 ----------------

    @classmethod
    def from_cfg(cls, cfg, verbose=True):
        """从 Hydra/OmegaConf 配置构造并校验。

        新配置:   cfg.data.channels 列表 (唯一真相源)。
        旧配置:   没有 'channels' 键 (老 checkpoint 的 .hydra/config.yaml)
                  -> 回落到历史 6 通道全 delta, 尊重 weight_u_by_alpha /
                  weight_nut_by_alpha。vis.py 载入老 checkpoint 时会走到。

        verbose=False 静音 info 打印 (run name 解析期间可能被调多次)。
        校验断言**从不**静音。
        """
        data_cfg = cfg.data
        if "channels" not in data_cfg or data_cfg.channels is None:
            return cls._legacy_default(data_cfg, verbose=verbose)

        names, rules, weights, aw = [], [], [], []
        all_names, disabled = [], []
        for ch in data_cfg.channels:
            name = str(ch["name"])
            all_names.append(name)
            if not bool(ch.get("enabled", True)):
                # enabled: false == 这一行不存在: 非输入、无 head、无 stats、
                # 不计入 field_dim。
                disabled.append(name)
                continue
            names.append(name)
            rules.append(str(ch.get("update_rule", "delta")))
            weights.append(float(ch.get("loss_weight", 1.0)))
            aw.append(bool(ch.get("alpha_weighted", False)))

        # ---- 显式的域级校验 (单层, 不用 ConfigStore) ----
        # 校验**所有**名字, 含被 disable 的 —— disabled 行里的拼写错误也必须
        # 大声失败, 而不是静默地 "禁用" 一个不存在的通道。
        unknown = [n for n in all_names if n not in DISK_CHANNELS]
        assert not unknown, (
            f"channel names not in DISK_CHANNELS {DISK_CHANNELS}: {unknown}")
        assert len(set(all_names)) == len(all_names), \
            f"duplicate channel names: {all_names}"
        assert "alpha" not in disabled, \
            "channel 'alpha' cannot be disabled — it is the VOF core field"
        if disabled and verbose:
            print(f"[schema] disabled channels (excluded entirely): {disabled}")
        assert len(names) > 0, "data.channels has no enabled channels"
        bad = [r for r in rules if r not in UPDATE_RULES]
        assert not bad, f"invalid update_rule {bad}; allowed: {UPDATE_RULES}"
        assert "alpha" in names, \
            "channel 'alpha' (alpha.water) must be present — it is the VOF core field"
        if "flux_div" in rules:
            raise NotImplementedError(
                "update_rule 'flux_div' is reserved for the E2 flux head and "
                "not implemented yet")
        for n, r, lw in zip(names, rules, weights):
            if r != "delta" and lw != 0.0 and verbose:
                print(f"[schema] WARNING: channel '{n}' has rule '{r}' but "
                      f"loss_weight={lw}; non-delta channels receive no loss "
                      f"— weight ignored.")

        schema = cls(tuple(names), tuple(rules), tuple(weights), tuple(aw))
        assert schema.out_dim > 0, "no 'delta' channels — nothing to predict"
        return schema

    @classmethod
    def _legacy_default(cls, data_cfg, verbose=True):
        wu = bool(data_cfg.get("weight_u_by_alpha", True))
        wn = bool(data_cfg.get("weight_nut_by_alpha", False))
        if verbose:
            print(f"[schema] no data.channels in config — legacy 6-channel "
                  f"fallback (weight_u={wu}, weight_nut={wn})")
        return cls(
            names=tuple(DISK_CHANNELS),
            update_rules=("delta",) * len(DISK_CHANNELS),
            loss_weights=(1.0, 1.0, 1.0, 1.0, 0.1, 0.1),
            alpha_weighted=(False, wu, wu, wu, False, wn),
        )


# ====================================================================
# rollout 闭包 —— 供**推理 (vis.py)** 做滑窗移位。
# 注: 训练 (train.py SelfStatePolicy.advance) 走 reconstruct + 内联 torch.cat,
# **不经**此函数; 两者数值等价但非同一实现, 改任一处需手动对齐 (见 train.py)。
# 将来的 flux head: 若沿此路径, flux_div 分派加在**这里**, 别处不加。
# ====================================================================

def advance_window(window, delta_partial, schema):
    """把时间窗口前进一个预测步。

    Args:
        window:        (B, N, W*F) — normalized 场窗口, 展平
        delta_partial: (B, N, out_dim) — 模型输出, 只有 DELTA 通道

    Returns:
        pred_frame: (B, N, F) — 下一帧。delta 通道由预测更新; frozen 通道
                    原样携带 (delta=0)。
        new_window: (B, N, W*F) — 窗口移位一格, 追加 pred。

    梯度经 delta_partial 回传 (index_copy 可微), 全 BPTT 行为与合并前的
    内联实现一致。
    """
    F = schema.field_dim
    current = window[..., -F:]                                  # (B, N, F)

    if schema.out_dim == F:
        delta_full = delta_partial                              # 快路径
    else:
        # 把部分 delta 散射回完整通道布局。
        idx = torch.as_tensor(schema.delta_indices,
                              device=delta_partial.device)
        delta_full = delta_partial.new_zeros(
            *delta_partial.shape[:-1], F).index_copy(-1, idx, delta_partial)

    pred_frame = current + delta_full                           # (B, N, F)
    new_window = torch.cat([window[..., F:], pred_frame], dim=-1)
    return pred_frame, new_window


# ====================================================================
# 纯 HPM 线的自动命名 —— 运行名本身也是派生量。
#
# 名字编码**相对 6 通道全 delta 基线的 diff**:
#   hpm_bl_h128            无 diff
#   hpm_no-nut_h128        nut enabled: false
#   hpm_frz-nut_h128       nut update_rule: frozen
#   ..._aU_...             存在任何 alpha_weighted 通道
#
# 只服务纯 HPM 线。fwv 线的命名 (hpm_fw[_nofb][_aU|_aUx|_aUz]_h{n}) 在
# train.py run_name 里, 由 data.window 分派 —— 两条线各自的命名空间。
# 接线: config.yaml 的 wandb.name: ${runname:} -> train.py run_name -> 这里。
# hydra 输出目录跟随 wandb.name, 所以整条链自动。
# CLI 覆盖照常: python train.py wandb.name=custom
# ====================================================================

def auto_run_name(cfg, prefix="hpm"):
    """从通道 schema + 关键超参派生运行名 (纯 HPM 线)。"""
    schema = ChannelSchema.from_cfg(cfg, verbose=False)
    parts = []
    for n in DISK_CHANNELS:
        if n not in schema.names:
            parts.append(f"no-{n}")
        else:
            rule = schema.update_rules[schema.names.index(n)]
            if rule == "frozen":
                parts.append(f"frz-{n}")
            elif rule == "flux_div":
                parts.append(f"flux-{n}")
    if any(schema.alpha_weighted):
        parts.append("aU")
    if not parts:
        parts = ["bl"]

    name = "_".join([prefix, *parts, f"h{cfg.model.n_hidden}"])
    suffix = str(cfg.wandb.get("name_suffix", "") or "")
    if suffix:
        name += f"_{suffix}"
    return name