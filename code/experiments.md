# 实验记录 (experiment log)

两条模型线，唯一的结构性差异是 **base（残差加在谁身上）**：

| | base | state | R | p | 状态 |
|---|---|---|---|---|---|
| **纯 HPM** | 自己上一帧 | W=6 滑窗 | 4 | 恒 0（`rollout.ss=false`） | 前一阶段，不再迭代 |
| **HPM + FUNWAVE** | prior(t) | 单帧反馈槽 | 4 | 1.0→0.1 | 活跃 |

BPTT 两条线都有，不构成区分。cold start 与 scheduled sampling 是"有反馈槽"的
衍生物 —— 纯 HPM 没有槽位，这两个问题都不存在。

复现命令见 `config.yaml` 头部的复现表。

## fw 线的臂

| 臂 | 配置 | 结论 |
|---|---|---|
| 无反馈基线 | `feedback=none, R=1` | prior 场 → CFD 场的单步映射能力检验，判据是优于 Δ=0（即 prior 本身） |
| 自反馈 + SS | 默认 | 治 exposure bias |
| **+ αU 加权** | `Ux/Uz alpha_weighted=true` | **当前最佳臂**，判据是可视化的近岸形态 |

### αU 的比较陷阱

αU 与非 αU 的 nRMSE 不在同一个空间，不能直接比大小：

```
αU     stats_c1-7_alpha.Uxw.Uzw.p_rgh.npy   Ux 的 Δ=0 基线 0.619
非 αU  stats_c1-7_alpha.Ux.Uz.p_rgh.npy     Ux 的 Δ=0 基线 0.896
```

归一化的量本身换了（αUx vs Ux），分母不同。跨臂比较只能用共同空间的指标 ——
`vis.py pred` 输出的 raw 空间 slice-RMSE，或分区域的近岸形态。

### 随机性

当前未固定随机种子。同配置重跑，`best_val` 有约 2% 抖动（实测 0.1549 vs
0.1512），且刷新出现的 epoch 会大幅移动（epoch 10 vs epoch 22）。两次的收敛
终点几乎重合（末 15 个 epoch 都在 0.163–0.170）。要让"重跑一遍"成为可比的
证据，需要先固定 seed。

## 误差分解

fw 线自回归 rollout 的误差约 0.125，其中：

- **≈0.10 的地板** = FUNWAVE prior 在破碎区失效。Boussinesq 理论在翻卷区本身
  就崩溃，这一段不是模型能解决的
- **≈0.04 的累积** = exposure bias，SS 针对的是且只是这一段

## 已排除的方向

| 方向 | 排除理由 |
|---|---|
| Clifford / 几何代数 | 域无旋转对称（重力各向异性 + 斜底 + 有界域）；`mlp_trans_weights` 训练后无可学习结构 |
| 逐点 PDE 时间导数残差 | Courant C≈1.6（近空气区），残差形式不成立 |
| 可微分求解器 | MULES 不可微 |
| 全局不变量约束 | 开放耗散域，不守恒 |
| 对称性 / 等变性约束 | 域各向异性 |
| mask 通道（2F+1） | 从未训练。屏蔽本已由 `x_f·m` 完成（算术恒等），mask 列只是把"这次是屏蔽"显式告知，是效率不是必要性 |
