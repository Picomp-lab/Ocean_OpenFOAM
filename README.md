# 波浪场代理模型（Surrogate Model）探索说明书

> 覆盖 `legacy/` 里的四条历史线（`pod-lstm/` `transolver++/` `fno/` `tsolverpp/` 与`hpm/`）、`code/`（HPM 两条线，**当前主力**），以及 `data/`（数据资产）与 `results/`（产物）。
> 目的：记录**做过什么、结果如何、怎么复现、东西放在哪**。
>
> **三份文档的分工**：本文＝项目本身（数据 / 六条线 / 结果）；
> [SETUP.md](SETUP.md)＝环境与集群操作；
> [code/web-demo/README.md](code/web-demo/README.md)＝web 交互演示。
>
> `legacy/hpm/` 是 `code/` 的上一代实现（同一条 HPM 线，重构前），本文档以 `code/` 为准。

---

## 环境

clone 之后先跑 `./setup.sh` —— 它会建 conda 环境、装依赖、检查数据齐不齐。
完整说明（setup.sh 的开关、依赖为什么分两份、activate.sh 怎么定位环境、SLURM 速查）
见 **[SETUP.md](SETUP.md)**。

web 交互演示单独一份：**[code/web-demo/README.md](code/web-demo/README.md)**。

---

## 目录布局

```
code/        当前主力线（HPM 两条），唯一在改的代码
legacy/      不再维护的历史线：hpm/ fno/ transolver++/ tsolverpp/ pod-lstm/ env_info/
results/     产物。只有 web/model/ 那份 demo 权重随仓库走，其余在 results_*.tar 里
data/        数据资产 48.6 G（3d 34.5 G + fwv 8.5 G + 2d 5.6 G）—— **不随仓库分发**，
             在 data_*.tar 里；库里只留 crop_fields.* / make_cases.py / input.txt 这类
             「怎么造出来的」的记录
archive/     restore.sh + 各包的 .manifest（进版本库）；从云端硬盘下下来的 .tar
             也放这里（不进版本库）。清单和还原方式见 archives.tsv
```

---

## 0. 快速导航

| # | 方向 | 目录 | 时间 | 维度 | 状态 | 短程精度 | 长期自回归 |
|---|---|---|---|---|---|---|---|
| 1 | POD + LSTM | `legacy/pod-lstm/` | 05-03 | 2D | 完成，判失败 | 系数空间 10–25% | 场空间 51–164%，失败 |
| 2 | Transolver++ | `legacy/transolver++/` | 05-11 | 2D 点云 149,758 | 完成（早停 ep180） | alpha 3.5% / Ux 16.3% / Uz 26.4% | 100 步 111%，发散 |
| 3 | FNO | `legacy/fno/` | 05-12 | 2D 网格 4096×128 | 撞时限 | alpha 2.3% / Ux 8.0% / Uz 14.5% | 96 步 119%，发散 |
| 4 | Transolver++ 3D | `legacy/tsolverpp/` | 05-17 | 3D 点云 574,163 | 未收敛（ep25/100） | alpha RMSE 0.036 | 93 步 RMSE 0.27，退化 |
| 5 | **纯 HPM** | `code/`（`window=6`） | 08-11 | 3D 点云 574,163 | **50 ep 跑完** | val 4 步 rollout nRMSE **0.2131** | 未测（lt 是 fwv 专属） |
| 6 | **HPM + FUNWAVE** | `code/`（`window=0`） | 08-04 | 3D 点云 574,163 | ep47/50（撞 24h 墙） | val 4 步 rollout nRMSE **0.1403** | **1000 帧不发散** ✅ |

**一句话结论**：前四条路线单步都能做，但自回归全部发散；HPM 这一代靠
**残差基座（自己上一帧 / FUNWAVE prior）+ 多步 BPTT + Δ=0 基线对照**，第一次把长期 rollout 稳住了。
难度排序始终是 `alpha < p_rgh < Ux < Uz`（`Uy`、`nut` 无效）。

---

## 1. 数据资产（`models/data/`，共 48 GB）

3D / 2D 数据源自同一个 OpenFOAM 算例：`$OCEAN_CASE` (/nfs/hpc/share/coast-lab/OpenFOAM/BA_TingKirby1994_3D_spilling/BA02_9M_Smag_old/0_org)
（interFoam 两相流，造波 + 海底地形，准二维，t = 0–100 s，Δt = 0.05 s）。

```
models/data/                               48G
├── 2d/                                   5.6G   ← §1.1  2D 切片（POD / Transolver++ / FNO 用）
├── fwv/                                  8.5G   ← §1.5  FUNWAVE 基准算例 TK94（仅此一个）
└── 3d/                                    34G
    ├── crop_fields.py / crop_fields.sh          ← 生成脚本（读 OpenFOAM 体场 + cellSet）
    └── cropped_0.05/                     34G    ← §1.2  **当前唯一在用的档**
```

名字里的数字是**展向（y）切片厚度**

| 档 | y 范围 | 厚度 | 点数 | `chunk_XXX_data.npy` | 总大小 |
|---|---|---|---|---|---|
| `cropped_0.05` | [0.2750, 0.3250] | 0.05 m | **574,163** | (100, 574163, 6) f32 = 1.28 GB | 34 G |
| ~~`cropped_0.1`~~ | [0.2500, 0.3500] | 0.10 m | 1,245,500 | (100, 1245500, 6) f32 = 2.78 GB | 28 G |
| ~~`cropped_0.3`~~ | [0.1500, 0.4500] | 0.30 m | 3,732,705 | (100, 3732705, 6) f32 = 8.34 GB | 84 G |

**怎么重造**：裁剪区域来自 OpenFOAM 的 cellSet。改`$OCEAN_CASE/system/topoSetDict` 里那个盒子的 y 上下界，跑 `topoSet`，再跑`crop_fields.sh`（改 `--output`）：

```
// topoSetDict：当前是 0.05 那档
box (-2.5 0.275 -0.41) (16.5 0.325 0.16);
//        ↑y下界          ↑y上界     ——  0.1 档用 0.25/0.35，0.3 档用 0.15/0.45
```

```bash
topoSet -case $OCEAN_CASE                          # 按新盒子生成 subdomainCells
cd data/3d && sbatch crop_fields.sh <目标目录>      # 不给参数则默认写 cropped_0.05
```

### 1.1 2D 切片 `data/2d/`

由 `postProcessing/sample`（y 中截面 `*_ySlice.raw`）导出，脚本 `legacy/transolver++/prepare_data.py`（Polars 读 `.raw`）。

| 文件 | 形状 | 说明 |
|---|---|---|
| `coords_2d.npy` | (149758, 2) float64 | (x, z)，x∈[-8.7, 16.45]，z∈[-0.4, 0.15] |
| `fields.npy` | (1001, 149758, 5) float64 | `[alpha.water, Ux, Uz, p_rgh, nut]`，5.6 GB |
| `times.npy` | (1001,) | t = 0 → 50 s，Δt = 0.05 s |
| `mesh_distribution.png` / `mesh_detail.png` / `alpha_snapshots.png` | | 网格与数据探查图 |

### 1.2 3D 裁剪 `data/3d/cropped_0.05/`（35 GB）

**主数据**（`crop_fields.py` 产出，通道顺序 `[alpha.water, Ux, Uy, Uz, p_rgh, nut]`）：

| 文件 | 形状 | 说明 |
|---|---|---|
| `coords.npy` | (574163, 3) float32 | 单元中心；x∈[-2.50, 16.45]，y∈[0.275, 0.325]，z∈[-0.40, 0.148] |
| `chunk_XXX_data.npy` | (100, 574163, 6) float32 | 每块 100 帧 ≈ **1.38 GB** |
| `chunk_XXX_times.npy` | (100,) float64 | chunk k 覆盖 t ∈ (5k, 5k+5] |
| `chunk_010_*` | (1000, …) | **例外**：t = 50–100 s 的 1000 帧，专供长期 rollout 演示 |

chunk 0–9 → t = 0–50 s。惯例划分：**train = 1–7，val = 8，test = 9**（chunk 0 含起振瞬态，不用）。

**派生资产**（都在同一目录下，按用途分）：

| 路径 | 内容 | 谁在用 |
|---|---|---|
| `lbo/lbo_eigenvectors.npy` | (574163, 128) float32，294 MB | **HPM 的谱基**，取前 `freq_num=64` 列 |
| `lbo/lbo_eigenvalues.npy`、`laplacian_info.txt` | 特征值 + 元信息 | 见下 |
| `prior_ktuned/prior_XXX_{data,valid,times,meta}.npy/json` | prior 001–010，每块 (100, 574163, 5) ≈ 1.15 GB | **fwv 线的基座** |
| `toffset_scan/c00X.json` | 逐 chunk 的 t-offset 标定结果 | `gen_prior.sh` 直接读，零转录 |
| `slice_y0.30/{slice_cell_map,slice_tri,slice_xz}.npy`、`sub_mesh.vtu` | y=0.3 切片的三角剖分与索引 | `vis.py` 的 tri 渲染 |
| `full_cell_ids.npy` | 裁剪单元 → 全网格 cell id | 回写 / 对照全网格 |
| `stats_*.npy` | (2, F) 的 [mean, std] | 训练/推理归一化，见 §1.3 |
| `slice_sanity_chunk6_f85.png` | 切片自检图 | 一次性核对 |

`laplacian_info.txt` 记录了谱基是怎么来的：
`cellSet=subdomainCells`，N=574,163，裁剪后内部面 1,590,604（全网格 28,093,814），
k=128，距离加权，特征值范围 [2.71e-04, 1.03]。
⚠️ **LBO 分解一旦重跑，特征向量的符号歧义会静默让旧 checkpoint 退化**（不会报错）——
`hpm_model.py` 里 `persistent=False` 的注释专门写了这一点。

`prior_XXX_meta.json` 是自描述的，抽 chunk 1 为例：

```json
{"chunk":1, "n_cells":574163, "n_frames":100,
 "channels":["alpha","Ux","Uy","Uz","p_rgh"],
 "x_offset":15.05, "t_offset":0.05, "beta":-0.531, "alpha":"sharp_heaviside", "pnh":true,
 "mglob":1575, "nglob":30, "dx":0.02, "dy":0.02, "plot_intv":0.05,
 "valid_ratio":0.890, "cells_always_valid":510767, "cells_never_valid":62876,
 "invalid_outside":32600, "invalid_below_bed":0, "dry_cells_total":63070,
 "dry_x_min":13.23, "dry_x_max":16.43, "horiz_points_outside_grid":114, "frames_missing":0}
```

要点：**约 11% 的单元 prior 无效**（干区 + 域外，集中在 x>13.2 的近岸），
`frames_missing: 0` 说明帧对齐没漏；`prior_XXX_valid.npy` 是逐帧逐点的 bool 掩码，供诊断用。

### 1.3 `stats_*.npy` 的命名规则

文件名由 `schema.py` 的 `signature()` 生成，**编码了「这份统计量属于哪个通道组合」**：

```
stats_{chunk 范围}_{通道列表}.npy
   stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.npy        纯 HPM 线（5 通道，原始速度）
   stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.nut.npy    纯 HPM 线（6 通道，含 nut）
   stats_c1-7_alpha.Ux.Uz.p_rgh.npy           fwv 线，非 αU
   stats_c1-7_alpha.Uxw.Uzw.p_rgh.npy         fwv 线，**αU 加权**（'w' 后缀）
   stats_c1-7_u{0,1}_nut{0,1}.npy             旧命名（u1 = alpha 加权速度），已弃用
   stats_c6_*.npy                             只用 chunk 6 的统计量，调试用
```

### 1.4 谁用哪份数据

| 路线 | 数据 | 备注 |
|---|---|---|
| POD / POD-LSTM | `$OCEAN_CASE/postProcessing/sample`（原始 raw） | 不经过 `models/data` |
| Transolver++ 2D | `data/2d/{coords_2d,fields,times}.npy` | 直接点云 |
| FNO | `data/2d/*` → `legacy/fno/processed_data/`（插值到规则网格，6.3 GB） | 多一步插值 |
| Transolver++ 3D | `data/3d/cropped_0.05/chunk_*` + `stats.npy`（缺失） | 见 §5.4-2 |
| 纯 HPM | `cropped_0.05/chunk_*` + `lbo/` + `stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.npy` | |
| HPM + FUNWAVE | 上面这些 **+ `prior_ktuned/`** + `stats_c1-7_alpha.Uxw.Uzw.p_rgh.npy` | |

另：`models/FUNWAVE-TVD/`（256 MB）是 Boussinesq 求解器源码 —— 第三方仓库的干净 clone（`Version_3.6`，**一个字节没改**），不进版本库，`./setup.sh` 缺了会自动拉。prior 用的实际输出在`/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output`。

### 1.5 FUNWAVE 基准算例 `data/fwv/`（8.5 GB，仅 TK94）

```
data/fwv/
├── TK94/                  基准（AMP_WK=0.0635, SLP=1:35），二进制编在这里
├── make_cases.py          从 TK94 造新算例（只改 input.txt 的 AMP_WK 和 SLP 两行）
└── wk_check.py            造波参数自检
```

TK94 与 CFD 真值同参数，是 `vis_adp.sh` 和 web-demo 的默认算例。8.5 G 全是输出，
**输入进版本库、输出不进** —— `input.txt` + `gauges.txt` 已跟踪，`.gitignore` 里
为此开了精确的例外；`LOG.txt`、`funwave--gnu-parallel-single` 二进制、场输出都排除。

> **这里曾经是一条 11 个算例的参数扫描线**：波高 `H0381`~`H0610` 五组（`SLP` 固定
> 1:35），以及变坡度的 `S325` = 1:32.5 / `S375` = 1:37.5 六组。2026-08-20 决定不再做
> 这个方向 —— 算例数据（93.5 G）、`results/fwv/` 下全部 prior 与渲染产物（124 G）
> 一并删除，版本库里也只留 TK94 的输入。
>
> **想重做扫描**：`make_cases.py` 从 TK94 复制并只改 `AMP_WK` / `SLP` 两行（其余逐
> 字节一致，所以算例间不存在隐藏变量），跑 FUNWAVE 拿到 `output/`，再用
> `vis_adp.sh STAGE=prior` 生成 prior。原始输出和 prior 都不随仓库分发，得自己跑。

`make_cases.py` 的严格原则：**只改 `AMP_WK` 和 `SLP` 两行，其余与 TK94 逐字节一致**，
`gauges.txt` 原样复制，可执行文件软链到 TK94 编好的那份 —— 所以算例之间的差异
就是文件名里那两个数，不存在别的隐藏变量。

---

## 2. POD / POD-LSTM（`legacy/pod-lstm/`，产物在 `$OCEAN_DATA`）

> §2–§5 为已退役路线，仅作记录。产物不随仓库分发（见 §7.2）；重跑前请先读
> [legacy/README.md](legacy/README.md)。

先以 POD 将场压缩为模态系数，再由 LSTM 预测系数演化，最后重构回物理场。
`pod_decomposition.py`（sklearn PCA）处理 1000 帧 × 149,758 点，每变量保留 300 模态，
产物约 1.8 GB（模态 / 系数 / 奇异值 / 能量谱）。

**能量收敛性**（达到该能量所需模态数）：

| 变量 | 90% | 95% | 99% | 第 1 阶占比 |
|---|---|---|---|---|
| `p_rgh` | 5 | 11 | 49 | 40.2% |
| `alpha.water` | 21 | 51 | 230 | 26.5% |
| `Ux` | 27 | 67 | 270 | 26.4% |
| `Uz` | 32 | 100 | >300 | 25.5% |
| `nut` | 189 | >300 | >300 | 16.6% |

该表是后续各路线通道取舍的依据：压力低秩，自由面与速度中等，`nut` 基本不可压缩 ——
后续模型对 `nut` 降权（`loss_weight=0.1`）或直接关闭。

LSTM 在系数空间训练，218 维（alpha 51 + Ux 67 + Uz 100，各取 95% 能量），
最佳配置为 v9（3 层 / hidden 128 / window 40 / 470,490 参数，早停 ep51）。

| | alpha | Ux | Uz |
|---|---|---|---|
| 系数空间 单步 | 10.4% | 11.9% | 24.6% |
| 系数空间 自回归 | 76.3% | 78.2% | 87.3% |
| 场空间 单步 rel L2 | 51.5% | 139.4% | 163.8% |
| 场空间 自回归 rel L2 | 51.3% | 141.0% | 163.3% |

**判定失败**：系数空间约 10% 的误差经重构放大至场空间 50% 以上。
注：现有重构数字取自 `lstm_results_v6_nop_wm` 而非最佳的 v9，公平比较需重跑。

---

## 3. Transolver++ 2D（`legacy/transolver++/`）

在原始 149,758 个网格点上直接做算子学习，不做插值。核心为 Eidetic Physics-Attention：
逐点投影为 token，经输入相关温度的 Gumbel-Softmax 软分配到 64 个物理切片，在切片间做注意力
后 deslice 回各点，复杂度由 O(N²) 降至 O(G²)。

配置 4 层 / hidden 128 / 4 头 / 64 切片，330,903 参数；train t∈[10,45)、test t∈[45,50]；
ep180 早停（上限 200，patience 30），H200 上 1.6 h，best val 0.034274 @ ep150。

| 场 | 归一化 MSE | 单步 rel L2 |
|---|---|---|
| alpha | 0.00295 | 3.47% |
| Ux | 0.02866 | 16.30% |
| Uz | 0.07125 | 26.37% |

自回归 100 步 rel L2：7.2% → 20.8%（5 步）→ 33.5%（10）→ 50.4%（20）→ 86.6%（50）→ 111.2%（100）。

**结论**：以 FNO 约 1/400 的参数量达到同量级单步精度，点云加切片注意力的路线可行；
但自回归误差近似线性增长，50 步后不可用。

**已知缺陷**（重跑前需处理）：

1. `transolver_pp.py` 第 220 行使用 `math.log` 而顶部未 `import math`，构造模型即 `NameError`。
2. `configs/default.yaml`（5 入 / 4 出）与 `results/` 中的 3 通道 rollout 不一致 ——
   已保存结果由改为滑窗版本之前的单帧模型产生。复现上表需回到单帧版本。

---

## 4. FNO（`legacy/fno/`）

将非结构网格插值到 4096×128 规则网格（Delaunay 重心坐标，权重预计算一次复用），
以标准 2D FNO 做「过去 5 帧 → 下一帧」：输入 5 帧 × 4 场 + 地形 mask 共 21 通道，输出 4 通道，
损失仅在 mask 内计算。

| 运行 | 通道 | 参数量 | 训练到 | 最好 val MSE |
|---|---|---|---|---|
| `2026-05-12/22-31-06` | 3 场，16 入通道 | 134.2 M | ep~175（4 h 时限） | ~0.0119（best.pt 为 ep102） |
| `2026-05-13/17-54-45` | 4 场，21 入通道 | ~134 M | ep30（被 kill） | 0.00932（best.pt 为 ep21） |

自回归 rollout（t=45 s 起 96 步，相对 L2）：

| 模型 | Step 1 | Step 49 | Step 96 |
|---|---|---|---|
| 3 场（ep102） | alpha 2.4% / Ux 8.2% / Uz 14.6% / 总 4.3% | 总 87% | 总 94% |
| 4 场（ep21） | alpha 2.3% / Ux 8.0% / Uz 14.5% / 总 3.7% | 总 96% | 总 119% |

**结论**：单步精度为前四条路线中最好，但 rollout 在约 50 步（2.5 s）内失效。
train loss 持续下降（1e-3 → 3e-4）而 val 停在 1.2e-2，为过拟合叠加单步训练与多步推理不匹配。
`modes1=256` 试过，显存与耗时翻倍且无改善。

---

## 5. Transolver++ 3D（`legacy/tsolverpp/`）

在 2D 版基础上改为：坐标 (x, y, z)、574,163 点、6 个场；`window=6` 的 21 维时间输入
（加权历史 + 一阶与二阶差分）；输出改为残差（`pred = fields[:,:,-1] + Δ`）；加权 MSE
（`p_rgh` / `nut` 权重 0.1）；`rollout_steps=4` 多步展开（前 3 步 detach）；按 chunk 划分
（train 1–7，val 8–9）。模型 hidden 256 / 6 层 / 8 头 / 64 切片 → 2,491,075 参数。

**该线未有一次训练跑完**：每 epoch 约 45 min，20 h 时限仅够约 25 epoch，而配置为 100。

| 运行 | 配置 | 跑到 | best val（加权 MSE） |
|---|---|---|---|
| `2026-05-19/03-21-52` | 默认 | E024 | 0.205489 @ E020 |
| `2026-05-19/12-58-01` | `dropout=0.2 weight_decay=1e-3` | E025 | 0.204876 @ E025 |

rollout 动画（midplane 377,215 点，93 步 ≈ 4.65 s，alpha RMSE）：chunk 6（训练区间）
起始 0.0389 / 结束 0.2714 / 均值 0.2096；chunk 9（验证区间）0.0357 / 0.2760 / 0.2161。

**结论**：两次运行结果几乎一致，瓶颈是训练时长而非正则化；训练与验证指标接近，
属欠拟合。RMSE 0.27 表现为界面模糊，但较 2D 版的完全发散稳定 —— **残差输出与多步训练
两项设计为 HPM 线所继承**。

**已知缺陷**：

1. `vis.sh` 中 `cd <repo>/tsolver3d` 为改名残留，应为 `tsolverpp`（现靠留在提交目录侥幸跑通）。
2. `data/3d/cropped_0.05/stats.npy` 已不存在（见 §1.3）。`dataset.py` 发现缺失会扫描目录下
   全部 chunk 重算（读约 13 GB）并写回。可复制 `stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.nut.npy` 代替，
   但注意 `dataset.py` 自算的含 chunk 0 与 10，且带 `u1` / `Uxw` 的变体是 alpha 加权速度，不可混用。
3. `train.sh` 写死 `--nodelist=dgxh-3`，该节点忙时会一直排队；`config.yaml` 的 `epochs: 100`
   与 20 h 时限不匹配（约需 75 h）。
4. 早期两次失败留档：一版 40 个 epoch 内 train/val loss 不动（梯度未流动）；12,991,637 参数的
   版本在 H200 上 OOM —— `gumbel_softmax` 单次需 28.5 GiB，瓶颈为 `(B, heads, N, slice_num)`。

---

## 6. HPM（`models/code/`）——当前主力，两条线

`code/` 是**唯一在迭代的目录**。它把前面所有教训收敛成一套：残差基座 + 多步 BPTT +
**每个指标都对照 Δ=0 基线**。

> `code/web-demo/` 是挂在这条线上的一个交互式演示页（把这套流水线跑给人看），
> **不在本文档范围内**，它有自己的 `README.md`。

两条线共用全部代码，**唯一的结构性差异是「残差加在谁身上」**：

| | base（基座） | state | 输入宽度 | R | scheduled sampling | 状态 |
|---|---|---|---|---|---|---|
| **纯 HPM** | 自己上一帧 | W=6 滑窗 | F（`window=6`） | 4 | 无（`ss=false`，p 恒 0 全自回归） | 前一阶段 |
| **HPM + FUNWAVE（fwv）** | **prior(t)** | 单帧反馈槽 | 2F（`window=0`） | 4 | p 1.0→0.1（前 60% epochs） | **活跃** |

由 `data.window` 单参数区分（`0` = fwv，`>=3` = 纯 HPM）。二者不是自由组合：`window>0` 才有历史帧
可作基座。`build_policy` 有断言，写错组合直接失败而不是静默跑成别的东西。

### 6.1 模型本体 `hpm_model.py`

HPM = Holistic Physics Mixer（ICML 2025），核心是 **Calibrated Spectral Mixer**：

1. 用 OpenFOAM 图 Laplacian 预算好的 **LBO 特征向量**作固定谱基（§1.2，取前 `freq_num=64` 列）；
2. 可学习 gate 网络逐点预测「频率偏好」→ `eigens = gate * basis`；
3. 正变换到谱域 → LayerNorm + 谱域线性混合 → 逆变换回物理点。

工程上两处优化：谱基只存**一份** (N, G) 并广播（而非每 head / 每 block 一份），`persistent=False`
不进 checkpoint —— 显存 50 GB → **42.8 GB**，输出 bit-identical。早期 checkpoint 里那份
持久化的谱基已经全部剥干净（见 §7.4），加载路径不再做任何兼容处理。

模型规模：`n_hidden=128 / 6 层 / 8 头 / freq_num=64 / mlp_ratio=2` → 约 **74 万参数**
（fw 线 740,852，纯线 742,780）。

### 6.2 纯 HPM 线（`window=6`，纯自回归）

- 输入 coords(3) + 时间特征 3F（macro 加权历史 + 一阶差分 + 二阶差分），与 §5 的 3D Transolver++ 同构；
- 基座 = 窗口末帧，模型只出 Δ；
- **`ss=false` → p 恒 0，训练时每一步都喂自己的预测**（= 部署条件），所以没有 exposure bias 的概念，
  也不存在冷启动问题（窗口本身就是状态）；
- R=4 真 BPTT（序列内严格顺序，序列间随机起点）。

**结果**：

| run | 通道 | 跑到 | best val（4 步 rollout nRMSE） |
|---|---|---|---|
| `hpm_bl_h128`（nut 开，6 通道） | alpha, Ux, Uy, Uz, p_rgh, nut | ep8（A40，1.8 h/ep，中断） | 0.1881 @ ep6 |
| `hpm_no-nut_h128`（nut 关，5 通道） | alpha, Ux, Uy, Uz, p_rgh | **50 ep 全部跑完**（H100，1678 s/ep） | **0.2131 @ ep16** |

⚠️ 两者**通道集不同**（6 vs 5），聚合 nRMSE 不在同一空间，不能直接比大小。

`hpm_no-nut_h128` 逐通道（末轮 ep49，模型 / Δ=0 基线，`✓` = 优于基线）：

| alpha | Ux | Uy | Uz | p_rgh |
|---|---|---|---|---|
| 0.071 / 0.189 ✓ | 0.206 / 0.353 ✓ | **1.000 / 1.031** | 0.361 / 0.604 ✓ | 0.071 / 0.286 ✓ |

- alpha 和 p_rgh 做到基线的 ~1/3，很好；
- **`Uy ≈ 1.0`**：准二维算例里 Uy 本身接近噪声，模型学不到也不该学 —— 这正是 fwv 线关掉 Uy 的依据；
- **val 从 ep16 的 0.2131 一路回升到 ep49 的 0.2367，train 却仍在降** → 明确过拟合，
  这条线 50 epoch 偏多，早停或加正则是下一步。

### 6.3 HPM + FUNWAVE 线（`window=0`，以 fwv 为骨架）

**核心想法**：不让网络从零外推时间，而是让 **FUNWAVE（Boussinesq 波浪求解器）先给出 t 时刻的
物理先验场 prior(t)**，网络只学「prior → CFD 真值」的修正量 Δ。基座从「自己上一帧」换成
「每步由外部物理模型重新提供的场」，误差就不再纯靠自己累积。

#### prior 生产管线（三阶段，都在 `code/`）

```
阶段一  scan.sh      → scan_toffset.py            逐 chunk 标定 t-offset  → toffset_scan/c00X.json
阶段二  gen_prior.sh → gen_prior.py + lift.py     Nwogu 剖面抬升成 3D     → prior_ktuned/
阶段三  训练时 dataset.py 丢 Uy 取 4 通道
```

- **`lift.py`（抬升算子）**：按 Nwogu (1993) 把 2D 状态 (η, u, v) 抬升成 3D：
  `Ux = u + (za−z)∂A/∂x + ½(za²−z²)∂B/∂x`（二次剖面）、`Uz = −(A + zB)`（连续性）、
  `p = ρgη − ρ[Ȧ(η−z) + ½Ḃ(η²−z²)]`，`alpha` 是锐利 Heaviside。**零自由参数**；剖面对 z 解析，
  所以 z 方向不做任何插值；干单元 → NaN，不置零不外推。
- **`scan_toffset.py`（时间配准）**：FUNWAVE 与 CFD 时间原点不对齐，逐 chunk 扫最优帧偏移。
  实测 `best_k[1..9] = [1,3,5,5,4,3,2,4,6]` —— **非单调**，没有「漂移模型」可套，只能逐 chunk 标。
  chunk 9 此前被判无效（固定 offset 下 Uz nRMSE = 1.02 > 1），重标后 k=+6 → 0.957，
  **是配准误差不是数据问题**，chunk 9 恢复为有效 test chunk。
  每个 `c00X.json` 还带逐通道的 `k_rmse / k_corr / k_subframe / contrast / flat` 诊断，
  `Uy` 恒为 `flat: true`（曲线平坦，不计入）—— 准二维的正确信号。
- x 方向偏移 `XOFF=15.05` 当已知输入，不标定。

#### prior 质量诊断 → 通道取舍

（chunk 1–9，`prior_ktuned`，raw 空间，跨 chunk 均值）

| 通道 | nRMSE_prior | corr | 判定 |
|---|---|---|---|
| alpha | 0.410 | 0.918 | 有效 |
| p_rgh | 0.647 | 0.806 | 有效 |
| Ux | 0.896 | 0.521 | 弱有效 |
| Uz | 0.930 | 0.403 | 弱有效 |
| **Uy** | 1.002 | −0.001 | **无效** —— 比常数基线还差，prior 在加噪 → 关掉 |
| **nut** | — | — | **结构上没有 prior**：FUNWAVE 是无粘 Boussinesq → 关掉 |

所以 fwv 线只留 4 个通道：`alpha, Ux, Uz, p_rgh`。

#### 反馈槽、SS、αU

`window=0` 意味着模型没有历史，于是引入**反馈槽**：

- `feedback=none` → 输入 = `[prior]`，F 通道（纯「prior 场 → CFD 场」单步映射，无 rollout 无冷启动）；
- `feedback=self` → 输入 = `[prior | x_f·m]`，2F 通道；屏蔽由 `x_f·m` 完成（算术恒等，无需额外 mask 列）。

有了槽就有 exposure bias，于是有 SS：`p` = 喂 GT 的概率，1.0 → 0.1 退火（前 60% epochs），
**val 恒 p=0，与部署一致**。`Ux/Uz` 的 `alpha_weighted=true`（物理空间乘水相体积分数）是
**当前最佳臂** —— 空气区的速度不该参与评价。

#### 结果

| run | 跑到 | best val（4 步纯 rollout nRMSE） |
|---|---|---|
| `hpm_fw_aU_h128 / 2026-08-04_14-37-31` | ep47/50（撞 24 h 墙） | 0.1512 @ ep22 |
| `hpm_fw_aU_h128 / 2026-08-12_15-31-45` | ep47/50（**又**撞 24 h 墙） | **0.1403 @ ep19** |

逐通道（末轮 ep47，模型 / Δ=0 基线）：

| alpha | Ux | Uz | p_rgh |
|---|---|---|---|
| 0.221 / 0.370 ✓ | 0.436 / 0.619 ✓ | 0.602 / 0.729 ✓ | 0.318 / 0.562 ✓ |

**四个通道全面优于「不学习直接用 prior」的基线** —— 这是这条线成立的最小证据。

chunk 9（test）100 帧冷启动 rollout，y=0.3 切片（107,466 cells）RMSE，用 ep19 的 best.pt：

| 场 | teacher forcing（末帧 / 均值） | rollout（末帧 / 均值） | gap = exposure bias |
|---|---|---|---|
| alpha | 0.0415 / 0.0365 | 0.1614 / 0.1309 | +0.120 |
| αUx | 0.0551 / 0.0390 | 0.1405 / 0.1116 | +0.085 |
| αUz | 0.0163 / 0.0145 | 0.0499 / 0.0465 | +0.034 |
| p_rgh | 34.07 / 27.09 | 152.34 / 119.70 | +118.3 |

同一次 rollout 的误差分布（`DIFF=both`，2026-08-16 补跑，S = GT 满量程）：

| 场 | S | bias | MAE | max\|Δ\| | MAE% | max% |
|---|---|---|---|---|---|---|
| alpha | 1.000 | −0.0044 | 0.0298 | 1.073 | 2.98% | 107.3% |
| αUx | 2.700 | −0.0278 | 0.0524 | 2.224 | 1.94% | 82.4% |
| αUz | 1.255 | −0.0006 | 0.0174 | 0.734 | 1.39% | 58.5% |
| p_rgh | 1734.8 | −10.11 | 53.15 | 1305 | 3.06% | 75.3% |

**四场 bias 都是负的但都很小**（−0.04% ~ −1.03%）—— 没有系统性偏高/偏低，误差是局部结构性的
而不是整体漂移。而 max|Δ| 顶到满量程量级（alpha 107%，即有 cell 完全错反），MAE 却只有 1.4~3.1%：
**误差高度集中在少数 cell 上**。这批 cell 具体落在哪（是否就是下面「误差分解」里说的破碎区）
要看视频，尚未逐帧核对 —— 视频在
`results/vis/pred/hpm_fw_aU_h128/2026-08-12_15-31-45_diffboth/`。

**长期 rollout（最重要的一条）**：chunk 10 上连续 **1000 帧（50 s 物理时间）没有崩**，
推理 **722 ms/帧**（574,163 cells，单卡），两个 run 各有一份
`results/vis/lt/hpm_fw_aU_h128/<时间戳>/longterm_chunk10_alpha_tri.mp4`。
**这是所有路线里第一次做到长期不发散。**

#### 误差分解

fw 线自回归 rollout 误差约 **0.125**，拆成两段：

- **≈0.10 的地板** = FUNWAVE prior 在破碎区失效。Boussinesq 理论在翻卷区本身就崩溃，
  这段不是模型能解决的；
- **≈0.04 的累积** = exposure bias，SS 针对的是且只是这段。

**这个分解决定了优先级**：继续调 SS 最多拿回 0.04；要突破 0.10 得换 prior。

### 6.4 工程约定

- **单 yaml + CLI 覆盖**：`config.yaml` 是唯一配置文件，头部带**复现表**（每个跑过的 run 一行命令）。
  纪律是「改 yaml = 定义新 baseline，跑变体走 CLI」。
- **`override_dirname`**：CLI 覆盖自动进 hydra 目录名和 wandb run 名，改 yaml 不会 ——
  所以靠改 yaml 跑两个变体会同名冲突。
- **`schema.py` 单一真源**：通道 enable / update_rule（delta | frozen）/ loss_weight / alpha_weighted
  全部从这里派生，包括 **stats 文件名签名**（§1.3）。磁盘文件恒为 6 通道、按**名字**选列 →
  **消融不需要重新生成数据**。
- **`run.sh pure` 快捷方式**：切纯 HPM 线要带四组参数（窗口/反馈/SS/通道），已封装；
  忘带 `rollout.ss=false` 会被 `build_policy` 断言拦住。
- **`vis.sh` 定位机制**：`CONFIG=... CKPT=...` 显式，或 `RUN=runname [TS=时间戳]` 从
  `results/train/` 解析（省 TS 取最新且含 best.pt 的）；输出目录非空时拒绝覆盖，除非 `FORCE=1`。
- **`vis.py` 子命令**：`gt`（纯数据探查）/ `align`（配准检查，训练前）/ `lift`（只渲 prior，
  FUNWAVE 现算）/ `pred`（GT|pred 两行，两条线通用，附 tf/rollout gap 自检）/ `nofb`（无反馈臂
  三行）/ `lt`（长期 rollout，流式，仅 fwv）。**`--diff` 只挂在 `pred` 上**（其余子命令没有 GT，
  无从算 Δ），走 `vis.sh` 时是 `DIFF=abs|pct|both`，见 §6.6「误差可视化」。

### 6.5 已排除的方向

| 方向 | 排除理由 |
|---|---|
| Clifford / 几何代数 | 域无旋转对称（重力各向异性 + 斜底 + 有界域）；`mlp_trans_weights` 训练后无可学习结构 |
| 逐点 PDE 时间导数残差 | Courant C≈1.6（近空气区），残差形式不成立 |
| 可微分求解器 | MULES 不可微 |
| 全局不变量约束 | 开放耗散域，不守恒 |
| 对称性 / 等变性约束 | 域各向异性 |
| mask 通道（2F+1） | 屏蔽已由 `x_f·m` 完成（算术恒等），mask 列只是显式告知，是效率不是必要性 |

### 6.6 怎么跑

```bash
cd <repo>/code && mkdir -p logs

# --- fwv 线（默认 = 当前最佳臂 hpm_fw_aU_h128）---
sbatch run.sh                                    # prior + 自反馈 + SS(R=4) + αU
sbatch run.sh rollout.R=8                        # 变体（自动进目录名/run 名）
sbatch run.sh rollout.feedback=none rollout.R=1 \
      data.channels.1.alpha_weighted=false \
      data.channels.3.alpha_weighted=false       # 无反馈基线臂

# --- 纯 HPM 线 ---
sbatch run.sh pure                               # 等价于四组参数，见 run.sh
sbatch run.sh pure data.channels.5.enabled=false # nut 消融

# --- prior 生产（仅 fwv 线，训练前一次性）---
sbatch scan.sh                                   # 阶段一：逐 chunk 标 t-offset
sbatch --dependency=afterok:<scan_jobid> gen_prior.sh   # 阶段二：抬升成 prior_ktuned/

# --- 可视化 ---
RUN=hpm_fw_aU_h128 sbatch vis.sh                 # 默认 SUB=pred, chunk 9
SUB=lt RUN=hpm_fw_aU_h128 CHUNK=10 sbatch vis.sh # 长期 rollout（仅 fwv）
python vis.py align --fw-dir <fw>/output --chunk 9        # 训练前配准检查

# --- 误差行（DIFF，仅 pred；默认不渲，见下方「误差可视化」）---
DIFF=both STYLE=tri RUN=hpm_fw_aU_h128 \
  FEATURE=hpm_fw_aU_h128/<时间戳>_diffboth sbatch --partition=gpu vis.sh
```

#### `vis_adp.sh` 的输出路径可以覆盖

`OUTROOT` / `PRIORROOT` / `VISROOT` / `LIFTROOT` / `CKDIR` 都写成了 `${VAR:-默认}`，
**不传就完全是原来的行为**（`results/fwv/{priors,vis,lift}` + `results/train/…`）。
传了就把产物写到别处，两边互不干扰：

```bash
STAGE=prior CHUNK=10 CASES='TK94' \
  PRIORROOT=$REPO/results/web/priors sbatch ... vis_adp.sh
```

加这几个变量是为了让 `code/web-demo/` 那个演示页把产物写进它自己的 `results/web/`，
不混进 ADP 扫描线的 `results/fwv/`（该目录已于 2026-08-20 清空，再跑会重建）。
手动跑 ADP 时什么都不用传。

#### 误差可视化（`--diff` / `DIFF=`）

`pred` 默认渲两行（GT | pred）。给 `DIFF=` 会在下方追加误差行 Δ = pred − GT，逐帧成动画，
用于观察误差的空间分布而非仅看 RMSE 一个数：

| `DIFF=` | 画什么 | 色标 | 用途 |
|---|---|---|---|
| `abs` | Δ，物理单位 | 自适应 ±p99\|Δ\|（`DIFF_PCT=` 调） | 看结构，误差再小也撑满画面 |
| `pct` | Δ% = Δ/S × 100 | 固定 ±100% | 跨 run / 跨 ckpt 并排比较 |
| `both` | 两者都渲（共 4 行） | | |

配套参数：`PCT_SCALE=range|rms|p99`（Δ% 的分母 S，默认 `range` = GT 满量程）、`DIFF_PCT=99`
（仅 abs 行）、`ROW_H=`（每行英寸高）。误差行使用独立的 coolwarm 对称色标（红 = 偏高、
蓝 = 偏低、白 = 准），不与主色标共用 —— 主色标会把负误差压到下端画没。

三点注意：

1. `abs` 与 `pct` 的差别在色标而非分母 —— S 在着色位置的分子分母中各出现一次会约掉，
   自适应色标下两行是同一张图；`pct` 行锚定 ±100% 才是它可跨 run 比较的机制。
2. `DIFF=both` 为 4 行，默认 `row_h=10.8` → 4320 px，越过 4096 的播放器硬解线。
   `vis.sh` 在未显式给 `ROW_H` 时自动降到 10.0（= 4000 px）。
3. alpha 上 `both` 基本冗余（S=1.0，p99 自适应为 ±0.894，与 ±100% 仅差 11%），看 `abs` 即可；
   真正拉开差距的是 Ux（±0.475 vs ±2.70）与 p_rgh（±556 vs ±1735）。

### 6.7 ⚠️ 已知问题

1. **未固定随机种子**。同配置重跑 `best_val` 有可观抖动（三次实测 0.1549 / 0.1512 / 0.1403），
   刷新出现的 epoch 也大幅移动（ep10 / ep22 / ep19），而末 15 个 epoch 的收敛终点几乎重合。
   **「重跑一遍」目前不构成可比的证据**，要当证据用必须先固定 seed。
2. **上表数字产自旧环境，与当前环境不是逐位可比**。2026-08-19 把 `ocean` 里的 numpy
   从 conda-forge 构建换成了 PyPI 构建（**版本号同为 2.2.6，只是编译产物不同** ——
   BLAS 从 conda 的 openblas 换成 wheel 自带的那份）。torch / torchvision 及其余 16 个
   依赖都没动，GPU 上的重活也全在 torch 手里，numpy 只参与数据预处理与统计，影响
   预期在浮点最低位；但严格说，跨这条环境线的数字只能定性比，不能当作精确复现。
   要拿新数字得在当前环境重跑一轮（与第 1 条叠加：还得先固定 seed）。
3. **αU 与非 αU 的 nRMSE 不在同一空间**，不能直接比大小（归一化的量从 `Ux` 换成 `αUx`，
   Δ=0 基线 0.896 vs 0.619，见 §1.3）。跨臂比较只能用共同空间的指标 —— `vis.py pred` 的
   raw 空间 slice-RMSE，或分区域的近岸形态。同理**纯线与 fw 线的 val 也不可直接比**。
4. **50 epoch × ~1780 s ≈ 24.7 h，撞 24 h SLURM 墙**——两次 fw run **都**停在 ep47。
   末 15 个 epoch 已平，降到 30 可跑完且不损失。纯线 1678 s/ep 刚好卡着跑完 50。
5. `update_rule: flux_div` 是预留占位，会抛 `NotImplementedError`。
6. 绝不要 `update_rule=delta` 配 `loss_weight=0.0`：无监督的 head 会污染 rollout，
   要去掉通道请用 `frozen` 或 `enabled: false`。

---

## 7. 结果与产物（`models/results/` 及各线输出）

### 7.1 `results/` 现状

2026-08-22 起 **`results/` 下只有 demo 的默认权重随仓库走**（`web/model/`，8.6 M），
其余全部移出并打包：

| 内容 | 现在在哪 |
|---|---|
| `train/` 87 M、`vis/` 56 M | `results_20260822.tar`（194 M / 69 个文件） |
| `web/` 演示页状态（中间场 / mp4 / 提交记录） | 不打包 —— 点一次自动重建，见 `code/web-demo/README.md` |
| `fwv/` ADP 扫描线产物 124 G | 2026-08-20 整个删除（见 §1.5），再跑 `vis_adp.sh` 会重建 |

仓库里留的是带 `.gitkeep` 的空目录骨架，每个 `.gitkeep` 写明内容在哪个包、怎么还原
（`./archive/restore.sh`，清单见 [archives.tsv](archives.tsv)）。还原之后是这个结构：

```
results/
├── train/<runname>/<override_dirname>/<时间戳>/     70M
│   ├── .hydra/{config,overrides,hydra}.yaml         ← 这次运行的**实际**配置（真相源）
│   ├── checkpoints/{best,latest}.pt                 ← 各 9.0 MB（已不含 LBO 基）
│   └── train.log                                    ← 通常是空的，真日志在 code/logs/
└── vis/<sub>/<runname>/<时间戳>/                    44M
    ├── pred:  compare_chunk9_<field>_pred_{tri,scatter}.mp4
    └── lt:    longterm_chunk10_alpha_tri.mp4
```

末级目录名由 `FEATURE=` 决定（默认 `<runname>/<时间戳>`）。同一个 ckpt 换渲染口径重跑时，
**加后缀另开目录**而不是 `FORCE=1` 覆盖 —— 例如误差行那次用的是
`FEATURE=hpm_fw_aU_h128/2026-08-12_15-31-45_diffboth`（12M，四场各一个 4 行 tri 视频）。

目前有的 run：

| runname | 线 | 时间戳 | best_val | 备注 |
|---|---|---|---|---|
| `hpm_fw_aU_h128` | fwv | `2026-08-04_14-37-31` | 0.1512 @ ep22 | 有 pred + lt 视频 |
| `hpm_fw_aU_h128` | fwv | `2026-08-12_15-31-45` | **0.1403 @ ep19** | 有 pred + lt 视频，另有 `_diffboth` 误差行一套，当前最佳 |
| `hpm_bl_h128` | 纯 | `…window-6…/2026-08-11_22-58-29` | 0.1881 @ ep6 | 只跑到 ep8 |
| `hpm_no-nut_h128` | 纯 | `…enabled-false…/2026-08-12_16-24-16` | 0.2131 @ ep16 | 50 ep 跑完，无 vis |

`override_dirname` 那一长串目录名不是噪音，它就是**这次 run 相对默认配置的 diff**
（例：`data.window-6_rollout.feedback-none_rollout.ss-false…` 一眼能看出是纯 HPM 线）。

**中间产物**：`vis.py pred` 会顺手存 `compare_chunk9_<field>_preds.npy`
（(100, 574163, 4) float32 = **918 MB / 个**，用于「改配色重渲染不必重跑推理」）
和 `_rmse{,_tf}.npy` 曲线。这批中间件目前已清掉，只留 mp4 —— 需要重渲染时重跑一次 `vis.sh` 即可。

### 7.2 旧线的产物在哪

| 路线 | 产物位置 | 体积 | 内容 |
|---|---|---|---|
| POD | `$OCEAN_DATA/pod_results/` | 1.8 G | 模态 / 系数 / 能量谱 / `mode_summary.txt` |
| POD-LSTM | `$OCEAN_DATA/lstm_results_v*/` ×9 | 小 | `best_model.pt`、`results_summary.json`、`var_info.json`、曲线图 |
| 场重构 | `$OCEAN_DATA/reconstruction_results/` | 中 | 预测/真值场 npy + `reconstruction_errors.json` + 快照图 |
| Transolver++ 2D | `legacy/transolver++/results/` | 354 M | `best_model.pt`、`training_history.json`、`rollout_{pred,gt}.npy`、`figs/` |
| FNO | `legacy/fno/outputs/`（在 `legacy_20260822.tar` 里） | **16 G** | `best.pt` + `epoch_*.pt`（134 M 参数 → 每个 ckpt 约 1.6 GB）+ `visualizations/` |
| FNO 中间数据 | `legacy/fno/processed_data/`（同上） | 6.3 G | 插值后的规则网格数据 |
| Transolver++ 3D | `legacy/tsolverpp/outputs/<日期>/<时间>/checkpoints/` | 129 M | `best.pt` / `latest.pt`；mp4 直接躺在 `legacy/tsolverpp/` 根目录 |
| HPM 上一代 | `legacy/hpm/`：`outputs/` 429 M 和 `vis/` 487 M 各有自己的包（`legacy_hpm_*.tar`）；`wandb/` 372 M 移出未打包；**`fwv/` 58 M 留在版本库里** | | 重构前的实现与产物。`fwv/` 的 40 个渲染 mp4 于 2026-08-23 收进版本库 —— 它们是全仓库唯一既不在 git 也不在包里的东西 |

**日志分布**（找一次运行的完整输出）：`legacy/fno/logs/`、`legacy/tsolverpp/logs/`、
`legacy/transolver++/tsolver_pp_*.log`、`code/logs/`（`hpm_<jobid>.log` + `.err`）、
`$OCEAN_DATA/*.log`。Hydra 的 `train.log` 基本是空的，**真日志看 SLURM 的那份**。

### 7.3 追溯一次运行的标准流程

1. 从 `results/train/<runname>/<override_dirname>/<时间戳>/` 找到那次 run；
2. 读 `.hydra/overrides.yaml` —— **这次改了什么**；读 `.hydra/config.yaml` —— **完整实际配置**
   （比读仓库里的 `config.yaml` 可靠，后者随时会变）；
3. 到 `code/logs/hpm_<jobid>.log` 看逐 epoch 曲线与逐通道 nRMSE / 基线对照；
4. `RUN=<runname> TS=<时间戳> sbatch vis.sh` 复现可视化（`vis.sh` 会自己去同一路径取 ckpt 和 config）。

### 7.4 磁盘与清理

2026-08-22/23 把数据和产物全部移出仓库、打成五个包（见 [archives.tsv](archives.tsv)）。
现在 clone 下来只有代码 + 空目录骨架：

| | 仓库里 | 包里 |
|---|---|---|
| `data/` | 19 K（6 个跟踪的脚本与输入） | `data_20260822.tar` 48.6 G |
| `legacy/`（除 hpm） | 7 M（只有源码） | `legacy_20260822.tar` 12.7 G |
| `legacy/hpm/` | 60 M（源码 + `fwv/` 的 40 个 mp4） | `legacy_hpm_{vis,outputs}_*.tar` 共 914 M |
| `results/` | 8.7 M（demo 权重） | `results_20260822.tar` 194 M |
| `code/` | 296 M —— 其中 290 M 是 `web-demo` 的 cargo/npm 构建缓存，可再生 | — |
| `FUNWAVE-TVD/` | 256 M，第三方 clone，不进版本库 | — |
| `.git` | 478 M ⚠️ | — |

⚠️ `.git` 那 478 M 基本全是 `hpm/vis/` 的 306 个 mp4（436 M，从 commit `d8c9d1c` 起就在
历史里）。它们已经不在工作区，但**历史对象还在，clone 照样要拉** —— 真瘦身要 filter-repo
改写历史。

**checkpoint 瘦身（2026-08-13 做过，工具已删）**：早期 checkpoint 把 LBO 谱基
（每 head / 每 block 一份）持久化进了权重文件，单个 ckpt 6.58 GiB，其中 6.6 GB 是重复的基。
当时写了 `code/strip_ckpt.py` / `strip_ckpt.sh` 批量剥离：**66 个文件，6.58 GiB → 8.6 MiB**，
`legacy/hpm/outputs` 从 409 G 降到 396 M（释放 407.9 GB）。

2026-08-18 复核：现存 **37 个 checkpoint（含 `legacy/hpm/outputs` 全部老货）legacy 键都是 0**，
工具使命完成，已连同 `hpm_model.py` 的 `strip_legacy_basis()` 一起删除 —— `train.py` 的
resume 和 `vis.py` 的加载都改成直接 `load_state_dict(..., strict=True)`，实测能读现役 ckpt。
真要找回：`git checkout <删除前的提交> -- code/strip_ckpt.py`。

**`.gitignore` 的原则**：产物与数据一律不进版本控制（`*.pt`、`*.npy`、`processed_data/`、
`logs/`、`outputs/`、`wandb/`、`data/*`、`vis*/`、`FUNWAVE-TVD/`、`.env.local`）。
所以 `.hydra/config.yaml` 是唯一能追溯历史配置的东西，别删。

**刻意开的例外**（都在 `.gitignore` 里逐层放行，各有注释说明理由）：

| 例外 | 为什么 |
|---|---|
| `data/fwv/TK94/{input.txt,gauges.txt}` + `make_cases.py` / `wk_check.py` | 「到底跑了什么」的唯一记录（§1.5） |
| `data/3d/crop_fields.{py,sh}` | 「数据怎么造出来的」 |
| `results/web/model/**`（除 `.hydra/hydra.yaml`） | demo 的默认权重，不带上 clone 下来下拉框是空的 |
| `code/web-demo/{web/dist,server/target/release/wave-demo}` | 让 clone 下来直接 `./start.sh`，不用编 |
| `legacy/hpm/fwv/{vis,hpm_fw_ss_R4}/` | 40 个渲染 mp4，55 M（2026-08-23 收进来） |
| `archive/restore.sh` + `archive/*.manifest` | 还原器和逐文件校验单，clone 就得有 |
| 各占位目录的 `.gitkeep` | 目录骨架，写明内容在哪个包 |

---

## 8. 横向对比与总体结论

### 8.1 六条路线的演化脉络

```
POD-LSTM      降维 + 序列模型          → 重构误差放大，判失败
   ↓ (放弃降维，直接在网格上学)
Transolver++  点云注意力，单步          → 单步好，rollout 发散
FNO           规则网格谱方法，单步      → 单步最好，rollout 发散更快
   ↓ (引入残差输出 + 多步 rollout 训练)
tsolverpp 3D  残差 + R=4 BPTT           → rollout 不再爆炸但欠拟合，训不完
   ↓ (换 LBO 谱基骨架 + Δ=0 基线对照)
纯 HPM        残差加在自己上一帧        → 各通道稳定优于基线
   ↓ (把基座换成外部物理先验)
HPM+FUNWAVE   残差加在 prior(t) 上      → 1000 帧长期 rollout 不发散 ✅
```

### 8.2 反复出现的三个现象

1. **单步好、rollout 崩**（前四条路线）。根因是训练目标（单步 MSE）与推理方式（自回归）不匹配。
   HPM 线用「残差基座 + 多步 BPTT + val 直接用纯 rollout 口径」正面解决了。
2. **难度排序恒定**：`alpha` < `p_rgh` < `Ux` < `Uz`（`Uy`、`nut` 无效）。
   `Uz` 量级小（3D αU 空间 std 0.058 vs `Ux` 0.182；raw 空间 0.101 vs 0.267）、尺度细，信噪比最差。
3. **`nut` 与 `Uy` 都该关掉，但理由完全不同**：`nut` 是**低秩性差**（POD 说 189 个模态才 90% 能量），
   `Uy` 是**信号本身接近噪声**（准二维算例，模型 nRMSE 1.000 ≈ 基线 1.031，
   连 prior 标定曲线都是平的）。

### 8.3 方法论上真正的进步

不是模型换了几次，而是**判据换了**：

| 时期 | val 口径 | 问题 |
|---|---|---|
| POD-LSTM / Transolver++ / FNO | 单步 MSE 或系数空间误差 | 与部署条件（自回归）脱节，看不出 rollout 会崩 |
| tsolverpp | 加权 MSE，仍是训练损失 | 没有「不学习」的对照 |
| **HPM** | **R 步纯 rollout nRMSE（p=0 = 部署条件）+ 每通道对照 Δ=0 基线** | 能直接回答「模型有没有比什么都不做更好」 |

---

## 9. Hydra 输出约定

环境怎么装、集群怎么用，全在 **[SETUP.md](SETUP.md)**。这里只留一条读结果时要知道的约定。

`fno`、`transolver++`、`tsolverpp` 的输出是 `outputs/<日期>/<时间>/`；
`code` 是 `results/train/<runname>/<override_dirname>/<时间戳>/`。两者都有
`.hydra/config.yaml`（实际配置）、`.hydra/overrides.yaml`（CLI 覆盖）、`checkpoints/`。
**排查任何一次历史运行，先看它的 `.hydra/overrides.yaml`**（详见 §7.3）。
