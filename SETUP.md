# 环境与集群操作

> 这份只管**怎么把环境弄起来、怎么在集群上跑**。
> 项目本身（六条模型线、数据资产、结果）看 [README.md](README.md)；
> web 交互演示看 [code/web-demo/README.md](code/web-demo/README.md)。

**只能在集群上跑**（数据在 `/nfs/hpc/share`、脚本是 SLURM 的、torch 要 cu130），
在别的机器上执行 `setup.sh` 会直接报错退出。

---

## 快速开始

```bash
cd /nfs/hpc/share/$USER
git clone -b fwv https://github.com/Picomp-lab/Ocean_OpenFOAM.git models
cd models && ./setup.sh
```

跑完之后：

```bash
source activate.sh                # 交互式用；之后可用 $REPO
cd code && sbatch run.sh          # 训练（sbatch 脚本自己会 source，不用先激活）
```

**唯一 `setup.sh` 解决不了的是数据** —— 254 GB 不在版本库里，它只会告诉你缺哪些、
能不能自己重算、得找谁要（第 5 步会分开讲）。

---

## setup.sh

```bash
./setup.sh            # 一条命令跑完：缺什么补什么
./setup.sh --check    # 只检不装（退出码非 0 = 有缺的）
```

**只有这一个开关。** 其余全自动：conda 环境和 FUNWAVE-TVD 缺了就装，`archive/` 里有包
就自动解压。**`archives.tsv` 里的包一个都不能少** —— 少了就算缺件，脚本以非 0 退出
（各步的检测照样全跑完再退）。三个环境变量可微调，正常不用管：
`OCEAN_ENV`（环境位置）、`OCEAN_ARCHIVE`（大文件包目录，默认 `<仓库根>/archive`）、
`SRUN_WAIT`（等计算节点的秒数，设 0 就全留在登录节点）。

五步：平台/SLURM → conda 环境（默认 `/nfs/hpc/share/$USER/.conda/envs/ocean`，
`OCEAN_ENV=` 可改；顺带生成 `.env.local`）→ python 依赖 → wandb → 目录与数据
（含 `archive/` 自动解压、FUNWAVE-TVD 自动 clone）。web-demo 不在其中，见下。

探测和 pip 安装默认 `srun -p share` 丢到计算节点（实测二十几秒调度到，计算节点能出网）
—— 登录节点每用户 `RLIMIT_NPROC=400` 且全节点共享，挤满时 numpy 起不来，好包会被误判
成坏的。**不用 `preempt`**：会被抢占 requeue，几十秒的检测反而添乱。`share` 排不上时
等 180 秒（`SRUN_WAIT=` 可改）就退回登录节点跑，不会把人吊着。

### web-demo：clone 下来就能跑

前端 `web/dist/`（108 K）、**后端二进制** `server/target/release/wave-demo`（4.5 M，
git 里约 1.6 M）、以及 demo 的默认权重（`results/web/model/`，`best.pt` 8.6 M）
**都在版本库里**，所以不用编、不用配：

```bash
./code/web-demo/start.sh
```

`setup.sh` 不碰它 —— 既不核对也不编译。要改要编，完整说明（含必须先前端后后端、
必须丢计算节点、两条 `srun` 命令）在 [code/web-demo/README.md](code/web-demo/README.md)。
这里只留一条最容易悄悄坏掉的：**改了 `server/src/` 或 `web/src/` 要自己重编并把新产物
一起提交**，没人替你检查，源码和二进制会不声不响地对不上。

二进制之所以敢进版本库：源码没改时 `cargo build --release` 是**逐字节可复现**的
（实测两次 md5 相同），不会产生假改动。前端同理 —— 2026-08-23 在 cn-e02 上重跑
`vite build`，产出与版本库里的逐字节相同。代价是二进制平台锁死：要 `GLIBC_2.28`、
Linux x86-64，只在这个集群上有意义。

---

## 依赖清单

全部在 **`requirements.txt`** 一个文件里，`setup.sh` 一次装完，**不写死任何包名**
（要加包改这个文件，别改脚本）。torch 系靠文件里的一行 `--extra-index-url` 指向
pytorch 的 cu130 源，其余走 PyPI。

PyPI 上同样叫 `2.11.0` 的 torch 是 cu12 轮子，装上去计算节点
`torch.cuda.is_available()` 是 `False`，所以 torch 必须从 pytorch 的源取。

**为什么写 `torch==2.11.0` 而不是 `torch==2.11.0+cu130`**（2026-08-20 用
`pip install --dry-run` 实测过）：

- 写 `torch==2.11.0`：两个源都有候选，但 PEP 440 规定本地版本号排序高于同基版本，
  `2.11.0+cu130` > `2.11.0`，pip 必取 cu130 那个。实测结果与旧的 `--index-url`
  写法完全一致 —— 是规范保证，不是巧合。
- 写 `torch==2.11.0+cu130`：**反而会坏事**。wheel 装完后 dist 元数据里的版本被剥成
  `2.11.0`（`+cu130` 只留在 `torch.__version__` 里），pip 每次都判定「没装」，
  于是每跑一次 `setup.sh` 就重下 2 GB。

`setup.sh` 另有一条检查比对 torch / torchvision 的构建标是否一致，混装了会警告。

> 顺带一提，cu130 源镜像了 117 个包，其中 `numpy` / `pillow` 也在清单里。这不构成
> 问题：`numpy==2.2.6` 在两个源上是同一份 wheel（sha256 实测相同）。

`polars` 用的是 `lts-cpu` 变体 —— `share` 分区上混着 ivybridge 老机器，普通轮子的
avx512 指令会 illegal instruction。

版本：Python 3.10.20 / PyTorch 2.11.0+cu130 / CUDA 13.0。

### ⚠️ 别往这个环境里 `conda install` 编译过的包，一律走 pip

集群是 el8，系统 libstdc++ 只到 `GLIBCXX_3.4.25`，比 conda-forge 编译产物要求的
`3.4.29` 旧。混进来会让 torch 和 numpy 的 C 扩展互相加载不上，而报出来的却是 torch 的
`NP_SUPPORTED_MODULES` 找不到，很有迷惑性。按 `requirements.txt` 走 PyPI 就不会碰到；
`setup.sh` 有一条运行时体检（`import torch, numpy.fft, torchvision`）盯着回归。

> 历史：numpy 曾是 conda-forge 构建，2026-08-19 换 PyPI 后根因消失，`LD_LIBRARY_PATH`
> 兜底 08-20 拆除。**同版本号、不同编译产物**对复现的影响见 [README.md](README.md) §6.7-2。

---

## activate.sh —— 脚本怎么找环境

**sbatch 脚本一律不写死环境路径**，统一 source 仓库根的 `activate.sh`：

```bash
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"      # 找 conda + 激活环境，之后可用 $REPO
```

`sbatch` 会把脚本拷到 spool，所以定位靠 `$SLURM_SUBMIT_DIR` 而不是 `$0`。

环境位置按这个顺序定位，**全程没有任何人的用户名**：

1. `$OCEAN_ENV`（显式指定）
2. `<repo>/.env.local` 里的 `OCEAN_ENV=`（`setup.sh` 生成，不进版本库）
3. `/nfs/hpc/share/$USER/.conda/envs/ocean`（默认）

### 提交前自查：别写死个人路径

各脚本统一 `source` 仓库根的 `activate.sh`，谁也不该再写死 `/nfs/hpc/share/<某人>/`
或 `/nfs/stak/users/<某人>/` —— 换个账号 clone 下来就跑不了。**改完脚本、提交之前**
在仓库根跑一下：

```bash
git ls-files -z '*.sh' '*.py' '*.md' '*.yaml' '*.rs' '*.toml' \
  | xargs -0 grep -nIE -- '/nfs/(hpc/share|stak/users)/[a-z][a-z0-9_-]*/' \
  | grep -vE '/nfs/stak/a1/rhel5apps|/nfs/hpc/share/coast-lab|\$USER|^legacy/'
```

无输出 = 干净。三个豁免不算写死个人路径：`rhel5apps` 是全校共享的 conda 安装、
`coast-lab` 是实验室共享的 FUNWAVE 数据（`gen_prior.sh` / `scan.sh` 指向它）、
`$USER` 是变量。`legacy/` 不再维护，不参与。

---

## 仓库外的东西

| | 怎么拿 |
|---|---|
| `data/`（49 G） | 在云端硬盘，`data_20260822.tar`。也能用 `data/3d/crop_fields.sh` + `code/gen_prior.sh` 重造 |
| `legacy/` 的产物 | 在云端硬盘，三个包（见下）。仓库里只留源码 |
| `results/train`、`results/vis` | 在云端硬盘，`results_20260822.tar`。仓库里只剩带 `.gitkeep` 的空目录 |
| `FUNWAVE-TVD/`（256 M） | 第三方求解器干净 clone，`./setup.sh` 缺了自动拉 |
| `$OCEAN_DATA` | POD/LSTM 线的 `ocean_project/`，有默认值 |
| `$OCEAN_CASE` | OpenFOAM 算例，有默认值 |

版本库本身很小（249 个文件），clone 下来是代码 + 一副带 `.gitkeep` 的空目录骨架，
每个 `.gitkeep` 写明这个目录的内容在哪个包里、怎么还原。

wandb 没登录**不影响训练**：`train.py` 开跑前自己检测，检测不过就在日志里写明原因、
本次不记录（不会停在交互提示上，把一个 GPU 作业白熬到超时）。
project：`hpm-wave`（HPM 两条线）、`tsolverpp`（3D Transolver++），用户 `cassan-osu`。
本地 `wandb/` 目录是可再生缓存 —— 所有 run 都已同步云端，删了不丢东西。

### 云端大文件（`archive/`）

历史的可视化产物和 checkpoint 近 1 G，二进制在 git 里不做 delta 压缩、每改一版就整份
再存一遍，所以放云端硬盘而不是版本库。清单在仓库根的 **`archives.tsv`**（6 列 TAB 分隔，
格式说明写在文件头部）：

| 包 | 解压到 | 大小 | 内容 |
|---|---|---|---|
| `data_20260822.tar` | `data/` | 48.6 G | 3d 34.5 G + fwv 8.5 G + 2d 5.6 G，12122 个文件 |
| `legacy_20260822.tar` | `legacy/`（**不含 hpm**） | 12.7 G | fno 12.6 G（`processed_data` + `outputs`）、`transolver++/results`、`tsolverpp/outputs` |
| `legacy_hpm_vis_20260820.tar` | `legacy/hpm/vis/` | 486 M | 334 个 mp4，46 组历史可视化 |
| `legacy_hpm_outputs_20260820.tar` | `legacy/hpm/outputs/` | 428 M | 46 个 run 的 `checkpoints/*.pt` + `.hydra/*.yaml` |
| `results_20260822.tar` | `results/` | 194 M | `train/` 87 M + `vis/` 56 M + web-demo 那两次跑的留档 |

还原逻辑只有 `restore.sh` 一份，`setup.sh` 第 5 步就是调它（`--check` 会透传）：

```bash
./setup.sh                      # 连环境一起装；包缺了算缺件，退出码非 0（各步照样查完）
./archive/restore.sh            # 只管包：全量扫描 -> 全了才解 -> 解完逐文件核 manifest
                                # 缺一件就列全问题 exit 1，一个字节都不解
./archive/restore.sh --check    # 只体检不解
./archive/restore.sh web        # 只解 web-demo 要的那个包（= data 包）
./archive/restore.sh data_      # 指名解手上有的（按包名子串挑，日期和 .tar 可不写）
```

**手上只有一部分包**时走最后那种 —— 不带参数是"全有才动手"，缺一个就什么都不解。

`restore.sh` 用 `tar --skip-old-files`，已存在的文件一律不覆盖 —— 有几个包跟版本库
是重叠的（比如 `data/fwv/TK94/input.txt`），不加这个开关一次还原就会把版本库的版本
静默盖掉。

行为（两个入口都一样，因为是同一份代码）：**探测路径已存在且非空就跳过**（`.gitkeep`
不算数；幂等，可以反复跑）；否则去 `archive/` 找包，**校验 md5 通过**才解压；`archive/`
里也没有就**报缺件**，并打印一条可以直接复制的下载命令（来自 `archives.tsv` 第 6 列，
支持 `gdrive:<FILE_ID>` 和 `https://` 直链；第 6 列留空时提示"找仓库主人要"）。

**不会自动联网下载** —— 近 1 G，什么时候拉由人决定。但**包不齐 `setup.sh` 就算失败**
（退出码非 0）：`data_*.tar` 是训练数据，没它什么都跑不了；其余几个虽然只影响翻历史，
也一并按缺件计，免得"装完了"和"装齐了"混为一谈。

每个包旁边还有一份 `.manifest`（逐文件 md5），拿到之后可以逐个核而不只是核整包：

```bash
cd <仓库根> && md5sum -c $OCEAN_ARCHIVE/legacy_hpm_vis_20260820.manifest
```

---

## SLURM 速查

| 分区 | 硬件 | 本项目用法 |
|---|---|---|
| `dgxh` | H100 80GB / H200 143GB | 所有 GPU 训练 |
| `ampere` | **A40 48G**（没有 A100） | 推理、出图、动画；不收纯 CPU 作业（QOS `MinTRES=gres/gpu=1`） |
| `eecs` / `share` | CPU（`eecs` 另有 RTX2080 11G） | 数据准备、POD、LSTM、prior 生成与标定 |

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,Elapsed,State,ExitCode,Reason,MaxRSS
tail -f code/logs/hpm_<jobid>.log
```

⚠️ `code/logs/` 必须在**提交前**就存在 —— `#SBATCH --output` 在脚本执行之前生效，
目录不在时 SLURM 会把日志整个丢掉，而作业状态照样是 `COMPLETED`（实测）。
`setup.sh` 会建好。

想知道现在投哪个分区最快，别只信 `sbatch --test-only`（它给的是最坏情况的优先级排队
模拟，不算 backfill，也不检查 pending 作业是否真能跑起来）。直接看：

```bash
sinfo -p <part> -N -o '%N|%t|%C|%G'        # 有没有 idle 节点
squeue -p <part> -t PD -o '%i|%u|%r|%b'    # PENDING 卡在什么原因上
```

有 IDLE 节点、PENDING 又全卡在 `Dependency`（尤其 `DependencyNeverSatisfied`，永远不会
跑）上 —— 直接投，会立刻起。
