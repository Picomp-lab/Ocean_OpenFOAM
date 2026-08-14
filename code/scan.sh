#!/bin/bash
#SBATCH --job-name=scan_toff
#SBATCH --output=logs/scan_%j.log
#SBATCH --error=logs/scan_%j.err
#SBATCH --partition=eecs            # ← 确认: CPU 分区 (prior 标定是纯 numpy, 不吃 GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00             # ← 确认: 上界猜测, 按实测调 (扫 1-9 全量)

# ══════════════════════════════════════════════════════════════════════════
#  阶段一 / STAGE 1 — t-offset 标定 (per-chunk best_k)
#  用法: 从 code/ 提交 ->  mkdir -p logs && sbatch scan_toffset.sh
#
#  ── 依赖代码 (code deps, 均在 code/ 平铺) ─────────────────────────────────
#     scan_toffset.py   本阶段入口
#       └ import lift.py          Nwogu 剖面公式 (CH_NAMES 通道序唯一真源)
#       └ import fw_io.py         FUNWAVE 文件读取 (load_static)
#       └ from gen_prior import Bilinear, build_frame   投射层复用
#
#  ── 输入数据 (inputs) ────────────────────────────────────────────────────
#     $FW/{eta,u,v,mask}_NNNNN, dep.out         FUNWAVE 原生 2D 输出
#     $DATA/coords.npy                          CFD cell 中心坐标 (N,3)
#     $DATA/chunk_00X_{data,times}.npy  X=1..9  GT (标定比对用)
#
#  ── 输出 (outputs) ───────────────────────────────────────────────────────
#     $DATA/toffset_scan/c00X.json  X=1..9      best_k / t_offset + 逐通道曲线
#     (老的 toffset_scan 已改名腾位; 默认会新建干净目录)
# ══════════════════════════════════════════════════════════════════════════

# ---- 可改配置 (edit here) ----
FW=/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output   # ← 确认: 大小写! (曾踩 TINGKIRBY 坑)
# --data-dir / --out 用脚本默认 (已验证解析到 ../data/3d/cropped_0.05, 与平铺布局一致)

# ---- 环境 ----
source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh        # ← 确认: conda 激活路径最易过时
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4      # 与 cpus-per-task 一致

# ---- 运行 ----
python scan_toffset.py --fw-dir "$FW" --chunks 1-9

# 跑完看三个诊断闸门再进阶段二:
#   1) 末尾汇总表 best_k 有无 "落在扫描边界" -> 有则放宽 --k-range 重扫
#   2) Uy 行应显示 "[曲线平坦, 不计入]" (准二维正确信号)
#   3) 通道间分散 >2 帧 -> x-offset(15.05) 可能有残差
