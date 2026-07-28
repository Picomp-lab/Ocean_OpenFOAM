#!/bin/bash
#SBATCH --job-name=scan_toff
#SBATCH --output=logs/scan_toff_%j.log
#SBATCH --error=logs/scan_toff_%j.err
#SBATCH --partition=eecs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

# ============================================================
# 逐 chunk 标定 t-offset (串行, 单进程, 末尾自带汇总)
#
# 目的: 检验"跨 chunk 单一 t-offset"这个假设。此前只在 chunk 2 / 6 上测过
#       (均为 k=+3, 相隔 20s 无漂移), 这里把 1-9 全测一遍。
#
# 读法 (脚本末尾自动打印):
#   全部一致      -> 假设成立, 什么都不用改
#   单调递增/递减 -> 相速度系统偏差, 逐 chunk 用各自 k 重生成 prior
#   个别偏离      -> 那个 chunk 本身有问题, 不是全局漂移
#
# 为什么不用 array: 静态量 (coords / (x,y) 去重 / Bilinear 权重 / 水深) 与
#   chunk 无关。array 下每个 task 都要重算一遍, 且汇总还得单开一步。串行
#   共享这些量, 一条命令出结论。
#
# 用法:
#   sbatch fwv/scan_toffset.sh                    chunk 1-9
#   CHUNKS=9 sbatch fwv/scan_toffset.sh           只跑 chunk 9
#   KRANGE="-40 40" sbatch fwv/scan_toffset.sh    扩大扫描范围
#   python fwv/scan_toffset.py --summary          只重新汇总已有结果 (秒出)
#
# 注: chunk 0 静水 (OOD), chunk 10 GT 损坏 (只有 1 帧) —— 均不扫。
# ============================================================
CHUNKS=${CHUNKS:-1-9}
XOFF=${XOFF:-15.05}
# 已知答案在 +3 附近, ±10 帧 (±0.5s) 足够覆盖漂移; 扫得越宽越慢越吃内存。
# 若日志出现 "最优 k 落在扫描边界", 用 KRANGE 扩大重跑。
KRANGE=${KRANGE:--10 10}
# 破碎区 prior 与 GT 本就该不同, 纳入只给扫描加噪声
XWIN=${XWIN:--2.5 7.0}
FW=${FW:-/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output}
DATA=${DATA:-../data/3d/cropped_0.05}
# OUT=${OUT:-fwv/toffset_scan}

cd ~/hpc-share/models/hpm
mkdir -p logs

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

export PYTHONPATH="$PWD:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "========================================"
echo "chunks   : ${CHUNKS}"
echo "x-offset : ${XOFF}   (给定, 不扫)"
echo "k-range  : ${KRANGE} (帧)"
echo "x-win    : ${XWIN}"
echo "out      : ${OUT}/c*/scan.json"
echo "node     : $(hostname)   $(date)"
echo "========================================"

python -u fwv/scan_toffset.py \
    --fw-dir "${FW}" \
    --data-dir "${DATA}" \
    --chunks "${CHUNKS}" \
    --x-offset "${XOFF}" \
    --k-range ${KRANGE} \
    --x-win ${XWIN} \
    ${OUT:+--out "$OUT"}

RC=$?
echo "exit=${RC}  $(date)"
exit $RC