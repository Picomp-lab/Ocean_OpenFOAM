#!/bin/bash
#SBATCH --job-name=vis_align
#SBATCH --output=logs/vis_align_%j.log
#SBATCH --error=logs/vis_align_%j.err
#SBATCH --partition=eecs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# ============================================================
# 配准检查: prior 与 CFD GT 两行对比 (训练前, 无需 checkpoint)
#
# 每个 chunk 的 t-offset 默认从 fwv/toffset_scan/c{cid}.json 读 —— 先跑
# scan_toffset.sh。没有 scan 结果时沿用 prior 落盘的 k, 并 warn。
#
# 用法:
#   sbatch fwv/vis_align.sh                        chunk 8,9 × alpha,p_rgh
#   CHUNKS="1 6 9" sbatch fwv/vis_align.sh         指定 chunk
#   FIELDS="alpha Umag" sbatch fwv/vis_align.sh    指定场
#   KSWEEP="2 3 4 5" CHUNKS=9 sbatch fwv/vis_align.sh
#       ^ 同一 chunk 出多支不同 k 的视频, 目视复核 scan 的结论。
#         留空则每个 chunk 只出一支 (用 scan 的 best_k)。
# ============================================================
CHUNKS=${CHUNKS:-8 9}
FIELDS=${FIELDS:-alpha p_rgh}
KSWEEP=${KSWEEP:-}
DATA=${DATA:-../data/3d/cropped_0.05}
PRIOR=${PRIOR:-../data/3d/prior_t015}

cd ~/hpc-share/models/hpm
mkdir -p logs

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

export PYTHONPATH="$PWD:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "========================================"
echo "chunks : ${CHUNKS}"
echo "fields : ${FIELDS}"
echo "ksweep : ${KSWEEP:-<用 scan 的 best_k>}"
echo "node   : $(hostname)   $(date)"
echo "========================================"

for CHUNK in $CHUNKS; do
  for FIELD in $FIELDS; do
    if [ -z "$KSWEEP" ]; then
      echo ""; echo "=== chunk ${CHUNK} / ${FIELD} ==="
      python -u fwv/vis_align.py --chunk "$CHUNK" --field "$FIELD" \
          --data-dir "$DATA" --prior-dir "$PRIOR" \
          || echo "  [warn] 失败, 继续"
    else
      for K in $KSWEEP; do
        echo ""; echo "=== chunk ${CHUNK} / ${FIELD} / k=${K} ==="
        python -u fwv/vis_align.py --chunk "$CHUNK" --field "$FIELD" --k "$K" \
            --data-dir "$DATA" --prior-dir "$PRIOR" \
            || echo "  [warn] 失败, 继续"
      done
    fi
  done
done

echo ""
echo "Done: $(date)"
ls -la fwv/vis_align/ 2>/dev/null
