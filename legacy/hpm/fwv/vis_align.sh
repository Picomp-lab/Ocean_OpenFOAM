#!/bin/bash
#SBATCH --job-name=vis_align
#SBATCH --output=logs/vis_align_%j.log
#SBATCH --error=logs/vis_align_%j.err
#SBATCH --partition=eecs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ============================================================
# 配准检查: prior 与 CFD GT 两行对比 —— 在 gen_prior 之前跑
#
# prior 由本脚本现算 (只算 y=0.30 切片的那几万个 cell), 不依赖 gen_prior。
# k 默认从 fwv/toffset_scan/c{cid}.json 读 —— 先跑 scan_toffset.sh。
#
# 流程:  scan_toffset -> vis_align (本步, 目视确认 k) -> gen_prior -> train
#
# 用法:
#   sbatch fwv/vis_align.sh                        chunk 8,9 × alpha,p_rgh
#   CHUNKS="1 6 9" sbatch fwv/vis_align.sh         指定 chunk
#   FIELDS="alpha Umag" sbatch fwv/vis_align.sh    指定场
#   NFRAMES=0 sbatch fwv/vis_align.sh              全长 (默认只取中间 40 帧)
#   KSWEEP="2 3 4 5" CHUNKS=9 sbatch fwv/vis_align.sh
#       ^ 同一 chunk 出多支不同 k 的视频, 目视复核 scan 的结论。
#         留空则每个 chunk 只出一支 (用 scan 的 best_k)。
# ============================================================
CHUNKS=${CHUNKS:-1 2 3 4 5 6 7 8 9}
FIELDS=${FIELDS:-alpha}
# 配准看相位, chunk 中间 40 帧足够; 0 = 全长 (慢很多)
NFRAMES=${NFRAMES:-40}
KSWEEP=${KSWEEP:-}
DATA=${DATA:-../data/3d/cropped_0.05}
FW=${FW:-/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output}

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/hpm"
mkdir -p logs


export PYTHONPATH="$PWD:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "========================================"
echo "chunks : ${CHUNKS}"
echo "fields : ${FIELDS}"
echo "ksweep : ${KSWEEP:-<用 scan 的 best_k>}"
echo "frames : ${NFRAMES} (chunk 正中; 0=全长)"
echo "node   : $(hostname)   $(date)"
echo "========================================"

for CHUNK in $CHUNKS; do
  for FIELD in $FIELDS; do
    if [ -z "$KSWEEP" ]; then
      echo ""; echo "=== chunk ${CHUNK} / ${FIELD} ==="
      python -u fwv/vis_align.py --chunk "$CHUNK" --field "$FIELD" \
          --fw-dir "$FW" --data-dir "$DATA" --n-frames "$NFRAMES" \
          || echo "  [warn] 失败, 继续"
    else
      for K in $KSWEEP; do
        echo ""; echo "=== chunk ${CHUNK} / ${FIELD} / k=${K} ==="
        python -u fwv/vis_align.py --chunk "$CHUNK" --field "$FIELD" --k "$K" \
            --fw-dir "$FW" --data-dir "$DATA" --n-frames "$NFRAMES" \
            || echo "  [warn] 失败, 继续"
      done
    fi
  done
done

echo ""
echo "Done: $(date)"
ls -la fwv/vis_align/ 2>/dev/null