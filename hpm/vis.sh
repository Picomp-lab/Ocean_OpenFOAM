#!/bin/bash
#SBATCH --job-name=vis
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpm_vis_%j.log

cd ~/hpc-share/models/hpm

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

# ============================================================
# Paths — update these after training finishes
# ============================================================
TIMEPOINT="2026-06-23/00-02-27"
CONFIG="outputs/$TIMEPOINT/.hydra/config.yaml"
CKPT="outputs/$TIMEPOINT/checkpoints/best.pt"
DATA=~/hpc-share/models/data/3d/cropped_0.05
FEATURE="flux_u"

echo "========================================"
echo "HPM Inference & Visualization"
echo "========================================"
echo "Config: $CONFIG"
echo "Ckpt:   $CKPT"
echo "Data:   $DATA"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

# ---- 覆盖保护：FEATURE 目录已有内容则终止（FORCE=1 可强制覆盖）----
if [ -d "vis/${FEATURE}" ] && [ -n "$(ls -A "vis/${FEATURE}" 2>/dev/null)" ]; then
    if [ "$FORCE" != "1" ]; then
        echo "ERROR: vis/${FEATURE}/ 已有内容，继续将覆盖已有结果。"
        echo "  - 要覆盖: FORCE=1 sbatch vis.sh"
        echo "  - 或换名: 修改 FEATURE 变量"
        exit 1
    fi
    echo "WARN: FORCE=1，将覆盖 vis/${FEATURE}/ 已有内容"
fi

declare -A FIELDS=( [0]=alpha [1]=ux [2]=uy [3]=uz [4]=prgh [5]=nut )

# style=both -> 每个 field×chunk 产出 *_scatter.mp4 和 *_tri.mp4 两个视频
for FIELD in 0 4 5; do
    NAME=${FIELDS[$FIELD]}
    OUTPUT="vis/${FEATURE}/${NAME}"
    mkdir -p "$OUTPUT"

    echo "=== Field $FIELD ($NAME) ==="
    for CHUNK in 6 9; do
        python -u vis.py \
            --config_path "$CONFIG" \
            --checkpoint "$CKPT" \
            --data_dir "$DATA" \
            --chunk_id "$CHUNK" \
            --n_frames 93 \
            --style both \
            --output "${OUTPUT}/compare_chunk${CHUNK}_${NAME}.mp4" \
            --field "$FIELD"
    done
done

# ============================================================
# RGB velocity (αU/U space, tri+scatter)
# ============================================================
OUTPUT="vis/${FEATURE}/U"
mkdir -p "$OUTPUT"
for CHUNK in 6 9; do
    python -u vis_u.py \
        --config_path "$CONFIG" \
        --checkpoint "$CKPT" \
        --data_dir "$DATA" \
        --chunk_id "$CHUNK" \
        --n_frames 93 \
        --style both \
        --output "${OUTPUT}/compare_chunk${CHUNK}_u.mp4"
done

echo ""
echo "Done: $(date)"