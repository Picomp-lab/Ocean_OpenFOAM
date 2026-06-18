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
CONFIG="outputs/2026-06-04/22-27-15/.hydra/config.yaml"  # <-- fill in after training
CKPT="outputs/2026-06-04/22-27-15/checkpoints/best.pt"   # <-- fill in after training
DATA=~/hpc-share/models/data/3d/cropped_0.05
FEATURE="a02k20"

echo "========================================"
echo "HPM Inference & Visualization"
echo "========================================"
echo "Config: $CONFIG"
echo "Ckpt:   $CKPT"
echo "Data:   $DATA"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

declare -A FIELDS=( [0]=alpha [1]=ux [2]=uy [3]=uz [4]=prgh [5]=nut )

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
            --output "${OUTPUT}/compare_chunk${CHUNK}_${NAME}.mp4" \
            --field "$FIELD"
    done
done

OUTPUT="vis/${FEATURE}/U"
mkdir -p "$OUTPUT"
for CHUNK in 6 9; do
    python -u vis_u.py \
        --config_path "$CONFIG" \
        --checkpoint "$CKPT" \
        --data_dir "$DATA" \
        --chunk_id "$CHUNK" \
        --n_frames 93 \
        --output "${OUTPUT}/compare_chunk${CHUNK}_u.mp4"

echo ""
echo "Done: $(date)"