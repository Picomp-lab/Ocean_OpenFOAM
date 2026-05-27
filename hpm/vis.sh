#!/bin/bash
#SBATCH --job-name=vis
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=hpm_vis_%j.log

cd ~/hpc-share/models/hpm

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

# ============================================================
# Paths — update these after training finishes
# ============================================================
CONFIG="outputs/2026-05-25/21-33-06/.hydra/config.yaml"  # <-- fill in after training
CKPT="outputs/2026-05-25/21-33-06/checkpoints/best.pt"   # <-- fill in after training
DATA=~/hpc-share/models/data/3d/cropped_0.05

echo "========================================"
echo "HPM Inference & Visualization"
echo "========================================"
echo "Config: $CONFIG"
echo "Ckpt:   $CKPT"
echo "Data:   $DATA"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

FIELD=5
OUTPUT="vis_n"
mkdir -p "$OUTPUT"


# Chunk 6: training range (~30s)
python -u vis.py \
    --config_path "$CONFIG" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --chunk_id 6 \
    --n_frames 93 \
    --output "${OUTPUT}/compare_chunk6_${OUTPUT#*_}.mp4" \
    --field "$FIELD"

# Chunk 9: validation range (~45s)
python -u vis.py \
    --config_path "$CONFIG" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --chunk_id 9 \
    --n_frames 93 \
    --output "${OUTPUT}/compare_chunk9_${OUTPUT#*_}.mp4" \
    --field "$FIELD"

echo ""
echo "Done: $(date)"
