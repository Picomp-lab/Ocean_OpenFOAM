#!/bin/bash
#SBATCH --job-name=compare_anim
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=compare_anim_%j.log

cd "$REPO/tsolver3d"

CONFIG=$REPO/legacy/tsolverpp/outputs/2026-05-19/03-21-52/.hydra/config.yaml
CKPT=$REPO/legacy/tsolverpp/outputs/2026-05-19/03-21-52/checkpoints/best.pt
DATA=$REPO/data/3d/cropped_0.05

# Chunk 6: training range ~30s
python -u vis.py \
    --config_path "$CONFIG" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --chunk_id 6 \
    --n_frames 93 \
    --output compare_chunk6_alpha.mp4

# Chunk 9: validation range ~45s
python -u vis.py \
    --config_path "$CONFIG" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --chunk_id 9 \
    --n_frames 93 \
    --output compare_chunk9_alpha.mp4