#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=inference_%j.log

cd ~/hpc-share/models/tsolverpp

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

CONFIG=~/hpc-share/models/tsolverpp/outputs/2026-05-19/03-21-52/.hydra/config.yaml
CKPT=~/hpc-share/models/tsolverpp/outputs/2026-05-19/03-21-52/checkpoints/best.pt
DATA=~/hpc-share/models/data/3d/cropped_0.05

# Chunk 6: ~30s, training range
python -u inference.py \
    --config_path "$CONFIG" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --chunk_id 6 \
    --rollout_steps 93 \
    --output rollout_chunk6_alpha.mp4 \
    --field 0

# Chunk 9: ~45s, validation range
python -u inference.py \
    --config_path "$CONFIG" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --chunk_id 9 \
    --rollout_steps 93 \
    --output rollout_chunk9_alpha.mp4 \
    --field 0
