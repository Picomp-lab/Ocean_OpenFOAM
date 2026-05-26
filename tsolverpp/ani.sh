#!/bin/bash
#SBATCH --job-name=gt_anim
#SBATCH --partition=dgxh
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gt_anim_%j.log

cd ~/hpc-share/models/tsolverpp

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

DATA=~/hpc-share/models/data/3d/cropped_0.05

# Chunk 6: ~30s training range
python -u gt_animation.py \
    --data_dir "$DATA" \
    --chunk_id 6 \
    --start_frame 0 \
    --n_frames 100 \
    --output gt_chunk6_alpha.mp4