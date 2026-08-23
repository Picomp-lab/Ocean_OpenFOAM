#!/bin/bash
#SBATCH --job-name=gt_anim
#SBATCH --partition=dgxh
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=gt_anim_%j.log

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/tsolverpp"


DATA=$REPO/data/3d/cropped_0.05

# Chunk 6: ~30s training range
python -u gt_animation.py \
    --data_dir "$DATA" \
    --chunk_id 6 \
    --start_frame 0 \
    --n_frames 100 \
    --output gt_chunk6_alpha.mp4