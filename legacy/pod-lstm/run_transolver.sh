#!/bin/bash
#SBATCH --job-name=transolver
#SBATCH --partition=ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=transolver_%j.log

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

# POD / LSTM 那条线的数据在仓库外（case 23 G、各 *_results 共几 G），没进版本库。
OCEAN_DATA="${OCEAN_DATA:-$HOME/hpc-share/ocean_project}"

python train_transolver.py \
    --data_dir "$OCEAN_DATA/transolver_data" \
    --output   "$OCEAN_DATA/transolver_results" \
    --gpu 0 \
    --n_hidden 128 \
    --n_layers 4 \
    --n_heads 4 \
    --slice_num 64 \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1e-3 \
    --epochs 200 \
    --patience 30 \
    --ar_steps 50