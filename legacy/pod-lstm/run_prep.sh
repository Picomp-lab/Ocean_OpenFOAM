#!/bin/bash
#SBATCH --job-name=wave_prep
#SBATCH --partition=eecs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=prep_%j.log

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

# POD / LSTM 那条线的数据在仓库外（case 23 G、各 *_results 共几 G），没进版本库。
OCEAN_DATA="${OCEAN_DATA:-$HOME/hpc-share/ocean_project}"

python prepare_data.py \
    --raw_dir "$OCEAN_DATA/case/postProcessing/sample" \
    --output  "$OCEAN_DATA/transolver_data"