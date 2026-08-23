#!/bin/bash
#SBATCH --job-name=data_ranges
#SBATCH --output=ranges_%j.log
#SBATCH --partition=share
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
 
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

# POD / LSTM 那条线的数据在仓库外（case 23 G、各 *_results 共几 G），没进版本库。
OCEAN_DATA="${OCEAN_DATA:-$HOME/hpc-share/ocean_project}"
 
cd "$OCEAN_DATA"
 
python check_data_ranges.py \
    --data_dir "$OCEAN_DATA/case/postProcessing/sample" \
    --plot