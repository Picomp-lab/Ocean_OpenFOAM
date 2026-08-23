#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --partition=eecs
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=prep_data_%j.log
 
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
 
cd "$REPO/legacy/transolver++"
 
python prepare_data.py \
    --raw_dir ~/hpc-share/ocean_project/case/postProcessing/sample \
    --output $REPO/data \
    --t_start 0.0 \
    --t_end 50.0
