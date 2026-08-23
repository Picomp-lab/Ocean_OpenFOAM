#!/bin/bash

# POD / LSTM 那条线的数据在仓库外（case 23 G、各 *_results 共几 G），没进版本库。
OCEAN_DATA="${OCEAN_DATA:-$HOME/hpc-share/ocean_project}"
#SBATCH --job-name=pod_wave
#SBATCH --output=pod_%j.log
#SBATCH --partition=share
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
 
cd "$OCEAN_DATA"
 
python pod_decomposition.py \
    --data_dir "$OCEAN_DATA/case/postProcessing/sample" \
    --output "$OCEAN_DATA/pod_results" \
    --n_components 300 \
    --n_workers 8 \
    --max_steps 1000