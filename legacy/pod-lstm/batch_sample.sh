#!/bin/bash

# POD / LSTM 那条线的数据在仓库外（case 23 G、各 *_results 共几 G），没进版本库。
OCEAN_DATA="${OCEAN_DATA:-$HOME/hpc-share/ocean_project}"
#SBATCH --job-name=wave_sample
#SBATCH --output=sample_%j.log
#SBATCH --partition=share
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1

module load openfoam/2412

cd "$OCEAN_DATA/case"
postProcess -func sample
