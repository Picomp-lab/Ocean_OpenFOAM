#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --partition=eecs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_data_%j.log

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

python -u prepare_data.py