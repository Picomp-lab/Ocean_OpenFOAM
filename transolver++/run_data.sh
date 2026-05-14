#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --partition=eecs
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=prep_data_%j.log
 
source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean
 
cd ~/hpc-share/models/transolver++
 
python prepare_data.py \
    --raw_dir ~/hpc-share/ocean_project/case/postProcessing/sample \
    --output ~/hpc-share/models/data \
    --t_start 0.0 \
    --t_end 50.0
