#!/bin/bash
#SBATCH --job-name=tsolver_pp
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=tsolver_pp_%j.log

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

cd ~/hpc-share/models/transolver++

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Default run
python -u train.py

# Override examples:
# python -u train.py model.n_hidden=256 model.n_layers=6 training.lr=5e-4
# python -u train.py training.batch_size=16 training.epochs=500
# python -u train.py data.train_end=40.0 data.test_start=40.0

# Sweep example:
# python -u train.py --multirun model.n_hidden=128,256 model.n_layers=4,6
