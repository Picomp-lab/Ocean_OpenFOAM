#!/bin/bash
#SBATCH --job-name=tsolver3d
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=6:00:00
#SBATCH --output=logs/tsolverpp_%j.log
#SBATCH --nodelist=dgxh-4


source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Node: $(hostname) | Partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Python: $(python --version 2>&1) | PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "========================================="

cd ~/hpc-share/models/tsolverpp

# torchrun --nproc_per_node=2 train.py
python -u train.py
