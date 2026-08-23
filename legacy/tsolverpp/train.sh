#!/bin/bash
#SBATCH --job-name=tsolver3d
#SBATCH --partition=dgxh
#SBATCH --nodelist=dgxh-3
#SBATCH --gres=gpu:1
# --constraint=h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=20:00:00
#SBATCH --output=logs/tsolverpp_%j.log


_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Node: $(hostname) | Partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Python: $(python --version 2>&1) | PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "========================================="

cd "$REPO/legacy/tsolverpp"



# torchrun --nproc_per_node=2 train.py
python -u train.py model.dropout=0.2 train.weight_decay=1e-3
