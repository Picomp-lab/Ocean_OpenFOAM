#!/bin/bash
#SBATCH --job-name=hpmt
#SBATCH --partition=dgxh
#SBATCH --exclude=dgxh-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# 相对提交目录 —— 从 legacy/hpm/ 里提交，先 mkdir -p logs
#SBATCH --output=logs/hpm_train_%j.log

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/hpm"


# ============================================================
# Environment info
# ============================================================
echo "========================================"
echo "HPM Training"
echo "========================================"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "GPU Mem:  $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:   $(python --version)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:     $(python -c 'import torch; print(torch.version.cuda)')"
echo "Date:     $(date)"
echo "Overrides: ${@:-<none (baseline)>}"
echo "========================================"

# ============================================================
# Train — all config from config.yaml
# Override via command line: python train.py model.n_hidden=128
# ============================================================
python -u train.py "$@"