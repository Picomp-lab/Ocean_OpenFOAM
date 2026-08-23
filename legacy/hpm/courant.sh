#!/bin/bash
#SBATCH --job-name=courant_diag
#SBATCH --output=logs/courant_%j.out
#SBATCH --error=logs/courant_%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=eecs

set -euo pipefail
echo "host=$(hostname)  start=$(date)"

# --- conda ---
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

# --- 线程: 脚本内也设了, 这里再兜一层与 --cpus-per-task 一致 ---
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

cd "$REPO/legacy/hpm"
python -u courant_diagnostic.py

echo "end=$(date)"
