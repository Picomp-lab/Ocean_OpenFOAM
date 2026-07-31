#!/bin/bash
#SBATCH --job-name=basis
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cmp_basis_%j.log

# 跨 checkpoint 比对 spectral_basis 指纹 (纯 CPU, 不用 GPU)。
#   sbatch fwv/cmp_basis.sh
# 逐个 mmap 读 outputs/**/{best,latest}.pt, 算完即释放; 峰值 ~7GB/文件。

cd ~/hpc-share/models/hpm
export PYTHONPATH="$PWD:$PYTHONPATH"

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

echo "========================================"
echo "cmp_basis — spectral_basis 指纹比对"
echo "Node: $(hostname)   Date: $(date)"
echo "========================================"

python -u fwv/cmp_basis.py

echo ""
echo "Done: $(date)"
