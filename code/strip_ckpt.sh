#!/bin/bash
#SBATCH --job-name=strip_ckpt
#SBATCH --partition=share
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=logs/strip_ckpt_%j.log
#
# ══════════════════════════════════════════════════════════════════════════
#  strip_ckpt.sh — 把 hpm/outputs 里的 legacy LBO basis 副本剥干净
#  用法: 从 code/ 提交 ->  mkdir -p logs && sbatch strip_ckpt.sh
#
#  为什么要上计算节点: 登录节点 ulimit -v 硬限 15 GB, 那 4 个 13.2 GiB 的 ckpt
#  (2026-06-05/11-51-47 与 2026-06-18/21-54-18 的 best+latest) 连 mmap 都开不了。
#  这里要 48G 内存、不要 GPU (纯 CPU 的搬运活)。
#
#  先干跑看清单 (登录节点就能跑, 不读张量数据):
#      python strip_ckpt.py
#  只处理小文件、不排队:
#      python strip_ckpt.py --apply --max-gb 10
#
#  参数透传: sbatch strip_ckpt.sh <目录或文件> ...   (默认 ../hpm/outputs)
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

if [ "$(basename "${SLURM_SUBMIT_DIR:-$PWD}")" != "code" ] || [ ! -f "strip_ckpt.py" ]; then
    echo "ERROR: 请从 code/ 目录提交 (sbatch strip_ckpt.sh)"
    echo "       当前提交目录: ${SLURM_SUBMIT_DIR:-<非 SLURM 环境>}   cwd: $PWD"
    exit 1
fi

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

export PYTHONUNBUFFERED=1

echo "=== 剥离前 ==="
du -sh ../hpm/outputs

python strip_ckpt.py --apply "$@"

echo "=== 剥离后 ==="
du -sh ../hpm/outputs
echo "Done: $(date)"
