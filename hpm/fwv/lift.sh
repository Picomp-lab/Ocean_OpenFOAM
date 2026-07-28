#!/bin/bash
#SBATCH --job-name=lift
#SBATCH --output=/nfs/stak/users/%u/hpc-share/models/hpm/logs/lift_%j.log
#SBATCH --error=/nfs/stak/users/%u/hpc-share/models/hpm/logs/lift_%j.err
#SBATCH --partition=eecs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# 用法: sbatch lift.sh <chunk>          (默认 chunk 6)
#       for c in 1 2 3; do sbatch lift.sh $c; done
CHUNK=${1:-6}

cd ~/hpc-share/models/hpm
source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

export PYTHONPATH="$PWD:$PYTHONPATH"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python fwv/gen_lift.py \
    --fw-dir /nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling/output \
    --chunk "${CHUNK}" \
    --out ../data/3d/lift
