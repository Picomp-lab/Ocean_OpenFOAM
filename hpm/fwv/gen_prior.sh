#!/bin/bash
#SBATCH --job-name=gen_prior
#SBATCH --output=logs/gen_prior_%A_%a.log
#SBATCH --error=logs/gen_prior_%A_%a.err
#SBATCH --partition=eecs
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00

# ============================================================
# 生成 FUNWAVE prior (投射到 CFD 不规则网格)
#
# 用法:
#   sbatch gen_prior.sh                    chunk 1..10 并行
#   sbatch --array=6 gen_prior.sh          只跑 chunk 6
#   sbatch --array=1-10%3 gen_prior.sh     限制同时最多 3 个 (省配额)
#   TOFF=0.0 sbatch gen_prior.sh           改 t-offset (默认 0.15)
#
# 参数来源:
#   x-offset 15.05  教授域图 + Xc_WK(6.35) <-> CFD startX(-8.7) 双重确认
#   t-offset 0.15   scan_toffset.py 在 chunk 2 / chunk 6 上独立标定, 均为 k=+3
#                   (相隔 20s 无漂移; 相对时钟误差上界 0.25%)
#   chunk 0 不生成: 静水, 已标为 OOD 排除训练
# ============================================================
CHUNK=${SLURM_ARRAY_TASK_ID}
TOFF=${TOFF:-0.15}
XOFF=${XOFF:-15.05}
FW=${FW:-/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output}
DATA=${DATA:-../data/3d/cropped_0.05}
OUT=${OUT:-../data/3d/prior_t015}

cd ~/hpc-share/models/hpm
mkdir -p logs

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

export PYTHONPATH="$PWD:$PYTHONPATH"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

CID=$(printf "%03d" "${CHUNK}")

echo "========================================"
echo "chunk    : ${CHUNK}"
echo "x-offset : ${XOFF}"
echo "t-offset : ${TOFF}"
echo "out      : ${OUT}"
echo "node     : $(hostname)   $(date)"
echo "========================================"

python -u fwv/gen_prior.py \
    --fw-dir "${FW}" \
    --coords "${DATA}/coords.npy" \
    --gt-times "${DATA}/chunk_${CID}_times.npy" \
    --chunk "${CHUNK}" \
    --x-offset "${XOFF}" \
    --t-offset "${TOFF}" \
    --out "${OUT}"

RC=$?
# prior_meta.json 会被各 chunk 互相覆盖 -> 存一份带 chunk 号的
if [ $RC -eq 0 ] && [ -f "${OUT}/prior_meta.json" ]; then
    cp "${OUT}/prior_meta.json" "${OUT}/prior_meta_c${CID}.json"
fi

echo "exit=${RC}  $(date)"
exit $RC
