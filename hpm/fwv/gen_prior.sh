#!/bin/bash
#SBATCH --job-name=gen_prior
#SBATCH --output=logs/gen_prior_%A_%a.log
#SBATCH --error=logs/gen_prior_%A_%a.err
#SBATCH --partition=preempt
#SBATCH --array=1-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00

# ============================================================
# 生成 FUNWAVE prior (投射到 CFD 不规则网格)
#
# 用法:
#   sbatch fwv/gen_prior.sh                    chunk 1..9 并行
#   sbatch --array=9 fwv/gen_prior.sh          只跑 chunk 9
#   sbatch --array=1-9%3 fwv/gen_prior.sh      限制同时最多 3 个 (省配额)
#   TOFF=0.15 sbatch fwv/gen_prior.sh          强制统一 t-offset (跳过逐 chunk 查表)
#
# 参数来源:
#   x-offset 15.05  教授域图 + Xc_WK(6.35) <-> CFD startX(-8.7) 双重确认
#   t-offset        **逐 chunk**, 从 fwv/toffset_scan/c{CID}.json 的 best_k 读。
#                   先跑 scan_toffset.sh。TOFF 环境变量可强制统一值。
#   chunk 0 不生成: 静水, 已标为 OOD 排除训练
#   chunk 10 不生成: GT 损坏 (只有 1 帧)
#
# 注: meta 现在由 gen_prior.py 直接写 prior_{cid}_meta.json (逐 chunk),
#     不再有"共用 prior_meta.json 再 cp"那个 array 竞态。
# ============================================================
CHUNK=${SLURM_ARRAY_TASK_ID}
XOFF=${XOFF:-15.05}
SCAN=${SCAN:-fwv/toffset_scan}
FW=${FW:-/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output}
DATA=${DATA:-../data/3d/cropped_0.05}
OUT=${OUT:-${DATA}/prior_ktuned}

cd ~/hpc-share/models/hpm
mkdir -p logs

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

export PYTHONPATH="$PWD:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

CID=$(printf "%03d" "${CHUNK}")

# ---- t-offset: 逐 chunk 查表 (TOFF 未显式给定时) ----
# fail loud: scan 结果缺失或 best_k 为 null 就退出, 不静默回落到某个默认值 ——
# 用错 offset 生成的 prior 看不出问题, 但整条训练都会被污染。
if [ -z "${TOFF}" ]; then
    SCAN_JSON="${SCAN}/c${CID}.json"
    if [ ! -f "${SCAN_JSON}" ]; then
        echo "ERROR: 找不到 ${SCAN_JSON}"
        echo "       先跑 sbatch fwv/scan_toffset.sh, 或显式给 TOFF=0.15"
        exit 1
    fi
    TOFF=$(python - "${SCAN_JSON}" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("t_offset")
if t is None:
    sys.exit("null")
print(t)
PYEOF
)
    if [ -z "${TOFF}" ]; then
        echo "ERROR: ${SCAN_JSON} 里 t_offset 为 null (该 chunk 标定失败)"
        exit 1
    fi
    BK=$(python -c "import json;print(json.load(open('${SCAN_JSON}'))['best_k'])")
    echo "[toff] chunk ${CHUNK}: 从 ${SCAN_JSON} 读到 best_k=${BK} -> t-offset=${TOFF}"
else
    echo "[toff] chunk ${CHUNK}: 使用显式 TOFF=${TOFF} (跳过查表)"
fi

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
echo "exit=${RC}  $(date)"
exit $RC