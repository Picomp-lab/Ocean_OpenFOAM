#!/bin/bash
#SBATCH --job-name=vis_fw_lt
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/vis_fw_lt_%j.log

# long-term rollout 可视化 (无 GT)。从 hpm/ 运行。
#   显式给 CKPT/CONFIG:
#     CKPT=... CONFIG=... FEATURE=hpm_fw_ss_R4 sbatch fwv/vis_fw_lt.sh
#   跑满 prior 全长 (1000 帧), 只 tri, 单 alpha, 边推边渲染 (不存 pred)。
# mem=64G: prior 11.5GB 常驻 + 渲染缓冲; 边推边渲染故 pred 不累积。

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/hpm"
export PYTHONPATH="$PWD:$PYTHONPATH"


# CKPT/CONFIG: 显式优先; 否则按 TIMEPOINT 在 outputs/ 下找 checkpoints/best.pt
#   TIMEPOINT=hpm_fw_ss_R4 sbatch fwv/vis_fw_lt.sh   (自动找最新 best.pt + 同目录 .hydra)
#   或显式:  CKPT=... CONFIG=... sbatch fwv/vis_fw_lt.sh
TIMEPOINT="${TIMEPOINT:-hpm_fw_ss_R4}"

if [ -z "$CKPT" ] || [ -z "$CONFIG" ]; then
    echo "[vis_fw_lt] 未显式给 CKPT/CONFIG, 按 TIMEPOINT='$TIMEPOINT' 解析"
    SEARCH_ROOT="outputs/${TIMEPOINT}"
    [ -z "$TIMEPOINT" ] && SEARCH_ROOT="outputs"
    LATEST=$(find "$SEARCH_ROOT" -path "*/checkpoints/best.pt" -printf '%T@ %p\n' \
             2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$LATEST" ]; then
        echo "ERROR: $SEARCH_ROOT 下找不到 checkpoints/best.pt"
        echo "       或显式给 CKPT=... CONFIG=..."
        exit 1
    fi
    RUN_DIR=$(dirname "$(dirname "$LATEST")")     # .../<timestamp>
    CKPT="$LATEST"
    CONFIG="${RUN_DIR}/.hydra/config.yaml"
    echo "[vis_fw_lt] 解析到:"
    echo "  CKPT   = $CKPT"
    echo "  CONFIG = $CONFIG"
fi
if [ ! -f "$CONFIG" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: config 或 checkpoint 不存在:"
    echo "  CONFIG=$CONFIG"; echo "  CKPT=$CKPT"
    exit 1
fi

DATA=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').data.dir)")
PRIOR=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').data.prior_dir)")

CHUNK="${CHUNK:-10}"           # long-term chunk (只需 prior)
NFRAMES="${NFRAMES:-0}"        # 0 = 跑满 prior 全长
FIELD="${FIELD:-alpha}"
FEATURE="${FEATURE:-hpm_fw_ss_lt}"

echo "========================================"
echo "long-term rollout viz (no GT)"
echo "  CONFIG : $CONFIG"
echo "  CKPT   : $CKPT"
echo "  DATA   : $DATA"
echo "  PRIOR  : $PRIOR"
echo "  chunk=$CHUNK n_frames=$NFRAMES(0=full) field=$FIELD"
echo "  out    : fwv/vis/${FEATURE}/${FIELD}/"
echo "  node   : $(hostname)   date: $(date)"
echo "========================================"

OUT="fwv/vis/${FEATURE}/${FIELD}"
mkdir -p "$OUT"

# 覆盖保护
if [ -f "${OUT}/longterm_chunk${CHUNK}_${FIELD}_tri.mp4" ] && [ "$FORCE" != "1" ]; then
    echo "ERROR: ${OUT}/longterm_chunk${CHUNK}_${FIELD}_tri.mp4 已存在。FORCE=1 覆盖。"
    exit 1
fi

python -u fwv/vis_fw_lt.py \
    --config_path "$CONFIG" \
    --checkpoint "$CKPT" \
    --data_dir "$DATA" \
    --prior_dir "$PRIOR" \
    --chunk_id "$CHUNK" \
    --n_frames "$NFRAMES" \
    --field "$FIELD" \
    --output "${OUT}/longterm_chunk${CHUNK}_${FIELD}.mp4"

echo "Done: $(date)"
