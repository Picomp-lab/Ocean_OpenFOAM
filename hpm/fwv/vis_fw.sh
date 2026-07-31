#!/bin/bash
#SBATCH --job-name=vis_fw
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vis_fw_%j.log

# 从 hpm/ 运行:  sbatch fwv/vis_fw.sh
# fwv/ 必须能 shadow-free import 父模块 -> PYTHONPATH 前置 $PWD(=hpm/)
cd ~/hpc-share/models/hpm
export PYTHONPATH="$PWD:$PYTHONPATH"

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

# ============================================================
# ⚠️ 路径确认: fw 的 best.pt / config 快照未必在 outputs/ 下
#   (train_fw 存 ckpt 用 cfg.save.dir, config 走 hydra run dir)。
# 首选: 直接给 CKPT 和 CONFIG 环境变量, 精确、无歧义:
#   CKPT=path/best.pt CONFIG=path/config.yaml sbatch fwv/vis_fw.sh
# 兜底: 只给 TIMEPOINT, 按旧 outputs/ 布局 + best.pt mtime 解析 (可能找不到)。
# ============================================================
TIMEPOINT="${TIMEPOINT:-hpm_fw_fb_arm1_h128}"

if [ -z "$CKPT" ] || [ -z "$CONFIG" ]; then
    echo "[vis_fw.sh] 未显式给 CKPT/CONFIG, 按 outputs/ 布局解析 TIMEPOINT='$TIMEPOINT'"
    if [ ! -f "outputs/$TIMEPOINT/.hydra/config.yaml" ]; then
        SEARCH_ROOT="outputs/${TIMEPOINT}"
        [ -z "$TIMEPOINT" ] && SEARCH_ROOT="outputs"
        LATEST=$(find "$SEARCH_ROOT" -path "*/checkpoints/best.pt" -printf '%T@ %p\n' \
                 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -z "$LATEST" ]; then
            echo "ERROR: $SEARCH_ROOT 下找不到 checkpoints/best.pt。"
            echo "       fw 的 ckpt 可能不在 outputs/ —— 请直接给 CKPT=... CONFIG=..."
            exit 1
        fi
        RESOLVED="${LATEST%/checkpoints/best.pt}"; RESOLVED="${RESOLVED#outputs/}"
        echo "[vis_fw.sh] 解析为: $RESOLVED"
        TIMEPOINT="$RESOLVED"
    fi
    CONFIG="outputs/$TIMEPOINT/.hydra/config.yaml"
    CKPT="outputs/$TIMEPOINT/checkpoints/best.pt"
fi

if [ ! -f "$CONFIG" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: config 或 checkpoint 不存在:"
    echo "  CONFIG=$CONFIG"
    echo "  CKPT=$CKPT"
    exit 1
fi

# data_dir / prior_dir 从训练快照 config 读 (与训练严格一致)
DATA=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').data.dir)")
PRIOR=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').data.prior_dir)")

# ---- 链路验证参数 (先小跑) ----
MODE="${MODE:-rollout}"
CHUNK="${CHUNK:-9}"
NFRAMES="${NFRAMES:-20}"
FIELDS="${FIELDS:-alpha}"
FEATURE="${FEATURE:-$(echo "$TIMEPOINT" | tr '/' '_')}"

echo "========================================"
echo "fw rollout vis"
echo "  CONFIG : $CONFIG"
echo "  CKPT   : $CKPT"
echo "  DATA   : $DATA"
echo "  PRIOR  : $PRIOR"
echo "  mode=$MODE chunk=$CHUNK n_frames=$NFRAMES fields='$FIELDS'"
echo "  out    : fwv/vis/${FEATURE}/"
echo "  node   : $(hostname)   date: $(date)"
echo "========================================"

# 覆盖保护
if [ -d "fwv/vis/${FEATURE}" ] && [ -n "$(ls -A "fwv/vis/${FEATURE}" 2>/dev/null)" ]; then
    if [ "$FORCE" != "1" ]; then
        echo "ERROR: fwv/vis/${FEATURE}/ 已有内容。FORCE=1 覆盖, 或换 FEATURE=xxx"
        exit 1
    fi
    echo "WARN: FORCE=1, 覆盖 fwv/vis/${FEATURE}/"
fi

for FIELD in $FIELDS; do
    OUT="fwv/vis/${FEATURE}/${FIELD}"
    mkdir -p "$OUT"
    echo "=== field $FIELD  chunk $CHUNK ==="
    python -u fwv/vis_fw.py \
        --config_path "$CONFIG" \
        --checkpoint "$CKPT" \
        --data_dir "$DATA" \
        --prior_dir "$PRIOR" \
        --chunk_id "$CHUNK" \
        --start_frame 0 \
        --n_frames "$NFRAMES" \
        --mode "$MODE" \
        --style both \
        --field "$FIELD" \
        --output "${OUT}/compare_chunk${CHUNK}_${FIELD}.mp4"
done

echo ""
echo "Done: $(date)"
