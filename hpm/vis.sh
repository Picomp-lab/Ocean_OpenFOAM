#!/bin/bash
#SBATCH --job-name=vis
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/hpm_vis_%j.log

cd ~/hpc-share/models/hpm

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

# ============================================================
# TIMEPOINT 三种写法（唯一必改项）：
#   1. 完整路径:  "hpm_bl_h128/2026-07-09_10-23-45"        -> 精确指定
#      （含超参 arm: "hpm_bl_h128/train.lr-1e-3/时间戳"）
#   2. 只写运行名: "hpm_no-nut_h128"                        -> 自动选该名下
#      最新一次训练（按 best.pt 修改时间 = 最近训练活动）
#   3. 留空:      ""                                        -> 全局最新一次训练
# ============================================================
TIMEPOINT="${TIMEPOINT:-hpm_no-nut_h128}"

# 可选项：输出目录名。不改则默认从解析后的 TIMEPOINT 派生（唯一、不冲突）。
FEATURE="${FEATURE:-}"

# 想看的标量场候选列表（意图声明）。实际生成 = WANTED ∩ 该 checkpoint 的
# schema 通道（自动裁剪：E1 无 nut 的模型会自动跳过 nut，不报错）。
WANTED="alpha p_rgh nut"

# ---- TIMEPOINT 解析：非精确路径 -> 按 best.pt mtime 找最新 ----
if [ ! -f "outputs/$TIMEPOINT/.hydra/config.yaml" ]; then
    SEARCH_ROOT="outputs/${TIMEPOINT}"
    [ -z "$TIMEPOINT" ] && SEARCH_ROOT="outputs"
    LATEST=$(find "$SEARCH_ROOT" -path "*/checkpoints/best.pt" -printf '%T@ %p\n' 2>/dev/null \
             | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$LATEST" ]; then
        echo "ERROR: 在 $SEARCH_ROOT 下找不到任何含 checkpoints/best.pt 的训练目录"
        exit 1
    fi
    RESOLVED="${LATEST%/checkpoints/best.pt}"
    RESOLVED="${RESOLVED#outputs/}"
    N_RUNS=$(find "$SEARCH_ROOT" -path "*/checkpoints/best.pt" 2>/dev/null | wc -l)
    echo "[vis.sh] TIMEPOINT='$TIMEPOINT' 解析为最新训练: $RESOLVED  (候选 ${N_RUNS} 个)"
    TIMEPOINT="$RESOLVED"
fi

FEATURE="${FEATURE:-$(echo "$TIMEPOINT" | tr '/' '_')}"

# ---- 以下全部自动派生，无需修改 ----
CONFIG="outputs/$TIMEPOINT/.hydra/config.yaml"
CKPT="outputs/$TIMEPOINT/checkpoints/best.pt"

if [ ! -f "$CONFIG" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: config 或 checkpoint 不存在："
    echo "  $CONFIG"
    echo "  $CKPT"
    exit 1
fi

# 数据目录：从训练时的快照 config 读取（与训练严格一致）
DATA=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').data.dir)")

# 标量场列表：WANTED ∩ schema（schema 来自快照 config，含 legacy fallback）
# 注意：$(...) 捕获 stdout，故 from_cfg 必须 verbose=False（否则 schema 日志
# 会被当成字段名 → 曾产生 '[schema]' 'WARNING:' 等垃圾目录）；tail -1 兜底。
FIELDS=$(python - <<PYEOF
from omegaconf import OmegaConf
from schema import ChannelSchema
import sys
s = ChannelSchema.from_cfg(OmegaConf.load("$CONFIG"), verbose=False)
wanted = "$WANTED".split()
present = [f for f in wanted if f in s.names]
skipped = [f for f in wanted if f not in s.names]
if skipped:
    print(f"[vis.sh] skipped (not in schema): {skipped}", file=sys.stderr)
print(" ".join(present))
PYEOF
)
FIELDS=$(echo "$FIELDS" | tail -1)
if [ -z "$FIELDS" ]; then
    echo "WARN: WANTED ∩ schema 为空，跳过标量场视频（仅跑速度管线）"
fi

echo "========================================"
echo "HPM Inference & Visualization"
echo "========================================"
echo "Config: $CONFIG"
echo "Ckpt:   $CKPT"
echo "Data:   $DATA"
echo "Fields: $FIELDS   (wanted: $WANTED)"
echo "Output: vis/${FEATURE}/"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

# ---- 覆盖保护：FEATURE 目录已有内容则终止（FORCE=1 可强制覆盖）----
if [ -d "vis/${FEATURE}" ] && [ -n "$(ls -A "vis/${FEATURE}" 2>/dev/null)" ]; then
    if [ "$FORCE" != "1" ]; then
        echo "ERROR: vis/${FEATURE}/ 已有内容，继续将覆盖已有结果。"
        echo "  - 要覆盖: FORCE=1 sbatch vis.sh"
        echo "  - 或换名: FEATURE=xxx sbatch vis.sh"
        exit 1
    fi
    echo "WARN: FORCE=1，将覆盖 vis/${FEATURE}/ 已有内容"
fi

# ============================================================
# 标量场 — --field 按通道名（schema 驱动，自动裁剪后的列表）
# style=both -> 每个 field×chunk 产出 *_scatter.mp4 和 *_tri.mp4
# ============================================================
for FIELD in $FIELDS; do
    OUTPUT="vis/${FEATURE}/${FIELD}"
    mkdir -p "$OUTPUT"

    echo "=== Field ${FIELD} ==="
    for CHUNK in 6 9; do
        python -u vis.py \
            --config_path "$CONFIG" \
            --checkpoint "$CKPT" \
            --data_dir "$DATA" \
            --chunk_id "$CHUNK" \
            --n_frames 93 \
            --style both \
            --output "${OUTPUT}/compare_chunk${CHUNK}_${FIELD}.mp4" \
            --field "$FIELD"
    done
done

# ============================================================
# RGB velocity (αU/U space, tri+scatter)
# 需要 schema 含 Ux/Uy/Uz（vis_u.py 内部 assert，fail loud）
# ============================================================
OUTPUT="vis/${FEATURE}/U"
mkdir -p "$OUTPUT"
for CHUNK in 6 9; do
    python -u vis_u.py \
        --config_path "$CONFIG" \
        --checkpoint "$CKPT" \
        --data_dir "$DATA" \
        --chunk_id "$CHUNK" \
        --n_frames 93 \
        --style both \
        --output "${OUTPUT}/compare_chunk${CHUNK}_u.mp4"
done

echo ""
echo "Done: $(date)"