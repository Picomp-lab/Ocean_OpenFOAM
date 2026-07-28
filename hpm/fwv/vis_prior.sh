#!/bin/bash
#SBATCH --job-name=vis1b
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vis_prior_%j.log

cd ~/hpc-share/models/hpm

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/baoh/.conda/envs/ocean

export PYTHONPATH="$PWD:$PYTHONPATH"

# ============================================================
# 1b (prior-only) 三行对比视频: prior / pred / GT
#
# TIMEPOINT 三种写法:
#   1. 完整路径:  "hpm_prior1b_h128/2026-07-23_21-30-00"   精确指定
#   2. 只写运行名: "hpm_prior1b_h128"                      该名下最新一次
#   3. 留空:      ""                                       全局最新一次
#
# 用法:
#   sbatch vis_prior.sh                     field/chunk 均从 config 推导
#   NFRAMES=10 sbatch vis_prior.sh          每支只出 10 帧 (快速验证)
#   CHUNKS="6 8" sbatch vis_prior.sh        覆盖 chunk (6 在训练集内, 看拟合上限)
#   WANTED="alpha Umag" sbatch vis_prior.sh 覆盖 field
#   FORCE=1 sbatch vis_prior.sh             覆盖已有输出
#
# 自动推导:
#   FIELDS = WANTED ∩ checkpoint 的 schema  (1b 无 Uy/nut, 自动剔除)
#   CHUNKS = config 的 val_chunk_range + test_chunk_range
# ============================================================
TIMEPOINT="${TIMEPOINT:-hpm_prior1b_h128}"

# 想看的场 (实际生成 = WANTED ∩ 该 checkpoint 的 schema)。
# 1b 无 Uy/nut; Umag 特判为 sqrt(Ux^2+Uz^2)。
WANTED="${WANTED:-alpha p_rgh Ux Uz}"
# CHUNKS 留空 -> 从 config 的 val_chunk_range + test_chunk_range 推导
CHUNKS="${CHUNKS:-}"
NFRAMES="${NFRAMES:-0}"          # 0 = 整个 chunk

# ---- TIMEPOINT 解析: 非精确路径 -> 按 best.pt mtime 找最新 ----
if [ ! -f "outputs/$TIMEPOINT/.hydra/config.yaml" ]; then
    SEARCH_ROOT="outputs/${TIMEPOINT}"
    [ -z "$TIMEPOINT" ] && SEARCH_ROOT="outputs"
    LATEST=$(find "$SEARCH_ROOT" -path "*/checkpoints/best.pt" -printf '%T@ %p\n' 2>/dev/null \
             | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$LATEST" ]; then
        echo "ERROR: 在 $SEARCH_ROOT 下找不到含 checkpoints/best.pt 的训练目录"
        exit 1
    fi
    RESOLVED="${LATEST%/checkpoints/best.pt}"
    TIMEPOINT="${RESOLVED#outputs/}"
    echo "[vis_prior.sh] 解析为最新训练: $TIMEPOINT"
fi

CONFIG="outputs/$TIMEPOINT/.hydra/config.yaml"
CKPT="outputs/$TIMEPOINT/checkpoints/best.pt"
for f in "$CONFIG" "$CKPT"; do
    [ -f "$f" ] || { echo "ERROR: 不存在 $f"; exit 1; }
done

FEATURE="${FEATURE:-$(echo "$TIMEPOINT" | tr '/' '_')}"
OUTDIR="vis_prior/${FEATURE}"

# ---- FIELDS / CHUNKS 从 config 推导 ----
# FIELDS = WANTED ∩ schema (Umag 特判放行)
# CHUNKS = val + test chunk range (未显式指定时)
# 注: $(...) 捕获 stdout, 故 from_cfg 必须 verbose=False (否则 schema 日志
#     会被当成字段名); 诊断信息一律走 stderr; tail 兜底取最后两行。
DERIVED=$(python - <<PYEOF
import sys
from omegaconf import OmegaConf
from schema import ChannelSchema
from dataset import expand_range

cfg = OmegaConf.load("$CONFIG")
s = ChannelSchema.from_cfg(cfg, verbose=False)

wanted = "$WANTED".split()
present = [f for f in wanted if f in s.names or f == "Umag"]
skipped = [f for f in wanted if f not in present]
if skipped:
    print(f"[derive] fields skipped (not in schema): {skipped}", file=sys.stderr)

chunks = []
for key in ("val_chunk_range", "test_chunk_range"):
    r = cfg.data.get(key)
    if r is not None:
        chunks += expand_range(r)
chunks = sorted(set(chunks))
print(f"[derive] chunks from config (val+test): {chunks}", file=sys.stderr)

print(" ".join(present))
print(" ".join(str(c) for c in chunks))
PYEOF
)
FIELDS=$(echo "$DERIVED" | tail -2 | head -1)
CHUNKS="${CHUNKS:-$(echo "$DERIVED" | tail -1)}"
[ -z "$FIELDS" ] && { echo "ERROR: WANTED ∩ schema 为空"; exit 1; }
[ -z "$CHUNKS" ] && { echo "ERROR: config 里没有 val/test chunk range"; exit 1; }

echo "========================================"
echo "1b visualization (prior / pred / GT)"
echo "========================================"
echo "Config:  $CONFIG"
echo "Ckpt:    $CKPT"
echo "Fields:  $FIELDS   (wanted: $WANTED)"
echo "Chunks:  $CHUNKS"
echo "Frames:  $([ "$NFRAMES" -eq 0 ] && echo 'all' || echo "$NFRAMES")"
echo "Output:  $OUTDIR/"
echo "Node:    $(hostname)   $(date)"
echo "========================================"

# ---- 覆盖保护 ----
if [ -d "$OUTDIR" ] && [ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ] && [ "$FORCE" != "1" ]; then
    echo "ERROR: $OUTDIR/ 已有内容。覆盖: FORCE=1 sbatch vis_prior.sh"
    echo "                            或换名: FEATURE=xxx sbatch vis_prior.sh"
    exit 1
fi
mkdir -p "$OUTDIR"

NF_ARG=""
[ "$NFRAMES" -gt 0 ] && NF_ARG="--n_frames $NFRAMES"

for FIELD in $FIELDS; do
    for CHUNK in $CHUNKS; do
        echo ""
        echo "=== chunk ${CHUNK} / ${FIELD} ==="
        python -u fwv/vis_prior.py \
            --config_path "$CONFIG" \
            --checkpoint "$CKPT" \
            --chunk_id "$CHUNK" \
            --field "$FIELD" \
            $NF_ARG \
            --output "${OUTDIR}/c${CHUNK}_${FIELD}.mp4" \
            || echo "  [warn] chunk ${CHUNK} / ${FIELD} 失败, 继续"
    done
done

echo ""
echo "Done: $(date)"
ls -la "$OUTDIR"
