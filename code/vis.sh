#!/bin/bash
#SBATCH --job-name=vis
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vis_%j.log
#
# ══════════════════════════════════════════════════════════════════════════
#  推理可视化 / VIS — pred | lt 派发 (两条线通用)
#  用法: 从 code/ 提交 ->  mkdir -p logs && sbatch vis.sh
#  ⚠️ logs/ 必须**提交前**就存在: #SBATCH --output 在脚本执行之前生效，目录不在时
#     SLURM 会把日志整个丢掉，而作业状态照样是 COMPLETED（实测），出事没法查。
#
#  ── 子命令 (SUB, 默认 pred) ──────────────────────────────────────────────
#     SUB=pred  GT|pred 逐帧对比 (fwv 附带 tf/rollout gap 与 [自检] tf nRMSE)
#               默认: chunk=9  style=both  FIELDS= pure 全 enabled / fwv 四场
#     SUB=lt    长期 rollout, 无 GT, 流式 (仅 fwv 线)
#               默认: chunk=10 style=tri   FIELDS= alpha
#
#  ── 误差行 (DIFF, 仅 pred; 默认空 = 不渲, 与老行为一致) ────────────────────
#     DIFF=abs   Δ = pred − GT, 物理单位, 色标自适应 ±p99|Δ| -> 看误差长在哪
#     DIFF=pct   Δ% = Δ/S x 100, 色标固定 ±100%       -> 跨 run/ckpt 并排比
#     DIFF=both  两行都要 (共 4 行)
#     配套: PCT_SCALE=range|rms|p99 (Δ% 的分母 S, 默认 range=GT 满量程)
#           DIFF_PCT=99             (仅 abs 行的色标分位数)
#           ROW_H=                  (每行英寸高; 留空 = DIFF=both 时自动 10.0,
#                                    见下方 —— 默认 10.8 x 4 行会越过 4096 px)
#     其余接口留在 vis.py, 手敲 (无需定位逻辑):
#       python vis.py gt    --data_dir <d> --chunks 0-10          # 纯数据探查
#       python vis.py align --fw-dir <fw>/output --chunk 9        # 配准 (训练前)
#       python vis.py nofb  --config_path ... --checkpoint ...    # 无反馈臂 3 行
#
#  ── 定位方式 (locate ckpt/config) ────────────────────────────────────────
#     方式 1 (首选): CONFIG=... CKPT=... sbatch vis.sh        显式, 无歧义
#     方式 2 (兜底): RUN=runname [TS=时间戳] sbatch vis.sh    results/train/ 下解析
#         RUN 必给且不含 '/'; TS 省则取该 RUN 下时间戳最新的一次 (按目录名字典序)
#
#  ── 依赖代码 (code deps, 均在 code/ 平铺) ─────────────────────────────────
#     vis.py            本入口 (SUB=pred|lt)
#       └ import schema.py        ChannelSchema (通道派生)
#       └ import dataset.py       assemble/reconstruct/resolve_stats (与训练共用)
#       └ import hpm_model.py     HPM
#
#  ── 输入 / 输出 (io) ─────────────────────────────────────────────────────
#     in : $CONFIG (.hydra/config.yaml)  $CKPT (best.pt)  DATA/PRIOR (从 config 读)
#     out: results/vis/$SUB/$FEATURE/  (pred: compare_*_{pred,tf}_{tri,scatter}.mp4;
#          lt: longterm_*_{tri}.mp4) —— 只有视频
#          npy 全部默认不存, 要离线分析时给 vis.py 加 --save_rmse (逐帧 RMSE,
#          各 ~1.7 KB) / --save_preds (全场预测, ~0.9 GB/场); RMSE 数值照常进日志
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# 必须从 code/ 提交: 提交目录名==code 且 cwd 有 vis.py。
# 用 $SLURM_SUBMIT_DIR (非 $BASH_SOURCE —— sbatch 拷到 spool, 对不上)。:- 防 set -u 崩。
if [ "$(basename "${SLURM_SUBMIT_DIR:-}")" != "code" ] || [ ! -f "vis.py" ]; then
    echo "ERROR: 必须从项目 code/ 目录提交:  cd <...>/models/code && sbatch vis.sh"
    echo "       当前提交目录: ${SLURM_SUBMIT_DIR:-<非 SLURM 环境>}   cwd: $PWD"
    exit 1
fi
mkdir -p logs                              # 兜底; 但 #SBATCH --output 在脚本跑之前就要用它,
                                           # 所以提交前 logs/ 就得在 (见顶部 banner)
# $REPO 由下面的 activate.sh 导出 (= 仓库根, results/ data/ 与 code/ 同级)

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

# ---- 定位 checkpoint / config (显式 CONFIG/CKPT 优先, 否则 RUN[/TS]) ----
# RUN = runname (必给, 不含 /); TS = 时间戳 (可选, 省则取该 RUN 下最新且含 best.pt 的)。
# "最新" 按目录名字典序 (非 mtime)。
TRAIN_ROOT="$REPO/results/train"
RUN="${RUN:-}"
TS="${TS:-}"

if [ -z "${CONFIG:-}" ] || [ -z "${CKPT:-}" ]; then
    [ -n "$RUN" ] || {
        echo "ERROR: 未给 CONFIG/CKPT 时, 必须给 RUN=runname (可选 TS=时间戳)"; exit 1; }
    case "$RUN" in */*)
        echo "ERROR: RUN 不能含 '/' (那是 runname, 时间戳请用 TS=...)。当前 RUN='$RUN'"
        exit 1 ;;
    esac

    RUN_DIR="$TRAIN_ROOT/$RUN"
    [ -d "$RUN_DIR" ] || { echo "ERROR: 找不到 $RUN_DIR (RUN 拼写错误?)"; exit 1; }

    # 目录布局是 <总方向 runname>/[<细节修改 override_dirname>/]<时间戳>/checkpoints/best.pt
    # —— 带 CLI 覆盖的 run 会被 hydra 多插一层 override_dirname，没覆盖时就只有两层。
    # 两种都要认，所以按 best.pt 去找，而不是假定深度。"最新" 仍按时间戳目录名字典序
    # (它们是 %Y-%m-%d_%H-%M-%S，字典序==时间序)。
    RESOLVED=""; TS_BEST=""; NCAND=0
    while IFS= read -r ck; do
        [ -n "$ck" ] || continue
        rel="${ck#"$TRAIN_ROOT/"}"; rel="${rel%/checkpoints/best.pt}"
        ts="${rel##*/}"
        if [ -n "$TS" ] && [ "$ts" != "$TS" ]; then continue; fi
        NCAND=$((NCAND + 1))
        if [ -z "$TS_BEST" ] || [[ "$ts" > "$TS_BEST" ]]; then TS_BEST="$ts"; RESOLVED="$rel"; fi
    done < <(find "$RUN_DIR" -mindepth 2 -maxdepth 4 -path '*/checkpoints/best.pt' 2>/dev/null | sort)

    if [ -z "$RESOLVED" ]; then
        if [ -n "$TS" ]; then echo "ERROR: $RUN_DIR 下没有 TS=$TS 且含 checkpoints/best.pt 的目录"
        else echo "ERROR: $RUN_DIR 下没有含 checkpoints/best.pt 的目录"; fi
        echo "       现有的是:"
        find "$RUN_DIR" -mindepth 2 -maxdepth 4 -path '*/checkpoints/best.pt' 2>/dev/null \
            | sed "s|$TRAIN_ROOT/||; s|/checkpoints/best.pt||; s/^/         /" | head -10
        exit 1
    fi
    if [ -n "$TS" ] && [ "$NCAND" -gt 1 ]; then
        echo "ERROR: TS=$TS 在 $RUN 下命中 $NCAND 个 (不同的细节修改层)，请直接给 CONFIG/CKPT:"
        find "$RUN_DIR" -mindepth 2 -maxdepth 4 -path "*/$TS/checkpoints/best.pt" 2>/dev/null \
            | sed 's/^/         /'
        exit 1
    fi
    TS="$TS_BEST"
    echo "[vis.sh] RUN='$RUN' -> $RESOLVED"
    if [ "$NCAND" -gt 1 ]; then
        echo "[vis.sh] （该 RUN 下有 $NCAND 个候选，取时间戳最新的这个）"
    fi

    CONFIG="$TRAIN_ROOT/$RESOLVED/.hydra/config.yaml"
    CKPT="$TRAIN_ROOT/$RESOLVED/checkpoints/best.pt"
    FEATURE="${FEATURE:-$RESOLVED}"        # 输出目录镜像 results/train 下的同一条路径
else
    echo "[vis.sh] 使用显式 CONFIG/CKPT"
    # ckpt 若就在 results/train 下, 直接镜像它那条路径, 别丢溯源;
    # 指到别处 (临时快照之类) 才回落到 explicit_<时间>。
    case "$CKPT" in
        "$TRAIN_ROOT"/*/checkpoints/*)
            _rel="${CKPT#"$TRAIN_ROOT/"}"; _rel="${_rel%/checkpoints/*}"
            FEATURE="${FEATURE:-$_rel}" ;;
        *)  FEATURE="${FEATURE:-explicit_$(date +%m%d_%H%M)}" ;;
    esac
fi

[ -f "$CONFIG" ] || { echo "ERROR: config 不存在: $CONFIG"; exit 1; }
[ -f "$CKPT" ]   || { echo "ERROR: ckpt 不存在:   $CKPT";   exit 1; }

# ---- 从快照 config 读 data_dir / prior_dir / window / enabled 通道 (与训练一致) ----
# helper 先注册 repo resolver (config 用 ${repo:}), 否则裸 OmegaConf.load 会抛。
# 输出单行: DATA PRIOR WINDOW ch1 ch2 ...  (prior 空时占位 "-", 保证位置对齐)。
# 抓输出+判退出码 (进程替换里 set -e 不可靠, 会静默读到空)。
DP=$(REPO="$REPO" python - "$CONFIG" <<'PYEOF'
import os, sys
from omegaconf import OmegaConf
if not OmegaConf.has_resolver("repo"):
    OmegaConf.register_new_resolver("repo", lambda: os.environ["REPO"])
cfg = OmegaConf.load(sys.argv[1])
prior = cfg.data.get("prior_dir", "") or "-"
window = int(cfg.data.window)
chs = cfg.data.get("channels", None)
names = [c["name"] for c in chs if c.get("enabled", True)] if chs else []
print(cfg.data.dir, prior, window, " ".join(names))
PYEOF
) || { echo "ERROR: 读取 config 失败 (见上方 traceback)"; exit 1; }
read -r DATA PRIOR WINDOW ENABLED_CHS <<< "$DP"
[ "$PRIOR" = "-" ] && PRIOR=""
[ -n "$DATA" ] || { echo "ERROR: 未能从 config 解析 data.dir"; exit 1; }

# ---- SUB: 子命令 pred | lt (gt/align/nofb 见顶部 banner, 手敲) ----
SUB="${SUB:-pred}"
case "$SUB" in pred|lt) ;; *)
    echo "ERROR: SUB 只支持 pred|lt (gt/align/nofb 请手敲 python vis.py ...)"; exit 1 ;;
esac

# 线别 by window (>0 pure / ==0 fwv)
if [ "$WINDOW" -gt 0 ]; then LINE=pure; else LINE=fwv; fi
# lt 仅 fwv 线 (需 prior; vis.py 内部亦 assert)
if [ "$SUB" = lt ] && [ "$LINE" != fwv ]; then
    echo "ERROR: lt 是 fwv 线专属 (需 prior); 当前 RUN 是 pure (window=$WINDOW)"; exit 1; fi

# ---- 默认参数 (环境变量覆盖), 按 SUB 分流 ----
#   pred: chunk=9  style=both  FIELDS= pure 全 enabled / fwv 四场
#   lt  : chunk=10 style=tri   FIELDS= alpha  (长期无 GT, 定性看界面)
if [ "$SUB" = lt ]; then
    CHUNK="${CHUNK:-10}"; STYLE="${STYLE:-tri}"; FIELDS="${FIELDS:-alpha}"
else
    CHUNK="${CHUNK:-9}";  STYLE="${STYLE:-both}"
    if [ "$LINE" = pure ]; then
        [ -n "$ENABLED_CHS" ] || { echo "ERROR: pure 线未从 config 解析到 enabled 通道"; exit 1; }
        FIELDS="${FIELDS:-$ENABLED_CHS}"
    else
        FIELDS="${FIELDS:-alpha Ux Uz p_rgh}"
    fi
fi
NFRAMES="${NFRAMES:-0}"                    # 0=跑到末尾; pred 验证设 8 触发快速自检
# lt 是流式 rollout, 不能渲两遍 (vis.py assert style!=both)
if [ "$SUB" = lt ] && [ "$STYLE" = both ]; then
    echo "ERROR: lt 不支持 STYLE=both; 用 tri 或 scatter"; exit 1; fi

# ---- 误差行 (仅 pred; vis.py 只给 pred 挂了 --diff) ----
# ROW_H 留空时按行数自适应: dpi 固定 100, 像素高 = row_h x 行数 x 100。DIFF=both
# 是 4 行, 默认 10.8 -> 4320 px, 越过不少播放器硬解的 4096 上限 (vis.py render()
# 只警告不改默认) -> 这里降到 10.0 = 4000 px。abs/pct 单行版共 3 行, 10.8 才 3240,
# 不动。显式给 ROW_H 一律照办。
DIFF="${DIFF:-}"
ROW_H="${ROW_H:-}"
DIFF_ARGS=()
if [ -n "$DIFF" ]; then
    case "$DIFF" in abs|pct|both) ;; *)
        echo "ERROR: DIFF 只支持 abs|pct|both (留空=不渲误差行)。当前 DIFF='$DIFF'"
        exit 1 ;;
    esac
    if [ "$SUB" != pred ]; then
        echo "ERROR: DIFF 仅 SUB=pred 支持 (lt 无 GT, 无从算 Δ)"; exit 1; fi
    DIFF_ARGS=(--diff "$DIFF"
               --pct-scale "${PCT_SCALE:-range}"
               --diff-pct  "${DIFF_PCT:-99}")
    if [ -z "$ROW_H" ] && [ "$DIFF" = both ]; then ROW_H=10.0; fi
fi
# 独立的 if, 不写 `[ ... ] && ...` —— 那在 set -e 下条件为假就是整脚本退出。
if [ -n "$ROW_H" ]; then DIFF_ARGS+=(--row-h "$ROW_H"); fi

OUT_ROOT="$REPO/results/vis/$SUB/$FEATURE"

NF_SHOW=$([ "$NFRAMES" -le 0 ] 2>/dev/null && echo "0(=full-chunk)" || echo "$NFRAMES")
echo "========================================"
echo "vis.py $SUB"
echo "  CONFIG : $CONFIG"
echo "  CKPT   : $CKPT"
echo "  DATA   : $DATA"
echo "  PRIOR  : ${PRIOR:-<from config>}"
echo "  line=$LINE (window=$WINDOW)  chunk=$CHUNK  n_frames=$NF_SHOW  style=$STYLE"
echo "  fields : $FIELDS"
if [ ${#DIFF_ARGS[@]} -gt 0 ]; then echo "  diff   : ${DIFF_ARGS[*]}"; fi
echo "  out    : $OUT_ROOT/"
echo "  node   : $(hostname)   date: $(date)"
echo "========================================"

# ---- 覆盖保护 (overwrite guard): FORCE=1 强制覆盖 ----
if [ -d "$OUT_ROOT" ] && [ -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "ERROR: $OUT_ROOT/ 已有内容。FORCE=1 覆盖, 或换 FEATURE=xxx。"
        exit 1
    fi
    echo "WARN: FORCE=1, 覆盖 $OUT_ROOT/"
fi
mkdir -p "$OUT_ROOT"

# ---- 逐场跑 (--field 一次一个; vis.py 自动追加样式/npy 后缀) ----
for FIELD in $FIELDS; do
    echo "=== [$SUB] field $FIELD  chunk $CHUNK ==="
    if [ "$SUB" = pred ]; then
        python -u vis.py pred \
            --config_path "$CONFIG" \
            --checkpoint  "$CKPT" \
            --data_dir    "$DATA" \
            --chunk_id    "$CHUNK" \
            --n_frames    "$NFRAMES" \
            --style       "$STYLE" \
            --field       "$FIELD" \
            --output      "$OUT_ROOT/compare_chunk${CHUNK}_${FIELD}.mp4" \
            ${DIFF_ARGS[@]+"${DIFF_ARGS[@]}"}
    else   # lt: 长期 rollout, 无 GT, 流式 (prior_dir 由 vis.py 从 config 读)
        python -u vis.py lt \
            --config_path "$CONFIG" \
            --checkpoint  "$CKPT" \
            --data_dir    "$DATA" \
            --chunk_id    "$CHUNK" \
            --n_frames    "$NFRAMES" \
            --style       "$STYLE" \
            --field       "$FIELD" \
            --output      "$OUT_ROOT/longterm_chunk${CHUNK}_${FIELD}.mp4" \
            ${DIFF_ARGS[@]+"${DIFF_ARGS[@]}"}
    fi
done

echo ""
echo "Done: $(date)"