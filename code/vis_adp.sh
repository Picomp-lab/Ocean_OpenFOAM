#!/bin/bash
#SBATCH --job-name=vis_adp
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/visadp_%x_%j.log
#SBATCH --error=logs/visadp_%x_%j.err
# 注: --time 按「作业数组、单 task 跑 1 组 x 4 场」定。若不用 --array 串行跑
#     多组串行时提交要放宽: sbatch --time=24:00:00 ...
#
# ══════════════════════════════════════════════════════════════════════════
#  ADP — FUNWAVE 变参 prior 替换 backbone (推理时换, 不重训)
#
#  目的: 拿在教授 prior 上训好的 fwv 线模型, 喂不同 FUNWAVE 算例生成的 prior,
#        看长期 rollout 的细节还原与泛化。**不与 GT 对比** —— 场景已变,
#        用 SUB=lt (无 GT, chunk 10)。
#
#  ⚠️ 2026-08-20: 这条扫描线已收缩到只剩基准算例 TK94。原来的 11 个变参算例
#     (波高 H0381~H0610 五组 + 变坡度 S325/S375 六组) 连同它们的 prior 产物
#     和 results/fwv/ 全部删除, 版本库里也只留 TK94 的 input.txt / gauges.txt。
#     要重做扫描: 用 data/fwv/make_cases.py 从 TK94 造算例 -> 跑 FUNWAVE
#     -> STAGE=prior 生成 prior。原始输出和 prior 都不再随仓库分发。
#
#  ── 三个阶段 (STAGE) ─────────────────────────────────────────────────────
#     STAGE=prior   CPU. 逐算例跑 gen_prior.py (仅 chunk 10, t_offset=0)
#     STAGE=vis     GPU. 逐算例跑 vis.py lt, 换 --prior_dir   [需要阶段一]
#     STAGE=lift    CPU. 逐算例跑 vis.py lift, 从 FUNWAVE 现算 prior 并出图
#                   [**不需要**阶段一 —— 现算, 不读产物]
#
#  ── 提交 (多算例时推荐作业数组: 并行, 单 task 只跑一组) ─────────────────
#     cd <...>/models/code && mkdir -p logs
#     # 阶段一 (CPU):
#     STAGE=prior sbatch --array=0-7 --partition=eecs --gres=none \
#                        --mem=32G --time=02:00:00 vis_adp.sh
#     # 阶段二 (GPU), 串在阶段一之后:
#     STAGE=vis sbatch --array=0-<N-1> --dependency=afterok:<prior_jobid> vis_adp.sh
#     # lift 阶段 (CPU, 独立于前两个):
#     STAGE=lift CHUNK=9 CASES="TK94" \
#         sbatch --partition=eecs --gres=none \
#                --mem=32G --time=04:00:00 vis_adp.sh
#
#     不给 --array 则单个作业内串行跑 $CASES 全部 (多组时需放宽 --time)。
#
#  ── 可覆盖变量 ──────────────────────────────────────────────────────────
#     CASES  要跑的算例 (默认只有 TK94; 空格分隔。--array 按此顺序索引)
#     RUN_TS checkpoint 时间戳目录 (默认 2026-08-12_15-31-45)
#     CHUNK  chunk id (默认 10 —— lt 的无 GT chunk; lift 可用任意 chunk)
#     FIELDS 可视化的场 (默认 alpha Ux Uz p_rgh = checkpoint 里 enabled 的四场)
#     STYLE  tri | scatter (默认 tri; lt / lift 不支持 both)
#     K      仅 lift: 帧移。留空 = vis.py 自己读 toffset_scan (chunk 9 -> +6,
#            chunk 10 无标定 -> 0)
#     FORCE  =1 覆盖已有输出
#
#  ── 依赖代码 ────────────────────────────────────────────────────────────
#     gen_prior.py  └ lift.py  └ fw_io.py        (阶段一)
#     vis.py        └ schema.py dataset.py hpm_model.py   (阶段二)
#
#  ── 输入 / 输出 ─────────────────────────────────────────────────────────
#     in : data/fwv/<case>/output/{eta,u,v,mask}_NNNNN, dep.out
#          data/3d/cropped_0.05/{coords.npy,chunk_010_times.npy}
#          results/train/hpm_fw_aU_h128/$RUN_TS/{.hydra/config.yaml,checkpoints/best.pt}
#     out: results/fwv/priors/<case>/prior_010_*.npy      (~12 GB/组)
#          results/fwv/vis/<case>/longterm_*.mp4
#
#  ── 注意 ────────────────────────────────────────────────────────────────
#     * TK94 与 CFD 真值同参数, 床面一致。自造的变参算例若改了 SLP, 床面会与
#       真值失配 (S325/S375 曾失配 1.3~3.4 cm) —— 那是有意的泛化测试, 不是 bug。
#     * t_offset 固定 0.0: chunk 10 本就如此 (见 gen_prior.sh 末尾)，无需 scan。
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# 必须从 code/ 提交 (与 vis.sh 同约定)
if [ "$(basename "${SLURM_SUBMIT_DIR:-$PWD}")" != "code" ] || [ ! -f "vis.py" ]; then
    echo "ERROR: 必须从项目 code/ 目录提交:  cd <...>/models/code && sbatch vis_adp.sh"
    echo "       当前提交目录: ${SLURM_SUBMIT_DIR:-<非 SLURM>}   cwd: $PWD"
    exit 1
fi
mkdir -p logs

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

STAGE="${STAGE:-}"
case "$STAGE" in
    prior|vis|lift) ;;
    *) echo "ERROR: 必须给 STAGE=prior | vis | lift"; exit 1 ;;
esac

# 只剩基准算例 TK94 (AMP_WK=0.0635, SLP=1:35) —— 也是 web-demo 的默认算例。
# 要扫参数就显式传 CASES=..., 并先用 make_cases.py 造出算例、跑完 FUNWAVE。
ALL_CASES="TK94"
CASES="${CASES:-$ALL_CASES}"
CHUNK="${CHUNK:-10}"
RUN_TS="${RUN_TS:-2026-08-12_15-31-45}"
FIELDS="${FIELDS:-alpha Ux Uz p_rgh}"
STYLE="${STYLE:-tri}"
FORCE="${FORCE:-0}"
# K: 仅 STAGE=lift。留空 = 交给 vis.py 自己定 (读 toffset_scan/c00X.json;
# 没有则回落 0)。chunk 9 有标定结果 k=+6, chunk 10 没有 -> 0, 与阶段一
# gen_prior 的 t_offset=0 一致。要复核别的 k 就显式给 K=<整数>。
K="${K:-}"

# 作业数组模式: --array=0-7 时每个 task 只处理一个算例。
# 不给 --array 则串行跑 $CASES 全部 (两种提交方式都可用)。
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r -a _arr <<< "$CASES"
    [ "$SLURM_ARRAY_TASK_ID" -lt "${#_arr[@]}" ] || {
        echo "ERROR: array id $SLURM_ARRAY_TASK_ID 超出算例数 ${#_arr[@]}"; exit 1; }
    CASES="${_arr[$SLURM_ARRAY_TASK_ID]}"
    echo "[array] task $SLURM_ARRAY_TASK_ID → $CASES"
fi

DATA="$REPO/data/3d/cropped_0.05"
FWROOT="$REPO/data/fwv"
# 输出根目录与 checkpoint 位置都可以从环境变量覆盖，不给就是原来的值。
# web demo 走 results/web/ 那一套（产物删掉即触发重算），手动跑 ADP 时什么都不传，
# 行为与以前完全一致。
OUTROOT="${OUTROOT:-$REPO/results/fwv}"
PRIORROOT="${PRIORROOT:-$OUTROOT/priors}"      # gen_prior 的 .npy 产物
VISROOT="${VISROOT:-$OUTROOT/vis}"             # lt: 模型 rollout 视频
LIFTROOT="${LIFTROOT:-$OUTROOT/lift}"          # lift: prior 本身的视频 (现算)
CKDIR="${CKDIR:-$REPO/results/train/hpm_fw_aU_h128/$RUN_TS}"

CID=$(printf "%03d" "$CHUNK")

echo "════════════════════════════════════════════════════════════"
echo " STAGE = $STAGE   chunk = $CHUNK"
echo " 算例  : $CASES"
echo " 输出  : $OUTROOT/"
echo "════════════════════════════════════════════════════════════"

# ──────────────────────────────── 阶段一 ────────────────────────────────
if [ "$STAGE" = prior ]; then
    export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

    for f in "$DATA/coords.npy" "$DATA/chunk_${CID}_times.npy"; do
        [ -f "$f" ] || { echo "ERROR: 缺输入 $f"; exit 1; }
    done

    for c in $CASES; do
        FW="$FWROOT/$c/output"
        OUT="$PRIORROOT/$c"
        if [ ! -d "$FW" ]; then
            echo "[skip] $c: 找不到 $FW"; continue
        fi
        if [ -f "$OUT/prior_${CID}_data.npy" ] && [ "$FORCE" != 1 ]; then
            echo "[skip] $c: $OUT/prior_${CID}_data.npy 已存在 (FORCE=1 覆盖)"; continue
        fi
        echo ">>> [prior] $c"
        mkdir -p "$OUT"
        python gen_prior.py --fw-dir "$FW" \
            --coords   "$DATA/coords.npy" \
            --gt-times "$DATA/chunk_${CID}_times.npy" \
            --chunk "$CHUNK" --x-offset 15.05 --t-offset 0.0 \
            --out "$OUT"
    done
    echo "阶段一完成。产物: $PRIORROOT/<case>/prior_${CID}_*.npy"
    exit 0
fi

# ─────────────────────── 阶段 lift: 只渲 prior 本身 ───────────────────────
# 不加载 checkpoint (vis.py lift 子命令), 只要 config 取 ChannelSchema ——
# 有了它才是 schema 通道空间 (选列 + αU 加权), 与 lt 的视频逐像素可比。
#
# 2026-08-16: 原名 priorvis, 走 `vis.py prior` 读 gen_prior 产物; 现改为
# `vis.py lift` 从 FUNWAVE 现算。两者数值路径同为 build_frame, 但 lift
# **不依赖阶段一** —— 只渲 prior 时不必先花 5-6 min / 11.5 GB 生成全域产物,
# 且 chunk 9 这类有 GT 的 chunk 也能直接跑。阶段二 (lt) 仍然需要产物。
#
# 注意: 每个 field 各跑一次 vis.py, 抬升会重算一遍 (vis.py 是一次一个 field)。
# 4 个 field 即 4x 冗余计算, 但省下的是一份 11.5 GB 的产物, 这笔账划算。
if [ "$STAGE" = lift ]; then
    CONFIG="$CKDIR/.hydra/config.yaml"
    [ -f "$CONFIG" ] || { echo "ERROR: 缺 $CONFIG"; exit 1; }
    [ "$STYLE" != both ] || { echo "ERROR: lift 阶段不支持 STYLE=both"; exit 1; }
    export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

    for c in $CASES; do
        FW="$FWROOT/$c/output"
        OUT="$LIFTROOT/$c"
        if [ ! -d "$FW" ]; then
            echo "[skip] $c: 缺 FUNWAVE 输出 $FW"; continue
        fi
        mkdir -p "$OUT"
        for FIELD in $FIELDS; do
            f="$OUT/lift_chunk${CHUNK}_${FIELD}_${STYLE}.mp4"
            if [ -f "$f" ] && [ "$FORCE" != 1 ]; then
                echo "[skip] $c/$FIELD: $f 已存在 (FORCE=1 覆盖)"; continue
            fi
            echo ">>> [lift] $c  chunk=$CHUNK  field=$FIELD"
            python vis.py lift \
                --fw-dir      "$FW" \
                --chunk       "$CHUNK" \
                --data-dir    "$DATA" \
                --config_path "$CONFIG" \
                --field       "$FIELD" \
                --style       "$STYLE" \
                ${K:+--k "$K"} \
                --output      "$OUT/lift_chunk${CHUNK}_${FIELD}.mp4"
        done
    done
    echo "lift 完成。产物: $LIFTROOT/<case>/lift_chunk${CHUNK}_*_${STYLE}.mp4"
    exit 0
fi

# ──────────────────────────────── 阶段二 ────────────────────────────────
CONFIG="$CKDIR/.hydra/config.yaml"
CKPT="$CKDIR/checkpoints/best.pt"
for f in "$CONFIG" "$CKPT"; do
    [ -f "$f" ] || {
        echo "ERROR: 缺 $f"
        echo "  - 换一次训练: RUN_TS=<时间戳>，或直接 CKDIR=<目录>"
        echo "  - results/train/ 已经不随仓库分发（在 results_*.tar 里），"
        echo "    要用先 ./archive/restore.sh results"
        exit 1
    }
done
[ "$STYLE" != both ] || { echo "ERROR: lt 不支持 STYLE=both, 用 tri 或 scatter"; exit 1; }

echo " checkpoint: $CKPT"

for c in $CASES; do
    PDIR="$PRIORROOT/$c"
    OUT="$VISROOT/$c"
    if [ ! -f "$PDIR/prior_${CID}_data.npy" ]; then
        echo "[skip] $c: 缺 $PDIR/prior_${CID}_data.npy (先跑 STAGE=prior)"; continue
    fi
    if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && [ "$FORCE" != 1 ]; then
        echo "[skip] $c: $OUT/ 已有内容 (FORCE=1 覆盖)"; continue
    fi
    mkdir -p "$OUT"
    for FIELD in $FIELDS; do
        echo ">>> [vis lt] $c  field=$FIELD  chunk=$CHUNK"
        python vis.py lt \
            --config_path "$CONFIG" \
            --checkpoint  "$CKPT" \
            --data_dir    "$DATA" \
            --prior_dir   "$PDIR" \
            --chunk_id    "$CHUNK" \
            --field       "$FIELD" \
            --style       "$STYLE" \
            --output      "$OUT/longterm_chunk${CHUNK}_${FIELD}.mp4"
    done
done
echo "阶段二完成。产物: $VISROOT/<case>/longterm_chunk${CHUNK}_*.mp4"
