#!/bin/bash
#SBATCH --job-name=vis_lift
#SBATCH --output=logs/vis_lift_%j.log
#SBATCH --error=logs/vis_lift_%j.err
#SBATCH --partition=eecs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ============================================================
# 用法:
#   sbatch vis_lift.sh <chunk>                 只出 lift 四支 mp4 (默认 chunk 6)
#   STITCH=1 sbatch vis_lift.sh <chunk>        额外把 GT alpha 拉来上下拼接
#   STITCH=1 GT_DIR=/path sbatch vis_lift.sh 6 自定 GT 目录
#
# 数据: TingKirby1994_3D_spilling_2 (参数对齐 CFD 的新 FUNWAVE run)
#   Mglob=1575 (域长 31.5m), Nglob=30, dx=dy=0.02, PLOT_INTV=0.05, 共 2000 帧
#
# 坐标对应 (教授给的域图):  x_fw = x_cfd + 15.05
#   FUNWAVE 6.35 <-> OpenFOAM -8.7 ; 14.35 <-> -0.7 ; 31.5 <-> 16.45
#   x_fw < 6.35 是 sponge layer + 造波板背面, 不属于 OpenFOAM 域, 永不入 prior。
#
# 显示层对齐 GT: --x-shift 换算到 CFD 坐标系, --x-lim/--z-lim 取 GT slice cache
#   的实测范围, figsize 3840x1080 与 GT mp4 同尺寸 (stitch 无缩放)。
#   纯显示层, lift.py 数值一字不动。
#
# 仍未对齐: t-offset (新 case 尚未标定) + 破碎区物理分叉 (设计特征, 不该对齐)
# ============================================================
CHUNK=${1:-6}
STITCH=${STITCH:-0}
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
GT_DIR=${GT_DIR:-$REPO/legacy/hpm/vis/gt_alpha}
OUT_DIR=${OUT_DIR:-fwv/vis_lift}

cd "$REPO/legacy/hpm"
mkdir -p logs "${OUT_DIR}"


export PYTHONPATH="$PWD:$PYTHONPATH"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ---- lift 渲染 (matplotlib; 无需 display / xvfb) ----
# --x-min/--x-max : FUNWAVE 原生坐标选窗 = CFD 域 [-2.495, 16.4475] + 15.05
# --x-shift       : 显示层平移到 CFD 坐标系
# --x-lim/--z-lim : 与 GT slice cache 逐位对齐 (slice_xz.npy 实测值)
# --margins       : detect_margins.py 从 gt_alpha_chunk006_tri.mp4 实测的 axes box
#                   (GT: 像素 3534x865 @ 3840x1080)。改了 GT 渲染脚本需重测。
python fwv/vis_lift.py \
    --fw-dir /nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output \
    --chunk "${CHUNK}" \
    --mglob 1575 --nglob 30 --dx 0.02 --dy 0.02 --plot-intv 0.05 \
    --x-min 12.555 --x-max 31.4975 \
    --x-shift 15.05 \
    --x-lim -2.495 16.4475 \
    --z-min -0.399772 --z-max 0.147884 \
    --z-lim -0.399772 0.147884 \
    --margins 0.0500 0.0991 0.9703 0.9000 \
    --out-dir "${OUT_DIR}"

# ============================================================
# 可选: 与 GT alpha 上下拼接 (STITCH=1 才执行)
#   lift 与 GT 现在同为 3840x1080 -> 直接 vstack, 无缩放损失
#   产物 3840x2160, lift 在上 / GT 在下
# ============================================================
if [ "${STITCH}" = "1" ]; then
    CID=$(printf "%03d" "${CHUNK}")
    LIFT_MP4="${OUT_DIR}/lift_${CID}_alpha_tri.mp4"
    GT_MP4="${GT_DIR}/gt_alpha_chunk${CID}_tri.mp4"
    STITCH_MP4="${OUT_DIR}/stitch_${CID}_alpha.mp4"

    echo "========================================"
    echo "[stitch] lift(上): ${LIFT_MP4}"
    echo "[stitch] GT  (下): ${GT_MP4}"
    echo "[stitch] out:      ${STITCH_MP4}"
    echo "========================================"

    if [ ! -f "${LIFT_MP4}" ]; then
        echo "[stitch] ERROR: lift mp4 不存在 (渲染是否失败? 查 .err): ${LIFT_MP4}"
        exit 1
    fi
    if [ ! -f "${GT_MP4}" ]; then
        echo "[stitch] ERROR: GT mp4 不存在: ${GT_MP4}"
        echo "        检查 GT_DIR 和 chunk 号 (期望 gt_alpha_chunk${CID}_tri.mp4)"
        exit 1
    fi

    # scale=3840:-2 在同尺寸时是 no-op; 保留以防 fig 尺寸被改。
    ffmpeg -y -hide_banner -loglevel error \
        -i "${LIFT_MP4}" -i "${GT_MP4}" \
        -filter_complex "[0:v]scale=3840:-2[top];[top][1:v]vstack=inputs=2[v]" \
        -map "[v]" -c:v libx264 -pix_fmt yuv420p \
        "${STITCH_MP4}"

    if [ -f "${STITCH_MP4}" ]; then
        echo "[stitch] done -> ${STITCH_MP4}"
    else
        echo "[stitch] ERROR: ffmpeg 拼接失败"
        exit 1
    fi
fi

echo "Done: $(date)"