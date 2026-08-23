#!/bin/bash
#SBATCH --job-name=gen_prior
#SBATCH --output=logs/gen_%j.log
#SBATCH --error=logs/gen_%j.err
#SBATCH --partition=eecs            # ← 确认: CPU 分区 (纯 numpy, 不吃 GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00             # ← 确认: 上界猜测 (全域 574163 cell x 9 chunk 串行)

# ══════════════════════════════════════════════════════════════════════════
#  阶段二 / STAGE 2 — 逐 chunk 生成 prior (串行)
#  用法: 阶段一跑完后 ->  sbatch gen_prior.sh
#        链式: sbatch --dependency=afterok:<scan_jobid> gen_prior.sh
#
#  ── 依赖代码 (code deps, 均在 code/ 平铺) ─────────────────────────────────
#     gen_prior.py      本阶段入口
#       └ import lift.py          Nwogu 剖面公式 (落盘 5 通道 [alpha,Ux,Uy,Uz,p_rgh])
#       └ import fw_io.py         FUNWAVE 文件读取 (load_static)
#
#  ── 依赖上一阶段产物 (depends on STAGE 1) ────────────────────────────────
#     $DATA/toffset_scan/c00X.json   ← 本脚本从中读 t_offset, 不手抄 best_k
#
#  ── 输入数据 (inputs) ────────────────────────────────────────────────────
#     $FW/{eta,u,v,mask}_NNNNN, dep.out         FUNWAVE 原生 2D 输出
#     $DATA/coords.npy                          CFD cell 中心坐标 (N,3)
#     $DATA/chunk_00X_times.npy  X=1..9         决定 t_cfd -> FUNWAVE 帧号
#
#  ── 输出 (outputs) ───────────────────────────────────────────────────────
#     $DATA/prior_ktuned/prior_00X_data.npy   (T,N,5) float32  [喂 HPM]
#     $DATA/prior_ktuned/prior_00X_valid.npy  (T,N)   bool     [诊断]
#     $DATA/prior_ktuned/prior_00X_times.npy  (T,)             t_cfd
#     $DATA/prior_ktuned/prior_00X_meta.json                   自描述元信息
#     (np.save 覆盖同名文件; prior_ktuned 无需预先改名)
#
#  ── 注意: 输出是 5 通道含 Uy; 下游 dataset.py 加载时丢 Uy 取 4 通道, 非本脚本职责
# ══════════════════════════════════════════════════════════════════════════

# ---- 可改配置 (edit here) ----
FW=/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output   # ← 确认: 大小写!
DATA=../data/3d/cropped_0.05
XOFF=15.05                          # ← 不标定, 当已知输入 (你确认不动)

# ---- 环境 ----
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---- 运行: 逐 chunk, t_offset 直接从 scan 的 JSON 读 (单一真源, 零转录) ----
for c in 1 2 3 4 5 6 7 8 9; do
  cid=$(printf "%03d" "$c")
  json=$DATA/toffset_scan/c${cid}.json
  if [ ! -f "$json" ]; then
    echo "[err] chunk $c: 缺 $json (阶段一未产出?) -> 跳过"; continue
  fi
  toff=$(python3 -c "import json,sys; v=json.load(open('$json'))['t_offset']; print('' if v is None else v)")
  if [ -z "$toff" ]; then
    echo "[err] chunk $c: t_offset 为空 (scan 无有效 best_k) -> 跳过"; continue
  fi
  echo ">>> chunk $c  t_offset=$toff"
  python gen_prior.py --fw-dir "$FW" \
      --coords   "$DATA/coords.npy" \
      --gt-times "$DATA/chunk_${cid}_times.npy" \
      --chunk "$c" --x-offset "$XOFF" --t-offset "$toff" \
      --out "$DATA/prior_ktuned"
done

python gen_prior.py --fw-dir "$FW" \
    --coords   $DATA/coords.npy \
    --gt-times $DATA/chunk_010_times.npy \
    --chunk 10 --x-offset 15.05 --t-offset 0.0 \
    --out $DATA/prior_ktuned
