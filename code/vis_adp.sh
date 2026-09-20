#!/bin/bash
#SBATCH --job-name=vis_adp
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/visadp_%x_%j.log
#SBATCH --error=logs/visadp_%x_%j.err
# Note: --time is set for "job array, one task runs 1 case x 4 fields". Without --array,
#     several cases run serially in one job and the submission must relax it:
#     sbatch --time=24:00:00 ...
#
# ==========================================================================
#  ADP -- FUNWAVE varied-parameter prior swapped into the backbone
#         (swapped at inference time, no retraining)
#
#  Purpose: take the fwv-line model trained on the professor's prior, feed it priors generated
#        from different FUNWAVE cases, and look at detail recovery and generalisation over a
#        long rollout. **No comparison against GT** -- the scenario has changed, so SUB=lt is
#        used (no GT, chunk 10).
#
#  NOTE 2026-08-20: this scan line has been reduced to the reference case TK94 alone. The
#     original 11 varied-parameter cases (five wave heights H0381~H0610 plus six varied slopes
#     S325/S375) were deleted together with their prior output and results/fwv/, and the repo
#     now keeps only input.txt / gauges.txt for TK94.
#     To redo the scan: build cases from TK94 with data/fwv/make_cases.py -> run FUNWAVE
#     -> generate priors with STAGE=prior. Neither the raw output nor the priors are
#     distributed with the repo any more.
#
#  -- the three stages (STAGE) ---------------------------------------------
#     STAGE=prior   CPU. Runs gen_prior.py case by case (chunk 10 only, t_offset=0)
#     STAGE=vis     GPU. Runs vis.py lt case by case, swapping --prior_dir   [needs stage 1]
#     STAGE=lift    CPU. Runs vis.py lift case by case, computing the prior from FUNWAVE on the
#                   fly and plotting it
#                   [does **not** need stage 1 -- computed live, reads no stored output]
#
#  -- submitting (with several cases a job array is preferred: parallel, one case per task) --
#     cd <...>/models/code && mkdir -p logs
#     # stage 1 (CPU):
#     STAGE=prior sbatch --array=0-7 --partition=eecs --gres=none \
#                        --mem=32G --time=02:00:00 vis_adp.sh
#     # stage 2 (GPU), chained after stage 1:
#     STAGE=vis sbatch --array=0-<N-1> --dependency=afterok:<prior_jobid> vis_adp.sh
#     # the lift stage (CPU, independent of the other two):
#     STAGE=lift CHUNK=9 CASES="TK94" \
#         sbatch --partition=eecs --gres=none \
#                --mem=32G --time=04:00:00 vis_adp.sh
#
#     Without --array, all of $CASES runs serially inside a single job (relax --time for
#     several cases).
#
#  -- overridable variables ------------------------------------------------
#     CASES  cases to run (default is TK94 alone; space-separated. --array indexes this order)
#     RUN_TS checkpoint timestamp directory (default 2026-08-12_15-31-45)
#     CHUNK  chunk id (default 10 -- the GT-free chunk used by lt; lift accepts any chunk)
#     FIELDS fields to visualise (default alpha Ux Uz p_rgh = the four enabled in the checkpoint)
#     STYLE  tri | scatter (default tri; lt / lift do not support both)
#     K      lift only: frame shift. Leave empty and vis.py reads toffset_scan itself
#            (chunk 9 -> +6, chunk 10 has no calibration -> 0)
#     FORCE  =1 overwrites existing output
#
#  -- code deps ------------------------------------------------------------
#     gen_prior.py  -> lift.py  -> fw_io.py              (stage 1)
#     vis.py        -> schema.py dataset.py hpm_model.py (stage 2)
#
#  -- io -------------------------------------------------------------------
#     in : data/fwv/<case>/output/{eta,u,v,mask}_NNNNN, dep.out
#          data/3d/cropped_0.05/{coords.npy,chunk_010_times.npy}
#          results/train/hpm_fw_aU_h128/$RUN_TS/{.hydra/config.yaml,checkpoints/best.pt}
#     out: results/fwv/priors/<case>/prior_010_*.npy      (~12 GB per case)
#          results/fwv/vis/<case>/longterm_*.mp4
#
#  -- notes ----------------------------------------------------------------
#     * TK94 shares its parameters with the CFD ground truth and the bed matches. A
#       hand-built varied-parameter case that changes SLP will have a bed that no longer
#       matches the truth (S325/S375 were off by 1.3~3.4 cm) -- that is a deliberate
#       generalisation test, not a bug.
#     * t_offset is fixed at 0.0: that is already the case for chunk 10 (see the end of
#       gen_prior.sh), so no scan is needed.
# ==========================================================================

set -euo pipefail

# Must be submitted from code/ (same convention as vis.sh)
if [ "$(basename "${SLURM_SUBMIT_DIR:-$PWD}")" != "code" ] || [ ! -f "vis.py" ]; then
    echo "ERROR: must be submitted from the project's code/ directory:  cd <...>/models/code && sbatch vis_adp.sh"
    echo "       current submit directory: ${SLURM_SUBMIT_DIR:-<not SLURM>}   cwd: $PWD"
    exit 1
fi
mkdir -p logs

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # find conda, activate the environment, and export $REPO

STAGE="${STAGE:-}"
case "$STAGE" in
    prior|vis|lift) ;;
    *) echo "ERROR: STAGE=prior | vis | lift is required"; exit 1 ;;
esac

# Only the reference case TK94 is left (AMP_WK=0.0635, SLP=1:35) -- also the default case for
# web-demo. To sweep parameters, pass CASES=... explicitly, having first built the cases with
# make_cases.py and run FUNWAVE on them.
ALL_CASES="TK94"
CASES="${CASES:-$ALL_CASES}"
CHUNK="${CHUNK:-10}"
RUN_TS="${RUN_TS:-2026-08-12_15-31-45}"
FIELDS="${FIELDS:-alpha Ux Uz p_rgh}"
STYLE="${STYLE:-tri}"
FORCE="${FORCE:-0}"
# K: STAGE=lift only. Leave it empty and vis.py decides (reading toffset_scan/c00X.json, falling
# back to 0 when absent). chunk 9 has a calibrated k=+6, chunk 10 has none -> 0, matching the
# t_offset=0 of gen_prior in stage 1. To re-check a different k, pass K=<integer> explicitly.
K="${K:-}"

# Job-array mode: with --array=0-7 each task handles exactly one case.
# Without --array, all of $CASES runs serially (both submission forms work).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r -a _arr <<< "$CASES"
    [ "$SLURM_ARRAY_TASK_ID" -lt "${#_arr[@]}" ] || {
        echo "ERROR: array id $SLURM_ARRAY_TASK_ID exceeds the number of cases ${#_arr[@]}"; exit 1; }
    CASES="${_arr[$SLURM_ARRAY_TASK_ID]}"
    echo "[array] task $SLURM_ARRAY_TASK_ID -> $CASES"
fi

DATA="$REPO/data/3d/cropped_0.05"
FWROOT="$REPO/data/fwv"
# Both the output root and the checkpoint location can be overridden from the environment;
# leave them unset and the original values apply.
# The web demo uses the results/web/ set of paths (deleting the output triggers recomputation);
# running ADP by hand passes nothing and behaves exactly as before.
OUTROOT="${OUTROOT:-$REPO/results/fwv}"
PRIORROOT="${PRIORROOT:-$OUTROOT/priors}"      # the .npy output of gen_prior
VISROOT="${VISROOT:-$OUTROOT/vis}"             # lt: the model's rollout videos
LIFTROOT="${LIFTROOT:-$OUTROOT/lift}"          # lift: videos of the prior itself (computed live)
CKDIR="${CKDIR:-$REPO/results/train/hpm_fw_aU_h128/$RUN_TS}"

CID=$(printf "%03d" "$CHUNK")

echo "============================================================"
echo " STAGE = $STAGE   chunk = $CHUNK"
echo " cases : $CASES"
echo " output: $OUTROOT/"
echo "============================================================"

# -------------------------------- stage 1 --------------------------------
if [ "$STAGE" = prior ]; then
    export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

    for f in "$DATA/coords.npy" "$DATA/chunk_${CID}_times.npy"; do
        [ -f "$f" ] || { echo "ERROR: missing input $f"; exit 1; }
    done

    for c in $CASES; do
        FW="$FWROOT/$c/output"
        OUT="$PRIORROOT/$c"
        if [ ! -d "$FW" ]; then
            echo "[skip] $c: cannot find $FW"; continue
        fi
        if [ -f "$OUT/prior_${CID}_data.npy" ] && [ "$FORCE" != 1 ]; then
            echo "[skip] $c: $OUT/prior_${CID}_data.npy already exists (FORCE=1 overwrites)"; continue
        fi
        echo ">>> [prior] $c"
        mkdir -p "$OUT"
        python gen_prior.py --fw-dir "$FW" \
            --coords   "$DATA/coords.npy" \
            --gt-times "$DATA/chunk_${CID}_times.npy" \
            --chunk "$CHUNK" --x-offset 15.05 --t-offset 0.0 \
            --out "$OUT"
    done
    echo "Stage 1 done. Output: $PRIORROOT/<case>/prior_${CID}_*.npy"
    exit 0
fi

# ------------------ stage lift: render the prior itself only ------------------
# Loads no checkpoint (the vis.py lift sub-command); it needs the config only to obtain the
# ChannelSchema -- that is what puts it in the schema channel space (column selection + alphaU
# weighting), making it pixel-comparable with the lt videos.
#
# 2026-08-16: formerly called priorvis, using `vis.py prior` to read gen_prior output; it now
# uses `vis.py lift`, computing from FUNWAVE on the fly. Both take the same numerical path
# through build_frame, but lift **does not depend on stage 1** -- rendering the prior alone no
# longer requires first spending 5-6 min and 11.5 GB generating whole-domain output, and a
# chunk with GT such as chunk 9 can be run directly too. Stage 2 (lt) still needs the output.
#
# Note: vis.py runs once per field, so the lifting is recomputed each time (vis.py handles one
# field at a time). Four fields means 4x redundant computation, but what it saves is an 11.5 GB
# artifact, which is a good trade.
if [ "$STAGE" = lift ]; then
    CONFIG="$CKDIR/.hydra/config.yaml"
    [ -f "$CONFIG" ] || { echo "ERROR: missing $CONFIG"; exit 1; }
    [ "$STYLE" != both ] || { echo "ERROR: the lift stage does not support STYLE=both"; exit 1; }
    export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

    for c in $CASES; do
        FW="$FWROOT/$c/output"
        OUT="$LIFTROOT/$c"
        if [ ! -d "$FW" ]; then
            echo "[skip] $c: missing FUNWAVE output $FW"; continue
        fi
        mkdir -p "$OUT"
        for FIELD in $FIELDS; do
            f="$OUT/lift_chunk${CHUNK}_${FIELD}_${STYLE}.mp4"
            if [ -f "$f" ] && [ "$FORCE" != 1 ]; then
                echo "[skip] $c/$FIELD: $f already exists (FORCE=1 overwrites)"; continue
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
    echo "lift done. Output: $LIFTROOT/<case>/lift_chunk${CHUNK}_*_${STYLE}.mp4"
    exit 0
fi

# -------------------------------- stage 2 --------------------------------
CONFIG="$CKDIR/.hydra/config.yaml"
CKPT="$CKDIR/checkpoints/best.pt"
for f in "$CONFIG" "$CKPT"; do
    [ -f "$f" ] || {
        echo "ERROR: missing $f"
        echo "  - for a different training run: RUN_TS=<timestamp>, or CKDIR=<directory> directly"
        echo "  - results/train/ is no longer distributed with the repo (it is in results_*.tar);"
        echo "    to use it, run ./archive/restore.sh results first"
        exit 1
    }
done
[ "$STYLE" != both ] || { echo "ERROR: lt does not support STYLE=both; use tri or scatter"; exit 1; }

echo " checkpoint: $CKPT"

for c in $CASES; do
    PDIR="$PRIORROOT/$c"
    OUT="$VISROOT/$c"
    if [ ! -f "$PDIR/prior_${CID}_data.npy" ]; then
        echo "[skip] $c: missing $PDIR/prior_${CID}_data.npy (run STAGE=prior first)"; continue
    fi
    if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && [ "$FORCE" != 1 ]; then
        echo "[skip] $c: $OUT/ already has content (FORCE=1 overwrites)"; continue
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
echo "Stage 2 done. Output: $VISROOT/<case>/longterm_chunk${CHUNK}_*.mp4"
