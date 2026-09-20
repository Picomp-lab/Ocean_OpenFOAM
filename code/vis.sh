#!/bin/bash
#SBATCH --job-name=vis
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vis_%j.log
#
# ==========================================================================
#  VIS -- inference visualisation, dispatching pred | lt (shared by both lines)
#  Usage: submit from code/ ->  mkdir -p logs && sbatch vis.sh
#  WARNING: logs/ must exist **before submitting**: #SBATCH --output takes effect before the
#     script runs, and when the directory is missing SLURM discards the log entirely while the
#     job still reports COMPLETED (measured) -- leaving nothing to debug with.
#
#  -- sub-command (SUB, default pred) --------------------------------------
#     SUB=pred  frame-by-frame GT|pred comparison (fwv additionally gets the tf/rollout gap
#               and the [self-check] tf nRMSE)
#               defaults: chunk=9  style=both  FIELDS= all enabled for pure / four fields for fwv
#     SUB=lt    long-term rollout, no GT, streaming (fwv line only)
#               defaults: chunk=10 style=tri   FIELDS= alpha
#
#  -- error rows (DIFF, pred only; empty by default = not rendered, matching old behaviour) --
#     DIFF=abs   Delta = pred - GT, physical units, colour scale adapts to +-p99|Delta|
#                -> shows where the error lives
#     DIFF=pct   Delta% = Delta/S x 100, colour scale fixed at +-100%
#                -> for side-by-side comparison across runs/checkpoints
#     DIFF=both  render both rows (4 rows in total)
#     Companions: PCT_SCALE=range|rms|p99 (the denominator S of Delta%, default range=GT full scale)
#                 DIFF_PCT=99             (percentile for the colour scale of the abs row only)
#                 ROW_H=                  (height of each row in inches; leave empty = 10.0
#                                          automatically when DIFF=both, see below -- the
#                                          default 10.8 x 4 rows would exceed 4096 px)
#     The remaining interfaces stay in vis.py and are typed by hand (no locating logic needed):
#       python vis.py gt    --data_dir <d> --chunks 0-10          # plain data inspection
#       python vis.py align --fw-dir <fw>/output --chunk 9        # registration (before training)
#       python vis.py nofb  --config_path ... --checkpoint ...    # 3 rows for the no-feedback arm
#
#  -- locating ckpt/config -------------------------------------------------
#     Form 1 (preferred): CONFIG=... CKPT=... sbatch vis.sh     explicit, unambiguous
#     Form 2 (fallback):  RUN=runname [TS=timestamp] sbatch vis.sh   resolved under results/train/
#         RUN is required and must not contain '/'; omit TS and the newest timestamp under that
#         RUN is used (by lexicographic order of the directory name)
#
#  -- code deps (all flat inside code/) ------------------------------------
#     vis.py            this entry point (SUB=pred|lt)
#       -> imports schema.py      ChannelSchema (channel derivation)
#       -> imports dataset.py     assemble/reconstruct/resolve_stats (shared with training)
#       -> imports hpm_model.py   HPM
#
#  -- io -------------------------------------------------------------------
#     in : $CONFIG (.hydra/config.yaml)  $CKPT (best.pt)  DATA/PRIOR (read from the config)
#     out: results/vis/$SUB/$FEATURE/  (pred: compare_*_{pred,tf}_{tri,scatter}.mp4;
#          lt: longterm_*_{tri}.mp4) -- videos only.
#          No npy is written by default; for offline analysis pass vis.py --save_rmse
#          (per-frame RMSE, about 1.7 KB each) / --save_preds (the full predicted field,
#          about 0.9 GB per field). The RMSE numbers still go to the log either way.
# ==========================================================================

set -euo pipefail

# Must be submitted from code/: the submit directory must be named code and vis.py must be in cwd.
# Uses $SLURM_SUBMIT_DIR (not $BASH_SOURCE -- sbatch copies to spool, so it would not match).
# The :- guards against set -u.
if [ "$(basename "${SLURM_SUBMIT_DIR:-}")" != "code" ] || [ ! -f "vis.py" ]; then
    echo "ERROR: must be submitted from the project's code/ directory:  cd <...>/models/code && sbatch vis.sh"
    echo "       current submit directory: ${SLURM_SUBMIT_DIR:-<not a SLURM environment>}   cwd: $PWD"
    exit 1
fi
mkdir -p logs                              # a fallback; but #SBATCH --output needs it before the
                                           # script runs at all, so logs/ has to exist before
                                           # submitting (see the banner at the top)
# $REPO is exported by activate.sh below (= the repo root; results/ and data/ are siblings of code/)

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # find conda, activate the environment, and export $REPO

# ---- locate checkpoint / config (explicit CONFIG/CKPT wins, otherwise RUN[/TS]) ----
# RUN = runname (required, no /); TS = timestamp (optional; omitted, the newest one under that RUN
# containing best.pt is used). "Newest" is lexicographic on the directory name, not mtime.
TRAIN_ROOT="$REPO/results/train"
RUN="${RUN:-}"
TS="${TS:-}"

if [ -z "${CONFIG:-}" ] || [ -z "${CKPT:-}" ]; then
    [ -n "$RUN" ] || {
        echo "ERROR: without CONFIG/CKPT you must give RUN=runname (TS=timestamp optional)"; exit 1; }
    case "$RUN" in */*)
        echo "ERROR: RUN must not contain '/' (that is the runname; give the timestamp as TS=...). Current RUN='$RUN'"
        exit 1 ;;
    esac

    RUN_DIR="$TRAIN_ROOT/$RUN"
    [ -d "$RUN_DIR" ] || { echo "ERROR: cannot find $RUN_DIR (RUN misspelled?)"; exit 1; }

    # The directory layout is <overall-direction runname>/[<detail-change override_dirname>/]<timestamp>/checkpoints/best.pt
    # -- a run with CLI overrides gets an extra override_dirname level inserted by hydra, while a
    # run without them has only two levels. Both have to be recognised, so the search is by
    # best.pt rather than by assuming a depth. "Newest" is still lexicographic on the timestamp
    # directory name (they are %Y-%m-%d_%H-%M-%S, where lexicographic == chronological).
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
        if [ -n "$TS" ]; then echo "ERROR: no directory under $RUN_DIR with TS=$TS and checkpoints/best.pt"
        else echo "ERROR: no directory under $RUN_DIR contains checkpoints/best.pt"; fi
        echo "       what does exist:"
        find "$RUN_DIR" -mindepth 2 -maxdepth 4 -path '*/checkpoints/best.pt' 2>/dev/null \
            | sed "s|$TRAIN_ROOT/||; s|/checkpoints/best.pt||; s/^/         /" | head -10
        exit 1
    fi
    if [ -n "$TS" ] && [ "$NCAND" -gt 1 ]; then
        echo "ERROR: TS=$TS matches $NCAND directories under $RUN (different detail-change levels); give CONFIG/CKPT directly:"
        find "$RUN_DIR" -mindepth 2 -maxdepth 4 -path "*/$TS/checkpoints/best.pt" 2>/dev/null \
            | sed 's/^/         /'
        exit 1
    fi
    TS="$TS_BEST"
    echo "[vis.sh] RUN='$RUN' -> $RESOLVED"
    if [ "$NCAND" -gt 1 ]; then
        echo "[vis.sh] (this RUN has $NCAND candidates; taking the one with the newest timestamp)"
    fi

    CONFIG="$TRAIN_ROOT/$RESOLVED/.hydra/config.yaml"
    CKPT="$TRAIN_ROOT/$RESOLVED/checkpoints/best.pt"
    FEATURE="${FEATURE:-$RESOLVED}"        # the output directory mirrors the same path under results/train
else
    echo "[vis.sh] using the explicit CONFIG/CKPT"
    # If the ckpt is itself under results/train, mirror its path so provenance is not lost;
    # only fall back to explicit_<time> when it points somewhere else (a temporary snapshot, say).
    case "$CKPT" in
        "$TRAIN_ROOT"/*/checkpoints/*)
            _rel="${CKPT#"$TRAIN_ROOT/"}"; _rel="${_rel%/checkpoints/*}"
            FEATURE="${FEATURE:-$_rel}" ;;
        *)  FEATURE="${FEATURE:-explicit_$(date +%m%d_%H%M)}" ;;
    esac
fi

[ -f "$CONFIG" ] || { echo "ERROR: config does not exist: $CONFIG"; exit 1; }
[ -f "$CKPT" ]   || { echo "ERROR: ckpt does not exist:   $CKPT";   exit 1; }

# ---- read data_dir / prior_dir / window / enabled channels from the snapshot config (same as training) ----
# The helper registers the repo resolver first (the config uses ${repo:}), otherwise a bare
# OmegaConf.load would raise.
# Output is a single line: DATA PRIOR WINDOW ch1 ch2 ...  (when prior is empty a "-" placeholder
# keeps the positions aligned).
# Capture the output and check the exit code (set -e is unreliable inside process substitution and
# would silently read an empty value).
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
) || { echo "ERROR: failed to read the config (see the traceback above)"; exit 1; }
read -r DATA PRIOR WINDOW ENABLED_CHS <<< "$DP"
[ "$PRIOR" = "-" ] && PRIOR=""
[ -n "$DATA" ] || { echo "ERROR: could not resolve data.dir from the config"; exit 1; }

# ---- SUB: sub-command pred | lt (for gt/align/nofb see the banner at the top; type them by hand) ----
SUB="${SUB:-pred}"
case "$SUB" in pred|lt) ;; *)
    echo "ERROR: SUB supports only pred|lt (for gt/align/nofb type python vis.py ... by hand)"; exit 1 ;;
esac

# Which line, by window (>0 pure / ==0 fwv)
if [ "$WINDOW" -gt 0 ]; then LINE=pure; else LINE=fwv; fi
# lt is fwv-only (it needs a prior; vis.py asserts this internally as well)
if [ "$SUB" = lt ] && [ "$LINE" != fwv ]; then
    echo "ERROR: lt belongs to the fwv line only (it needs a prior); this RUN is pure (window=$WINDOW)"; exit 1; fi

# ---- defaults (overridable by environment variable), split by SUB ----
#   pred: chunk=9  style=both  FIELDS= all enabled for pure / four fields for fwv
#   lt  : chunk=10 style=tri   FIELDS= alpha  (no GT over the long term, so the interface is
#         judged qualitatively)
if [ "$SUB" = lt ]; then
    CHUNK="${CHUNK:-10}"; STYLE="${STYLE:-tri}"; FIELDS="${FIELDS:-alpha}"
else
    CHUNK="${CHUNK:-9}";  STYLE="${STYLE:-both}"
    if [ "$LINE" = pure ]; then
        [ -n "$ENABLED_CHS" ] || { echo "ERROR: no enabled channels resolved from the config for the pure line"; exit 1; }
        FIELDS="${FIELDS:-$ENABLED_CHS}"
    else
        FIELDS="${FIELDS:-alpha Ux Uz p_rgh}"
    fi
fi
NFRAMES="${NFRAMES:-0}"                    # 0=run to the end; set 8 to trigger a quick self-check when validating pred
# lt is a streaming rollout and cannot be rendered twice (vis.py asserts style!=both)
if [ "$SUB" = lt ] && [ "$STYLE" = both ]; then
    echo "ERROR: lt does not support STYLE=both; use tri or scatter"; exit 1; fi

# ---- error rows (pred only; vis.py exposes --diff for pred alone) ----
# When ROW_H is left empty it adapts to the number of rows: dpi is fixed at 100, so pixel height
# = row_h x rows x 100. DIFF=both means 4 rows, and the default 10.8 -> 4320 px, past the 4096
# ceiling that quite a few players hardware-decode up to (vis.py render() only warns and does not
# change the default) -> so it drops to 10.0 = 4000 px here. The single-row abs/pct versions come
# to 3 rows, where 10.8 is only 3240, so they are left alone. An explicit ROW_H is always obeyed.
DIFF="${DIFF:-}"
ROW_H="${ROW_H:-}"
DIFF_ARGS=()
if [ -n "$DIFF" ]; then
    case "$DIFF" in abs|pct|both) ;; *)
        echo "ERROR: DIFF supports only abs|pct|both (empty = do not render error rows). Current DIFF='$DIFF'"
        exit 1 ;;
    esac
    if [ "$SUB" != pred ]; then
        echo "ERROR: DIFF is supported for SUB=pred only (lt has no GT, so there is no Delta to compute)"; exit 1; fi
    DIFF_ARGS=(--diff "$DIFF"
               --pct-scale "${PCT_SCALE:-range}"
               --diff-pct  "${DIFF_PCT:-99}")
    if [ -z "$ROW_H" ] && [ "$DIFF" = both ]; then ROW_H=10.0; fi
fi
# A separate if, not `[ ... ] && ...` -- under set -e the latter exits the whole script when the
# condition is false.
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

# ---- overwrite guard: FORCE=1 overwrites ----
if [ -d "$OUT_ROOT" ] && [ -n "$(ls -A "$OUT_ROOT" 2>/dev/null)" ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "ERROR: $OUT_ROOT/ already has content. Use FORCE=1 to overwrite, or pick another FEATURE=xxx."
        exit 1
    fi
    echo "WARN: FORCE=1, overwriting $OUT_ROOT/"
fi
mkdir -p "$OUT_ROOT"

# ---- one field at a time (--field takes a single field; vis.py appends the style/npy suffix) ----
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
    else   # lt: long-term rollout, no GT, streaming (prior_dir is read from the config by vis.py)
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
