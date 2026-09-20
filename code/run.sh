#!/bin/bash
#SBATCH --job-name=hpm
#SBATCH --partition=dgxh
#SBATCH --exclude=dgxh-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# ============================================================
# run.sh -- single-job SLURM script
#
# Usage:
#   cd <repo>/code
#   mkdir -p logs
#   sbatch run.sh
#   sbatch run.sh rollout.R=8
#   sbatch run.sh pure
#   sbatch run.sh pure data.channels.5.enabled=false
#
# Overriding SLURM parameters for one submission:
#   sbatch --partition=ampere --time=02:00:00 run.sh
#   sbatch --job-name=hpm_pure run.sh pure
#
# First thing to look at after a crash:
#   sacct -j <jobid> --format=JobID,State,ExitCode,Reason
# ============================================================

set -euo pipefail

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # find conda, activate the environment, and export $REPO

# ---- the `pure` shortcut: the pure HPM line ----
# window=6 -> the base becomes the last frame of the window; feedback must be off
# ss=false -> p is constantly 0, pred is always fed
# Uy / nut turned on
# alphaU turned off
if [[ "${1:-}" == "pure" ]]; then
    shift
    set -- \
        data.window=6 \
        rollout.feedback=none \
        rollout.ss=false \
        data.channels.1.alpha_weighted=false \
        data.channels.3.alpha_weighted=false \
        data.channels.2.enabled=true \
        data.channels.5.enabled=true \
        data.channels.5.loss_weight=0.1 \
        "$@"
fi

# unbuffered stdout
export PYTHONUNBUFFERED=1

python train.py "$@"
