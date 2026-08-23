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
# run.sh — 单 job SLURM 脚本
#
# 使用:
#   cd <repo>/code
#   mkdir -p logs
#   sbatch run.sh
#   sbatch run.sh rollout.R=8
#   sbatch run.sh pure
#   sbatch run.sh pure data.channels.5.enabled=false
#
# 临时覆盖 SLURM 参数:
#   sbatch --partition=ampere --time=02:00:00 run.sh
#   sbatch --job-name=hpm_pure run.sh pure
#
# 崩了先看:
#   sacct -j <jobid> --format=JobID,State,ExitCode,Reason
# ============================================================

set -euo pipefail

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

# ---- `pure` 快捷方式: 纯 HPM 线 ----
# window=6 -> 基座变成窗口末帧; feedback 必须关
# ss=false -> p 恒 0, 恒喂 pred
# Uy / nut 打开
# αU 关闭
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

# stdout 不缓冲 (unbuffered stdout)
export PYTHONUNBUFFERED=1

python train.py "$@"