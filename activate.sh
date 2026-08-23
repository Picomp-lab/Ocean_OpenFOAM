#!/bin/bash
# ============================================================
# activate.sh — 找到 conda 并激活本项目的环境。被各 sbatch 脚本 source。
#
# 脚本里这么用（三行，与自己在第几层无关；sbatch 会把脚本拷到 spool，
# 所以不能靠 $0 定位，得用 sbatch 设的 $SLURM_SUBMIT_DIR）：
#
#   _d="${SLURM_SUBMIT_DIR:-$PWD}"
#   while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
#   source "$_d/activate.sh"
#
# source 之后可用 $REPO（仓库根）。环境位置优先级：
#   1. $OCEAN_ENV                              显式指定
#   2. <repo>/.env.local 里的 OCEAN_ENV=       setup.sh 生成，不进版本库
#   3. /nfs/hpc/share/$USER/.conda/envs/ocean  集群上的默认位置
#
# 这里不写死任何人的家目录或用户名 —— 换个账号 clone 下来照样跑。
# ============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO

# 集群共享的 conda（不是个人路径）。别处可以 CONDA_SH= 覆盖。
CONDA_SH="${CONDA_SH:-/nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ]; then
  echo "activate.sh: 找不到 conda（$CONDA_SH）。先跑 $REPO/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

[ -f "$REPO/.env.local" ] && source "$REPO/.env.local"
OCEAN_ENV="${OCEAN_ENV:-/nfs/hpc/share/$USER/.conda/envs/ocean}"

if [ ! -x "$OCEAN_ENV/bin/python" ]; then
  echo "activate.sh: 环境不存在: $OCEAN_ENV" >&2
  echo "            先跑 $REPO/setup.sh（或 OCEAN_ENV=<你的环境> 再来）" >&2
  return 1 2>/dev/null || exit 1
fi
conda activate "$OCEAN_ENV"
