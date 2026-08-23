#!/bin/bash
#SBATCH --job-name=hpmfw
#SBATCH --partition=dgxh
#SBATCH --exclude=dgxh-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpm_fw_fb_%j.log

# ============================================================
# fw —— prior + self-feedback 输入分支 (teacher forcing, 单步)
#   输入 = [prior(t) | x_f*m | m],  x_f = GT(t-1),  base = prior(t)
#
# 相对 train_prior.sh (无自反馈) 只多一个变量, 其余全同 —— 单变量对照。
#
# 用法:
#   sbatch fwv/train_fw.sh                          全量 (config_fw.yaml)
#   SMOKE=1 sbatch fwv/train_fw.sh                  冒烟: 单 chunk, 2 epoch, 无 wandb
#   sbatch fwv/train_fw.sh train.cond_dropout=0.2   CLI 覆盖 (记得手改 wandb.name)
#
# 显存: 输入维 3+F 变成 3+2F+1 (F=4: 7 -> 12), 只影响第一层 Linear,
#   backbone 不变 -> 相对 train_prior 增量很小。
# 内存: PriorFeedbackDataset 与 PriorPairDataset 持有同样的张量 (GT + prior),
#   gt_prev 是 gts 的切片视图, 不额外占用。~1.84GB/chunk × 8 ≈ 16.5GB。
#   96G 给的是余量, 实测后可下调。
# ============================================================

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/hpm"


export PYTHONPATH="$PWD:$PYTHONPATH"

echo "========================================"
echo "HPM fw — prior + self-feedback (teacher forcing)"
echo "========================================"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "GPU Mem:   $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:    $(python --version)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "Date:      $(date)"
echo "Overrides: ${@:-<none>}"
echo "========================================"

# ---- 前置检查: prior 数据必须存在 ----
# 注: 必须用 OmegaConf.select 只解析这一个键。to_container(resolve=True) 会
#     解析整个配置, 撞上 hydra.run.dir 里的 ${now:} —— 那是 Hydra 运行时才
#     注册的 resolver, 脱离 Hydra 直接加载会抛 UnsupportedInterpolationType。
PRIOR_DIR=$(python -c "
from omegaconf import OmegaConf
print(OmegaConf.select(OmegaConf.load('fwv/config_fw.yaml'), 'data.prior_dir'))" 2>/dev/null)

if [ -z "$PRIOR_DIR" ]; then
    echo "WARN: 无法从 config_fw.yaml 解析 data.prior_dir, 跳过前置检查"
elif [ ! -d "$PRIOR_DIR" ]; then
    echo "ERROR: prior 目录不存在: $PRIOR_DIR"
    echo "       先跑 fwv/scan_toffset.sh 再 fwv/gen_prior.sh"
    exit 1
else
    NC=$(ls "$PRIOR_DIR"/prior_*_data.npy 2>/dev/null | wc -l)
    echo "prior dir: $PRIOR_DIR  (${NC} chunks)"
    if [ "$NC" -eq 0 ]; then
        echo "ERROR: 目录存在但没有 prior_*_data.npy"
        exit 1
    fi
fi
echo "========================================"

if [ "${SMOKE}" = "1" ]; then
    echo "[SMOKE] 单 chunk / 2 epoch / 无 wandb —— 只验管线, 不看结果"
    python -u fwv/train_fw.py \
        data.train_chunk_range=[6,6] data.val_chunk_range=[9] \
        train.epochs=2 train.num_workers=0 \
        wandb.enabled=false "$@"
else
    python -u fwv/train_fw.py "$@"
fi

echo "Done: $(date)"