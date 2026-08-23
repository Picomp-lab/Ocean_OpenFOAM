#!/bin/bash
#SBATCH --job-name=hpm1b
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpm_prior1b_%j.log

# ============================================================
# 1b capability check —— prior 场 -> CFD 场 的单步映射
#   输入 = prior(t), base = prior(t), 无时间窗、无自反馈、R=1
#
# 用法:
#   sbatch train_prior.sh                        全量 (config_prior.yaml)
#   实测 ~320s/200样本 -> 7 chunk 训练约 20min/epoch, 50 epoch 约 17h
#   SMOKE=1 sbatch train_prior.sh                冒烟: 单 chunk, 2 epoch, 无 wandb
#   sbatch train_prior.sh model.n_hidden=256     CLI 覆盖 (注意手动改 wandb.name)
#
# 显存: 相比 E0 (window=6, F=6, R=4) 大幅下降 ——
#   R=4->1 省约 4x (BPTT 不再保留四步), F=6->4, 输入维 21->7。
#   估计 12-18GB, 故不再 --exclude=dgxh-1 (40G H100 够用, 节点池翻倍)。
#   若实测 OOM: 取消下面一行注释后重投。
# #SBATCH --exclude=dgxh-1
#
# 内存: PriorPairDataset 同时持有 GT 与 prior (各 4 通道) ->
#   ~1.84GB/chunk, 8 train + 1 val ≈ 16.5GB, 加载峰值再多 ~3GB。64G 充裕。
# ============================================================

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/hpm"


export PYTHONPATH="$PWD:$PYTHONPATH"

echo "========================================"
echo "HPM capability check — 1b (prior-only, single step)"
echo "========================================"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "GPU Mem:  $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:   $(python --version)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "Date:     $(date)"
echo "Overrides: ${@:-<none>}"
echo "========================================"

# ---- 前置检查: prior 数据必须存在 ----
# 注: 必须用 OmegaConf.select 只解析这一个键。to_container(resolve=True) 会
#     解析整个配置, 撞上 hydra.run.dir 里的 ${now:} —— 那是 Hydra 运行时才
#     注册的 resolver, 脱离 Hydra 直接加载会抛 UnsupportedInterpolationType。
PRIOR_DIR=$(python -c "
from omegaconf import OmegaConf
print(OmegaConf.select(OmegaConf.load('fwv/config_prior.yaml'), 'data.prior_dir'))" 2>/dev/null)

if [ -z "$PRIOR_DIR" ]; then
    echo "WARN: 无法从 config_prior.yaml 解析 data.prior_dir, 跳过前置检查"
elif [ ! -d "$PRIOR_DIR" ]; then
    echo "ERROR: prior 目录不存在: $PRIOR_DIR"
    echo "       先跑 gen_prior.sh 生成 prior 数据"
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
    python -u fwv/train_prior.py \
        data.train_chunk_range=[6,6] data.val_chunk_range=[9] \
        train.epochs=2 train.num_workers=0 \
        wandb.enabled=false "$@"
else
    python -u fwv/train_prior.py "$@"
fi

echo "Done: $(date)"