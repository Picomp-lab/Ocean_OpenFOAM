#!/bin/bash
#SBATCH --job-name=hpmfwss
#SBATCH --partition=dgxh
#SBATCH --exclude=dgxh-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/hpm_fw_ss_%j.log

# ============================================================
# fw —— prior + self-feedback, R 步真 BPTT + scheduled sampling
#   r=0     x_f=0, m=0 冷启动
#   r>=1    掷骰子: prob p 喂 GT(t+r-1) 断链 / 否则喂 pred(t+r-1) 保链
#   base = prior(t+r);  pred = prior + Δ;  loss 用 gt(t+r);  R 步等权
#   p 从 ss_p_start 退到 ss_p_end, 占前 ss_anneal_frac 的 epochs
#
# 治的是 exposure bias (部署 rollout 近岸误差积累)。cold start 从头训。
# 与单步 baseline (train_fw.py) 并存, 作单变量对照 —— 不覆盖旧文件。
#
# 用法:
#   sbatch fwv/train_fw_ss.sh                        全量, R=4, p 1.0->0.1
#   SMOKE=1 sbatch fwv/train_fw_ss.sh                冒烟: 单 chunk 2 epoch 无 wandb
#   sbatch fwv/train_fw_ss.sh +train.R=6             CLI 覆盖 SS 超参 (记得改 wandb.name)
#
# ⚠️ 显存: 真 BPTT 留 R 份 forward 图 (bs=1 × 574k cells)。
#   先 SMOKE 跑 2 epoch 看 mem=xx/xxGB; 爆了先加 model.use_ckpt=true。
# ============================================================

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/hpm"


export PYTHONPATH="$PWD:$PYTHONPATH"

# SS 超参默认值 (可被 CLI +train.xxx 覆盖); 也在 train_fw_ss.py 里有同名默认
R="${R:-4}"
P_START="${P_START:-1.0}"
P_END="${P_END:-0.1}"
ANNEAL="${ANNEAL:-0.6}"
RUN_NAME="${RUN_NAME:-hpm_fw_ss_R${R}}"

echo "========================================"
echo "HPM fw — scheduled sampling + 真 BPTT (R=${R}, cold start)"
echo "========================================"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "GPU Mem:   $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:    $(python --version)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "R=${R}  p ${P_START}->${P_END}  anneal_frac=${ANNEAL}  run=${RUN_NAME}"
echo "Date:      $(date)"
echo "Overrides: ${@:-<none>}"
echo "========================================"

# ---- 前置检查: prior 数据必须存在 (同 train_fw.sh, 用 OmegaConf.select) ----
PRIOR_DIR=$(python -c "
from omegaconf import OmegaConf
print(OmegaConf.select(OmegaConf.load('fwv/config_fw.yaml'), 'data.prior_dir'))" 2>/dev/null)

if [ -z "$PRIOR_DIR" ]; then
    echo "WARN: 无法解析 data.prior_dir, 跳过前置检查"
elif [ ! -d "$PRIOR_DIR" ]; then
    echo "ERROR: prior 目录不存在: $PRIOR_DIR"
    echo "       先跑 fwv/scan_toffset.sh 再 fwv/gen_prior.sh"
    exit 1
else
    NC=$(ls "$PRIOR_DIR"/prior_*_data.npy 2>/dev/null | wc -l)
    echo "prior dir: $PRIOR_DIR  (${NC} chunks)"
    [ "$NC" -eq 0 ] && { echo "ERROR: 目录存在但没有 prior_*_data.npy"; exit 1; }
fi
echo "========================================"

# SS 超参用 +train.xxx (config_fw.yaml 里没有这些键, + 是 hydra 新增语法)
SS_ARGS="+train.R=${R} +train.ss_p_start=${P_START} +train.ss_p_end=${P_END} +train.ss_anneal_frac=${ANNEAL}"

if [ "${SMOKE}" = "1" ]; then
    echo "[SMOKE] 单 chunk / 2 epoch / 无 wandb —— 验管线 + 看显存, 不看结果"
    python -u fwv/train_fw_ss.py \
        $SS_ARGS \
        data.train_chunk_range=[6,6] data.val_chunk_range=[9] \
        train.epochs=2 train.num_workers=0 train.batch_size=1 \
        wandb.enabled=false "$@"
else
    python -u fwv/train_fw_ss.py \
        $SS_ARGS \
        train.batch_size=1 \
        wandb.name=${RUN_NAME} "$@"
fi

echo "Done: $(date)"
