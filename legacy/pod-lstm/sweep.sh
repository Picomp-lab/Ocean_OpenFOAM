#!/bin/bash
# ============================================================
# LSTM POD Hyperparameter Random Search
# ============================================================
# Randomly samples hyperparameter combinations and submits
# each as a separate SLURM job. Results tracked via wandb.
#
# Usage: bash sweep.sh [N_RUNS]
#   N_RUNS: number of random combinations (default: 25)
# ============================================================

N_RUNS=${1:-25}
POD_DIR=$OCEAN_DATA/pod_results
BASE_OUTPUT=$OCEAN_DATA/lstm_sweep
SCRIPT=$OCEAN_DATA/lstm_pod.py

# Search space
WINDOWS=(10 20 40 60 80)
HIDDENS=(32 64 96 128)
DROPOUTS=(0.1 0.2 0.3 0.5)
LRS=(0.0005 0.001 0.002)
LAYERS=(1 2 3 4)

mkdir -p ${BASE_OUTPUT}

echo "============================================================"
echo "Launching ${N_RUNS} random hyperparameter runs"
echo "============================================================"

declare -A SEEN
count=0
attempts=0
max_attempts=$((N_RUNS * 10))

while [ ${count} -lt ${N_RUNS} ] && [ ${attempts} -lt ${max_attempts} ]; do
    attempts=$((attempts + 1))

    # Random sample from each array
    W=${WINDOWS[$((RANDOM % ${#WINDOWS[@]}))]}
    H=${HIDDENS[$((RANDOM % ${#HIDDENS[@]}))]}
    D=${DROPOUTS[$((RANDOM % ${#DROPOUTS[@]}))]}
    LR=${LRS[$((RANDOM % ${#LRS[@]}))]}
    L=${LAYERS[$((RANDOM % ${#LAYERS[@]}))]}

    RUN_NAME="w${W}_h${H}_d${D}_lr${LR}_l${L}"

    # Skip duplicates
    if [ -n "${SEEN[${RUN_NAME}]}" ]; then
        continue
    fi
    SEEN[${RUN_NAME}]=1
    count=$((count + 1))
    i=${count}
    RUN_DIR="${BASE_OUTPUT}/${RUN_NAME}"
    mkdir -p ${RUN_DIR}

    echo "  [${i}/${N_RUNS}] ${RUN_NAME}"

    sbatch --job-name="sw_${RUN_NAME}" \
           --partition=eecs \
           --output="${RUN_DIR}/slurm_%j.log" \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=8G \
           --time=00:30:00 \
           <<EOF
#!/bin/bash
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO

# POD / LSTM 那条线的数据在仓库外（case 23 G、各 *_results 共几 G），没进版本库。
OCEAN_DATA="${OCEAN_DATA:-$HOME/hpc-share/ocean_project}"

# Temporarily patch dropout and num_layers for this run
sed "s/DROPOUT = .*/DROPOUT = ${D}/" ${SCRIPT} > ${RUN_DIR}/lstm_run.py
sed -i "s/NUM_LAYERS = .*/NUM_LAYERS = ${L}/" ${RUN_DIR}/lstm_run.py

WANDB_PROJECT=pod-lstm-sweep python ${RUN_DIR}/lstm_run.py \
    --pod_dir ${POD_DIR} \
    --output ${RUN_DIR} \
    --window ${W} \
    --hidden ${H} \
    --lr ${LR}
EOF

    # Small delay to avoid overwhelming scheduler
    sleep 0.5
done

echo "============================================================"
echo "All ${N_RUNS} jobs submitted!"
echo "Monitor: squeue -u \$USER"
echo "Results: wandb.ai/cassan-osu/pod-lstm"
echo "============================================================"