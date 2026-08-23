#!/bin/bash
#SBATCH --job-name=lboei
#SBATCH --partition=eecs
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lbo_eigen_%j.log

# ============================================================
# Paths — update these to match your setup
# ============================================================

# OpenFOAM case directory (contains constant/polyMesh/owner, neighbour, sets/)
CASE_DIR="$HOME/hpc-share/ocean_project/case"

# Cropped cell coordinates (output from crop_fields.py)
# Use the coords.npy that matches your target subdomain
COORDS="$HOME/hpc-share/models/data/3d/cropped_0.05/coords.npy"

# Name of the cellSet used for cropping
CELLSET="subdomainCells"

# Number of eigenvectors to compute
K=128

# Output directory for eigenvectors
OUTPUT_DIR="$HOME/hpc-share/models/data/3d/cropped_0.05/lbo"

_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # 找 conda + 激活环境，并导出 $REPO
cd "$REPO/legacy/hpm"


# ============================================================
# Run
# ============================================================

echo "========================================"
echo "LBO Eigenbasis Precomputation"
echo "========================================"
echo "Case:     $CASE_DIR"
echo "Coords:   $COORDS"
echo "CellSet:  $CELLSET"
echo "k:        $K"
echo "Output:   $OUTPUT_DIR"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Memory:   $SLURM_MEM_PER_NODE"
echo "Date:     $(date)"
echo "========================================"

python -u lbo.py \
    --case "$CASE_DIR" \
    --cellset "$CELLSET" \
    --coords "$COORDS" \
    --k "$K" \
    --output "$OUTPUT_DIR"

echo ""
echo "Exit code: $?"
echo "Finished: $(date)"
