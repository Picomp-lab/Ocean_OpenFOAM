#!/bin/bash
#SBATCH --job-name=crop_fields
#SBATCH --partition=eecs
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=crop_fields_%j.log

# 裁剪区域**不是这个脚本的参数** —— 它来自 OpenFOAM 的 cellSet。要换区域（比如换
# 展向 y 的厚度）得先改算例里的 system/topoSetDict，再跑 topoSet 生成 cellSet：
#
#   box (-2.5 0.275 -0.41) (16.5 0.325 0.16);   ← 当前这档 = cropped_0.05
#            ↑y下界            ↑y上界              0.1 档用 0.25/0.35，0.3 档用 0.15/0.45
#   topoSet -case "$OCEAN_CASE"
#
# 历史上生成过 0.1（1,245,500 点）和 0.3（3,732,705 点）两档，从没进过训练，
# 2026-08-21 删了，参数记录见 README §1。
#
# 两个路径都走环境变量，别写死个人目录（$OCEAN_CASE 的默认值与 README §1 一致）。
CASE_DIR="${OCEAN_CASE:-$HOME/hpc-share/ocean_project/case}"
OUTPUT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cropped_0.05}"

echo "case  : $CASE_DIR"
echo "output: $OUTPUT_DIR"
[ -d "$CASE_DIR" ] || { echo "ERROR: 算例目录不存在，用 OCEAN_CASE= 指过去"; exit 1; }

python -u crop_fields.py \
    --case "$CASE_DIR" \
    --output "$OUTPUT_DIR" \
    --cellset subdomainCells \
    --chunk-size 100
