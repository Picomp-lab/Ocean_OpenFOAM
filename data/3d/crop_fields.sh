#!/bin/bash
#SBATCH --job-name=crop_fields
#SBATCH --partition=eecs
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=crop_fields_%j.log

# The crop region is **not a parameter of this script** -- it comes from an OpenFOAM cellSet.
# Changing the region (the spanwise y thickness, say) means editing system/topoSetDict in the
# case first and then running topoSet to generate the cellSet:
#
#   box (-2.5 0.275 -0.41) (16.5 0.325 0.16);   <- the current setting = cropped_0.05
#            ^y lower         ^y upper             the 0.1 setting uses 0.25/0.35, the 0.3
#                                                  setting uses 0.15/0.45
#   topoSet -case "$OCEAN_CASE"
#
# Historically a 0.1 (1,245,500 points) and a 0.3 (3,732,705 points) setting were generated;
# neither ever entered training, and they were deleted on 2026-08-21. The parameters are
# recorded in README section 1.
#
# Both paths come from environment variables -- do not hard-code a personal directory (the
# default for $OCEAN_CASE matches README section 1).
CASE_DIR="${OCEAN_CASE:-$HOME/hpc-share/ocean_project/case}"
OUTPUT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cropped_0.05}"

echo "case  : $CASE_DIR"
echo "output: $OUTPUT_DIR"
[ -d "$CASE_DIR" ] || { echo "ERROR: case directory does not exist; point OCEAN_CASE= at it"; exit 1; }

python -u crop_fields.py \
    --case "$CASE_DIR" \
    --output "$OUTPUT_DIR" \
    --cellset subdomainCells \
    --chunk-size 100
