#!/bin/bash
#SBATCH --job-name=scan_toff
#SBATCH --output=logs/scan_%j.log
#SBATCH --error=logs/scan_%j.err
#SBATCH --partition=eecs            # <- confirmed: CPU partition (prior calibration is pure numpy, no GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00             # <- confirmed: upper-bound guess, tune from measurement (full 1-9 scan)

# ==========================================================================
#  STAGE 1 -- t-offset calibration (per-chunk best_k)
#  Usage: submit from code/ ->  mkdir -p logs && sbatch scan_toffset.sh
#
#  -- code deps (all flat inside code/) ------------------------------------
#     scan_toffset.py   entry point for this stage
#       -> imports lift.py        Nwogu profile formula (CH_NAMES is the single source of
#                                 truth for channel order)
#       -> imports fw_io.py       reading FUNWAVE files (load_static)
#       -> from gen_prior import Bilinear, build_frame   reuse of the projection layer
#
#  -- inputs ---------------------------------------------------------------
#     $FW/{eta,u,v,mask}_NNNNN, dep.out         FUNWAVE native 2D output
#     $DATA/coords.npy                          CFD cell-centre coordinates (N,3)
#     $DATA/chunk_00X_{data,times}.npy  X=1..9  GT (for the calibration comparison)
#
#  -- outputs --------------------------------------------------------------
#     $DATA/toffset_scan/c00X.json  X=1..9      best_k / t_offset + per-channel curves
#     (any old toffset_scan has been renamed out of the way; a clean directory is created
#      by default)
# ==========================================================================

# ---- edit here ----
FW=/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output   # <- confirmed: mind the case! (the TINGKIRBY trap has been hit)
# --data-dir / --out use the script defaults (verified to resolve to ../data/3d/cropped_0.05,
# matching the flat layout)

# ---- environment ----
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # find conda, activate the environment, and export $REPO
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4      # matching cpus-per-task

# ---- run ----
python scan_toffset.py --fw-dir "$FW" --chunks 1-9

# Check three diagnostic gates before moving on to stage 2:
#   1) does any best_k in the summary table at the end sit "on the scan boundary"? -> if so,
#      widen --k-range and rescan
#   2) the Uy row should read "[curve flat, not counted]" (the correct quasi-2D signal)
#   3) spread between channels >2 frames -> x-offset(15.05) may still carry a residual
