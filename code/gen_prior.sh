#!/bin/bash
#SBATCH --job-name=gen_prior
#SBATCH --output=logs/gen_%j.log
#SBATCH --error=logs/gen_%j.err
#SBATCH --partition=eecs            # <- confirmed: CPU partition (pure numpy, no GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00             # <- confirmed: upper-bound guess (574163 cells over the whole domain x 9 chunks, serial)

# ==========================================================================
#  STAGE 2 -- generate the prior chunk by chunk (serial)
#  Usage: once stage 1 has finished ->  sbatch gen_prior.sh
#         chained: sbatch --dependency=afterok:<scan_jobid> gen_prior.sh
#
#  -- code deps (all flat inside code/) ------------------------------------
#     gen_prior.py      entry point for this stage
#       -> imports lift.py        Nwogu profile formula (writes 5 channels
#                                 [alpha,Ux,Uy,Uz,p_rgh])
#       -> imports fw_io.py       reading FUNWAVE files (load_static)
#
#  -- depends on STAGE 1 ---------------------------------------------------
#     $DATA/toffset_scan/c00X.json   <- this script reads t_offset from it rather than
#                                       copying best_k by hand
#
#  -- inputs ---------------------------------------------------------------
#     $FW/{eta,u,v,mask}_NNNNN, dep.out         FUNWAVE native 2D output
#     $DATA/coords.npy                          CFD cell-centre coordinates (N,3)
#     $DATA/chunk_00X_times.npy  X=1..9         determines t_cfd -> FUNWAVE frame number
#
#  -- outputs --------------------------------------------------------------
#     $DATA/prior_ktuned/prior_00X_data.npy   (T,N,5) float32  [fed to HPM]
#     $DATA/prior_ktuned/prior_00X_valid.npy  (T,N)   bool     [diagnostic]
#     $DATA/prior_ktuned/prior_00X_times.npy  (T,)             t_cfd
#     $DATA/prior_ktuned/prior_00X_meta.json                   self-describing metadata
#     (np.save overwrites a file of the same name; prior_ktuned does not need renaming first)
#
#  -- NOTE: the output has 5 channels including Uy; downstream dataset.py drops Uy and takes
#     4 channels at load time, which is not this script's concern
# ==========================================================================

# ---- edit here ----
FW=/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output   # <- confirmed: mind the case!
DATA=../data/3d/cropped_0.05
XOFF=15.05                          # <- not calibrated, treated as a known input (confirmed to stay fixed)

# ---- environment ----
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"          # find conda, activate the environment, and export $REPO
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---- run: chunk by chunk, reading t_offset straight from the scan JSON (single source of
#      truth, zero transcription) ----
for c in 1 2 3 4 5 6 7 8 9; do
  cid=$(printf "%03d" "$c")
  json=$DATA/toffset_scan/c${cid}.json
  if [ ! -f "$json" ]; then
    echo "[err] chunk $c: missing $json (stage 1 did not produce it?) -> skipping"; continue
  fi
  toff=$(python3 -c "import json,sys; v=json.load(open('$json'))['t_offset']; print('' if v is None else v)")
  if [ -z "$toff" ]; then
    echo "[err] chunk $c: t_offset is empty (the scan found no valid best_k) -> skipping"; continue
  fi
  echo ">>> chunk $c  t_offset=$toff"
  python gen_prior.py --fw-dir "$FW" \
      --coords   "$DATA/coords.npy" \
      --gt-times "$DATA/chunk_${cid}_times.npy" \
      --chunk "$c" --x-offset "$XOFF" --t-offset "$toff" \
      --out "$DATA/prior_ktuned"
done

python gen_prior.py --fw-dir "$FW" \
    --coords   $DATA/coords.npy \
    --gt-times $DATA/chunk_010_times.npy \
    --chunk 10 --x-offset 15.05 --t-offset 0.0 \
    --out $DATA/prior_ktuned
