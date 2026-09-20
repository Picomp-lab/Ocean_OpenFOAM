#!/bin/bash
# ============================================================
# setup.sh -- run this one script after cloning: install the environment / find what is missing
#
#   ./setup.sh              # one command, end to end: installs whatever is missing
#   ./setup.sh --check      # probe only, install nothing; a non-zero exit means something is missing
#
# Five steps in one pass: platform / conda environment / python packages / wandb /
# directories and data.
# A missing environment or FUNWAVE-TVD is installed automatically, and a large package dropped
# into archive/ is unpacked automatically (that whole step is delegated to
# archive/restore.sh -- manifest parsing and unpacking logic exist in that one place only).
# **Not one package in archives.tsv may be absent** -- content not in place and no package under
# archive/ counts as a missing part and the script exits non-zero (it still finishes every check
# it can make in this pass before exiting, rather than stopping halfway).
#
# Environment variables (all have defaults; normally none need setting):
#   OCEAN_ENV=/path/to/env   conda environment location (default under "step 2" below)
#   OCEAN_ARCHIVE=/path      where the large .tar packages live (default <repo root>/archive)
#   SRUN_WAIT=90             seconds to wait for a compute node; set 0 to keep everything on the
#                            login node
#
# Probing and pip installation go by default through srun onto a compute node in the share
# partition (usually queued for a few tens of seconds; not preempt -- that one gets preempted and
# requeued). On a login node RLIMIT_NPROC is 400 per user and shared across all users, so
# importing numpy fails when OpenBLAS cannot start threads and a perfectly good package is judged
# broken.
#
# The dependency list is in requirements.txt -- add packages there; no package name is hard-coded
# here. **Runs on the cluster only**; on any other machine it errors out immediately.
#
# This script does three things only: probe, install, warn. It touches no data and submits no jobs.
# ============================================================

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- arguments ----
# One switch only. Everything else is automatic: install what is missing, warn about what cannot
# be installed.
DO_INSTALL=1
for a in "$@"; do
  case "$a" in
    --check)     DO_INSTALL=0 ;;
    -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a (only --check is supported; --help for usage)"; exit 2 ;;
  esac
done

# ---- output ----
if [ -t 1 ]; then R=$'\e[31m' G=$'\e[32m' Y=$'\e[33m' B=$'\e[1m' N=$'\e[0m'
else R= G= Y= B= N=; fi
MISSING=0 WARNED=0
ok()   { printf '  %s✔%s %s\n' "$G" "$N" "$*"; }
bad()  { printf '  %s✘%s %s\n' "$R" "$N" "$*"; MISSING=$((MISSING+1)); }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; WARNED=$((WARNED+1)); }
info() { printf '    %s\n' "$*"; }
# py_run is called inside $( ), where an ordinary warn would be swallowed as part of the probe
# result -- this one goes to stderr
warn_raw() { printf '  %s!%s %s\n' "$Y" "$N" "$*" >&2; }
step() { printf '\n%s== %s ==%s\n' "$B" "$*" "$N"; }

# ============================================================
# 1. which machine is this
# ============================================================
step "1/5 platform"
ON_CLUSTER=0
CONDA_SH_CLUSTER=/nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
[ -f "$CONDA_SH_CLUSTER" ] && [ -d /nfs/hpc/share ] && ON_CLUSTER=1

if [ "$ON_CLUSTER" = 1 ]; then
  ok "OSU College of Engineering cluster ($(hostname -s))"
  case "$(hostname -s)" in
    submit*) info "Login node. Send heavy work to sbatch -- there is a ulimit -v of about 15 GB on address space here" ;;
  esac
  command -v sbatch >/dev/null 2>&1 \
    && ok "SLURM available ($(sbatch --version 2>/dev/null))" \
    || warn "sbatch is not on PATH -- that is what a non-interactive ssh looks like; wrap the command in bash -lc"
else
  printf '  %s✘%s not on the OSU cluster (%s %s)\n' "$R" "$N" "$(uname -s)" "$(hostname -s)"
  cat <<'ERR'

    This only runs on the OSU College of Engineering cluster:
      - the 152 GB of data lives under /nfs/hpc/share; it is not here and cannot be copied over
      - the training / rendering scripts are SLURM scripts (sbatch, partitions, --gres)
      - torch needs cu130, matched to the cluster's H100 / H200 / A40 machines

    ssh over first, then run it:
      ssh <user>@submit.hpc.engr.oregonstate.edu
      cd ~/hpc-share/models && ./setup.sh

    To work on the frontend only (code/web-demo/web/), this script is not needed:
    just cd code/web-demo/web && npm install && npm run dev.
ERR
  exit 1
fi

# ============================================================
# 2. conda environment
# ============================================================
step "2/5 conda environment"

CONDA_SH=""
for c in "${CONDA_SH_OVERRIDE:-}" "$CONDA_SH_CLUSTER" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniforge3/etc/profile.d/conda.sh"; do
  [ -n "$c" ] && [ -f "$c" ] && { CONDA_SH="$c"; break; }
done
if [ -z "$CONDA_SH" ] && command -v conda >/dev/null 2>&1; then
  cand="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
  [ -f "$cand" ] && CONDA_SH="$cand"
fi

if [ -z "$CONDA_SH" ]; then
  bad "cannot find conda"
  info "on the cluster it is at $CONDA_SH_CLUSTER"
  info "elsewhere, install a miniforge, or CONDA_SH_OVERRIDE=/path/to/conda.sh ./setup.sh"
  exit 1
fi
ok "conda: $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"

# Environment location: on the cluster it goes under /nfs/hpc/share (home has a quota that torch
# will not fit in); elsewhere the default envs directory is used
if [ -n "${OCEAN_ENV:-}" ]; then ENV_PREFIX="$OCEAN_ENV"
elif [ "$ON_CLUSTER" = 1 ];  then ENV_PREFIX="/nfs/hpc/share/$USER/.conda/envs/ocean"
else                              ENV_PREFIX="$(conda info --base)/envs/ocean"
fi

if [ -x "$ENV_PREFIX/bin/python" ]; then
  ok "environment exists: $ENV_PREFIX"
elif [ "$DO_INSTALL" = 0 ]; then
  bad "environment does not exist: $ENV_PREFIX (drop --check and it is created automatically)"
else
  echo "  creating the environment (python 3.10 + ffmpeg, a few minutes): $ENV_PREFIX"
  mkdir -p "$(dirname "$ENV_PREFIX")"
  conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.10 ffmpeg || {
    bad "conda create failed"; exit 1; }
  ok "environment created"
fi

if [ -x "$ENV_PREFIX/bin/python" ]; then
  conda activate "$ENV_PREFIX" || { bad "activate failed: $ENV_PREFIX"; exit 1; }
  ok "python $("$ENV_PREFIX/bin/python" -V 2>&1 | awk '{print $2}')"
  PY="$ENV_PREFIX/bin/python"

  # Write the environment location into .env.local (not committed) -- activate.sh reads it, so no
  # sbatch script has to hard-code anyone's path.
  if [ ! -f "$REPO/activate.sh" ]; then
    bad "activate.sh is missing -- it should be in the repo; every sbatch script uses it to find the environment"
  elif [ "$DO_INSTALL" = 1 ]; then
    printf 'OCEAN_ENV=%s\n' "$ENV_PREFIX" > "$REPO/.env.local"
    ok ".env.local written (activate.sh reads it to locate the environment)"
  elif [ -f "$REPO/.env.local" ]; then
    ok ".env.local: $(sed -n 's/^OCEAN_ENV=//p' "$REPO/.env.local")"
  else
    warn "no .env.local -- sbatch scripts will fall back to /nfs/hpc/share/\$USER/.conda/envs/ocean"
    info "run once without --check and it will be generated"
  fi
else
  PY=""
fi

# ---- send the heavy work to a compute node ----
# On a login node RLIMIT_NPROC is 400 per user and shared across all users, so OpenBLAS cannot
# start threads while importing a large package and a perfectly good package is judged broken. A
# compute node has a couple of hundred thousand nproc, no address-space limit, and outbound
# network for installing packages.
# Only share is used, **never preempt** -- preempt gets preempted and requeued, turning a check
# that should take tens of seconds into being kicked off halfway and rescheduled, for no gain.
# When share cannot be scheduled (QOSGrpCpuLimit is a group-wide quota, so a full group means
# PENDING), it falls back to the login node after SRUN_WAIT seconds. SRUN_WAIT=0 forces everything
# to stay on the login node.
USE_SRUN=0
if [ "${SRUN_WAIT:-90}" -gt 0 ] && command -v srun >/dev/null 2>&1; then USE_SRUN=1; fi
SRUN_BASE=(-p share -n1 -c2 -J hpc_setup)
SRUN_WAIT=${SRUN_WAIT:-90}           # fall back to the login node rather than leaving someone
                                     # hanging here (the local path can already tell "not
                                     # measured reliably" and will not report a false failure)

# Single-threaded environment used when falling back to the login node
NOTHREAD=(env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
          NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1)

# Run python on a compute node; stdin is the script and $@ are its arguments. If that fails, fall
# back to the login node.
py_run() {
  local script args a out rc
  script=$(cat); args=""
  for a in "$@"; do args="$args $(printf '%q' "$a")"; done
  if [ "$USE_SRUN" = 1 ]; then
    out=$(printf '%s' "$script" | timeout "$SRUN_WAIT" srun "${SRUN_BASE[@]}" --mem=4G -t 00:10:00 \
          bash -lc "source '$CONDA_SH' && conda activate '$ENV_PREFIX' && exec python -$args" 2>/dev/null)
    rc=$?
    if [ $rc -eq 0 ] && [ -n "$out" ]; then printf '%s\n' "$out"; return 0; fi
    if [ $rc -eq 124 ]; then
      warn_raw "the share partition did not schedule within ${SRUN_WAIT}s (most likely QOSGrpCpuLimit); falling back to the login node (single-threaded)"
    else
      warn_raw "did not run on a compute node (srun rc=$rc); falling back to the login node (single-threaded)"
    fi
  fi
  printf '%s' "$script" | "${NOTHREAD[@]}" "$PY" - "$@" 2>/dev/null
}

# The same, for pip. Output goes straight out; nothing is captured.
pip_run() {
  local args a
  args=""
  for a in "$@"; do args="$args $(printf '%q' "$a")"; done
  if [ "$USE_SRUN" = 1 ]; then
    srun "${SRUN_BASE[@]}" --mem=8G -t 00:40:00 \
      bash -lc "source '$CONDA_SH' && conda activate '$ENV_PREFIX' && exec python -m pip$args" && return 0
    warn "could not install on a compute node; retrying on the login node"
  fi
  "$PY" -m pip "$@"
}

# ============================================================
# 3. python packages
# ============================================================
step "3/5 python packages"

# Package names and versions all live in requirements.txt; the script itself hard-codes no package
# -- to add a dependency, edit that file.
# The torch family is in there too, relying on the --extra-index-url in the file to point at
# pytorch's cu130 index.
REQ="$REPO/requirements.txt"
[ -f "$REQ" ] || { bad "${REQ#"$REPO"/} is missing -- this file should be in the repo"; exit 1; }

if [ -z "$PY" ]; then
  bad "no usable python; skipping the package probe"
else
  [ "$USE_SRUN" = 1 ] && echo "  probing on a compute node (share partition) -- usually queued for tens of seconds; falls back locally if it cannot be scheduled"

  # One srun / one python process probes all of requirements, and reports torch's cuda version and
  # whether the torch/torchvision builds match. Output:
  #   OK|import name|version     importable
  #   MISSING|import name|pip line   not installed
  #   BROKEN|import name|explanation|pip name   installed but import raised
  #   INFO|text                  printed directly
  #   TVDIFF|text                torch and torchvision builds do not match
  probe_pkgs() {
  need_pip=0
  probe=$(py_run "$REQ" <<'PYEOF'
import importlib, importlib.metadata as md, re, sys

# numpy is imported first, to establish the machine's resources up front: a login node has an
# RLIMIT_NPROC of only 400 shared across all users, and when it is full numpy's OpenBLAS cannot
# start threads and the C extension simply will not come up. Hitting that here lets every later
# import failure be labelled "not measured reliably" rather than "the package is broken".
import socket
print("WHERE|%s" % socket.gethostname())
try:
    import numpy  # noqa: F401
except Exception as e:
    # numpy itself will not start -- what a login node looks like when its resources are
    # exhausted. Every later import failure is then void, and the bash side marks the result
    # "not measured reliably" when it sees this line.
    print("NUMPYFAIL|%s" % str(e).replace("|", "/")[:80])

# The few whose pip name != import name
ALIAS = {"pillow": "PIL", "pyyaml": "yaml", "hydra-core": "hydra",
         "polars-lts-cpu": "polars", "imageio-ffmpeg": "imageio_ffmpeg"}

for path in sys.argv[1:]:
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):      # blank line / comment / --index-url
            continue
        dist = re.split(r"[=<>!~;\[ ]", line)[0].strip()
        imp = ALIAS.get(dist.lower(), dist.lower().replace("-", "_"))
        try:
            m = importlib.import_module(imp)
            ver = getattr(m, "__version__", None)
            if ver is None:                       # things like xxhash have no __version__
                try:
                    ver = md.version(dist)
                except md.PackageNotFoundError:
                    ver = "?"
            print("OK|%s|%s||%s" % (imp, ver, path))
        except Exception as e:
            msg = str(e).replace("|", "/")[:90]
            try:
                v = md.version(dist)
                print("BROKEN|%s|%s installed (%s) but import raised %s: %s|%s|%s"
                      % (imp, dist, v, type(e).__name__, msg, dist, path))
            except md.PackageNotFoundError:
                print("MISSING|%s|%s||%s" % (imp, line, path))

# torch's cuda / GPU situation; also whether torch and torchvision are from the same build
# (the +cu130 local version only exists in the module's __version__; the dist metadata is
# stripped of it)
try:
    import torch
    print("INFO|torch %s / cuda %s / GPU visible: %s"
          % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
    try:
        import torchvision
        a = torch.__version__.partition("+")[2] or "none"
        b = torchvision.__version__.partition("+")[2] or "none"
        if a != b:
            print("TVDIFF|torch=%s torchvision=%s" % (a, b))
    except Exception:
        pass
except Exception:
    pass
PYEOF
)
  untrusted=0 ran_on="" hint_shown=0
  while IFS='|' read -r st imp detail dist file; do
    [ -z "$st" ] && continue
    case "$st" in
      WHERE)  ran_on="$imp"
              case "$ran_on" in
                "$(hostname -s)"*) [ "$USE_SRUN" = 1 ] && info "(this time it ran on the login node)" ;;
                *) info "(ran on $ran_on)" ;;
              esac ;;
      NUMPYFAIL)
              untrusted=1
              warn "numpy will not even start on this machine; this round of package checks is void (it does not mean the packages are broken)"
              info "$imp"
              info "That is what an overloaded login node looks like. Re-run later, or let it go through srun (do not pass --local)." ;;
      OK)     ok "$imp $detail" ;;
      INFO)   info "$imp" ;;
      TVDIFF) warn "torch / torchvision are not from the same build ($imp)"
              info "torchvision was most likely installed from PyPI (a cu12 build). To align them:"
              info "  $PY -m pip install --force-reinstall --no-deps torch torchvision \\"
              info "       --extra-index-url https://download.pytorch.org/whl/cu130" ;;
      BROKEN)
              # A resource error != a broken package. When a login node is overloaded, numpy's C
              # extension cannot start and what surfaces is PyCapsule_Import and friends.
              case "$detail$untrusted" in
                *PyCapsule_Import*|*"Resource temporarily unavailable"*|\
                *"CPU dispatcher tracer"*|*1)
                  warn "$imp not measured reliably ($(printf '%s' "$detail" | cut -c1-60)...)"
                  if [ "$hint_shown" = 0 ]; then
                    hint_shown=1
                    info "Errors of this kind come from the machine running out of resources, not a broken package."
                    info "Re-run without --local so it is measured on a compute node. (Not repeated for later ones.)"
                  fi ;;
                *)
                  bad "$detail"
                  info "The package is there but the import failed. Uninstall cleanly and reinstall:"
                  info "  $PY -m pip uninstall -y $dist && ./setup.sh" ;;
              esac ;;
      MISSING)
              bad "$imp missing -> $detail"
              need_pip=1 ;;
    esac
  done <<< "$probe"
  }

  pre_missing=$MISSING
  probe_pkgs

  if [ "$DO_INSTALL" = 1 ] && [ "$need_pip" = 1 ]; then
    # Install everything in one go. The torch family follows the --extra-index-url in the file to
    # pytorch's cu130 index; the rest comes from PyPI. Compute nodes have outbound network
    # (measured: both pypi and the npm registry are reachable), so the install is sent there.
    echo "  pip install -r ${REQ#"$REPO"/}"
    pip_run install -r "$REQ" && ok "dependencies installed" || bad "pip install failed; see the errors above"
    # Probe again: the count from the round before installing no longer means anything
    printf '  %s-- re-checking after the install --%s\n' "$B" "$N"
    MISSING=$pre_missing
    probe_pkgs
  fi

  # Mixed-install health check: can the C extensions of torch and numpy load together?
  # requirements.txt pins the PyPI numpy, so under normal conditions this line always passes; a
  # failure usually means someone conda-installed a compiled package into the environment and
  # scrambled the libstdc++ dependencies.
  if "${NOTHREAD[@]}" "$PY" -c "import torch, numpy.fft, torchvision" >/dev/null 2>&1; then
    ok "torch + numpy.fft + torchvision work together"
  else
    err=$("${NOTHREAD[@]}" "$PY" -c "import torch, numpy.fft, torchvision" 2>&1 | tail -1)
    bad "the torch and numpy extensions are fighting"
    info "$(printf '%s' "$err" | cut -c1-110)"
    info "Put numpy back to the PyPI build that requirements.txt specifies:"
    info "  conda remove --force -p $ENV_PREFIX numpy && $PY -m pip install -r ${REQ#"$REPO"/}"
  fi

  # ffmpeg: vis.py writes mp4 through matplotlib's ffmpeg writer, which needs a real ffmpeg on PATH
  if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
  else
    bad "no ffmpeg on PATH -- vis.py will fail when saving mp4"
    [ "$DO_INSTALL" = 1 ] && conda install -y -p "$ENV_PREFIX" -c conda-forge ffmpeg \
      && { ok "ffmpeg installed"; MISSING=$((MISSING-1)); }
  fi
fi

# ============================================================
# 4. wandb
# ============================================================
step "4/5 wandb"
# train.py has wandb on by default (config.yaml: wandb.enabled=true, project=hpm-wave), but
# wandb_ready() in train.py first decides whether it is usable -- not installed / not logged in /
# init failed all just write one line to the log and training runs as normal. So not being logged
# in here is only a warning, not a missing part.
if [ -n "${WANDB_API_KEY:-}" ]; then
  ok "WANDB_API_KEY is set"
elif grep -qs 'api\.wandb\.ai' "$HOME/.netrc"; then
  ok "logged in (api.wandb.ai is in ~/.netrc)"
else
  warn "wandb is not logged in -- nothing is recorded this time, training runs as normal (wandb_ready in train.py skips it)"
  info "To record, pick one of three:"
  info "  wandb login                    # do it on a login node; a compute node may not reach the internet"
  info "  export WANDB_MODE=offline      # writes only to the local wandb/ directory; wandb sync afterwards"
  info "  sbatch run.sh wandb.enabled=false"
fi
[ "${WANDB_MODE:-}" = offline ] && info "currently WANDB_MODE=offline"

# ============================================================
# 5. directories and data
# ============================================================
step "5/5 directories and data"
# Only code/logs is created -- results/ and its subdirectories carry .gitkeep and are in the repo,
# so a clone already has them.
# logs/ has to be created: #SBATCH --output takes effect **before** the script runs, and when the
# directory is missing SLURM discards the log entirely while the job still reports COMPLETED
# (measured; see the banner of code/vis.sh). The mkdir -p logs inside the script is only a
# fallback and cannot save the first submission. --check looks without creating.
if [ -d "$REPO/code/logs" ]; then ok "code/logs/"
elif [ "$DO_INSTALL" = 1 ]; then mkdir -p "$REPO/code/logs" && ok "code/logs/ (created)"
else warn "code/logs/ missing (drop --check and it is created; without it the log of the first sbatch is lost)"; fi

# The training data itself (config.yaml: data.dir = data/3d/cropped_0.05) is not checked here --
# it is the probe path of data_*.tar in archives.tsv, and the loop below handles md5 and unpacking
# together. What is handled here is only prior_ktuned/ (config.yaml: data.prior_dir), which the
# loop does not cover: it is computed on demand by gen_prior.sh and is not packaged.
PRIOR="$REPO/data/3d/cropped_0.05/prior_ktuned"
if [ -d "$PRIOR" ] && [ -n "$(ls -A "$PRIOR" 2>/dev/null)" ]; then
  ok "prior_ktuned/"
else
  warn "prior_ktuned/ missing -- once the data is in place, cd code && sbatch gen_prior.sh to generate it"
  info "To rebuild the whole of data/ yourself (without taking the packages from the cloud):"
  info "  data/3d/crop_fields.sh   crops a copy out of the volume fields of the OpenFOAM case (\$OCEAN_CASE)"
  info "  code/gen_prior.sh        then generates the prior_ktuned/ that the fwv line needs"
fi

# ---- large files (historical visualisations / checkpoints / training data) ----
# This whole section is delegated to archive/restore.sh -- manifest parsing, md5, unpacking and
# per-file manifest verification exist in that one implementation and are not copied here (the
# copy that existed omitted --skip-old-files and silently overwrote repo files that overlap with
# the package). Its rule is **act only when everything is present**: one missing package and it
# lists every problem and exits 1 without unpacking a byte.
# Idempotent: it skips by itself any package whose probe path is non-empty, so re-running setup.sh
# does not unpack again.
RESTORE="$REPO/archive/restore.sh"
if [ ! -x "$RESTORE" ]; then
  bad "archive/restore.sh is missing (or not executable) -- it should be in the repo; the large files are restored with it"
else
  # --check is passed through: scan only, do not unpack. It reads the rest from OCEAN_ARCHIVE /
  # archives.tsv itself.
  [ "$DO_INSTALL" = 1 ] && r_args=() || r_args=(--check)
  "$RESTORE" "${r_args[@]+"${r_args[@]}"}" 2>&1 | sed 's/^/  /'
  r_rc=${PIPESTATUS[0]}
  case "$r_rc" in
    0) ok "all large files in place" ;;
    1) bad "the large files are incomplete (see restore.sh's scan above)"
       info "Put the packages in ${OCEAN_ARCHIVE:-$REPO/archive}/ and run again; they unpack automatically."
       info "It **acts only when everything is present** -- with only some of the packages in hand, name those:"
       info "  ./archive/restore.sh data_    # matched by substring of the package name; the date and .tar can be omitted"
       info "  ./archive/restore.sh web      # unpacks only what web-demo needs (= the data package)"
       info "  ./archive/restore.sh          # everything (one missing and nothing is unpacked)" ;;
    *) bad "restore.sh exited $r_rc (a usage error?)" ;;
  esac
fi

# FUNWAVE-TVD -- the third-party Boussinesq solver (the source of the fwv line's prior). Not
# committed; it is a clean clone of someone else's repository, pulled on demand. The tag is pinned
# so an upstream change cannot make it drift.
FW_DIR="$REPO/FUNWAVE-TVD"
FW_URL=https://github.com/fengyanshi/FUNWAVE-TVD.git
FW_TAG=Version_3.6
if [ -d "$FW_DIR/.git" ]; then
  ok "FUNWAVE-TVD ($(cd "$FW_DIR" && git describe --tags 2>/dev/null || echo '?'))"
  if [ -n "$(cd "$FW_DIR" && git status --porcelain 2>/dev/null)" ]; then
    info "There are local changes -- to pass them on to others, save them as a patch and commit it:"
    info "  cd FUNWAVE-TVD && git diff > ../funwave.patch"
  fi
elif [ "$DO_INSTALL" = 1 ]; then
  echo "  git clone $FW_URL ($FW_TAG, about 250 M)"
  if git clone --depth 1 --branch "$FW_TAG" "$FW_URL" "$FW_DIR"; then
    ok "FUNWAVE-TVD cloned"
    if [ -f "$REPO/funwave.patch" ]; then
      (cd "$FW_DIR" && git apply "$REPO/funwave.patch") \
        && ok "funwave.patch applied" || warn "funwave.patch does not apply; have a look yourself"
    fi
  else
    bad "clone failed (if the login node cannot reach github, try a compute node)"
  fi
else
  info "no FUNWAVE-TVD (drop --check and $FW_TAG is cloned automatically, about 250 M)"
fi

# ============================================================
step "summary"
printf '  environment: %s\n' "${ENV_PREFIX:-none}"
[ "${untrusted:-0}" = 1 ] && printf '  %s⚠ this round of package checks was not reliable (numpy would not even start); treat the results above as indicative only%s\n' "$Y" "$N"
printf '  missing: %s%d%s   warnings: %s%d%s\n' \
  "$([ "$MISSING" -gt 0 ] && echo "$R" || echo "$G")" "$MISSING" "$N" \
  "$([ "$WARNED" -gt 0 ] && echo "$Y" || echo "$G")" "$WARNED" "$N"
cat <<TIP

  Every time you open a new shell:
    source $REPO/activate.sh          # find conda, activate the environment, and export \$REPO

  Then:
    cd code && sbatch run.sh          # training (dgxh by default; may wait 2-3 days)
    cd code && sbatch vis.sh          # rendering
  The sbatch scripts source activate.sh themselves, so there is no need to activate first.
  For what the six lines are about and where the data lives, see README.md.
TIP

[ "$MISSING" -gt 0 ] && exit 1
exit 0
