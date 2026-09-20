#!/bin/bash
# Restorer for the large-file packages -- **two-phase, acts only when everything is present**.
#
#   Phase 1 scan:    read-only. Checks item by item that what should be there is there and that
#                    each package's md5 matches. Any one missing or corrupt -> list every problem
#                    at once, exit 1, unpack nothing.
#   Phase 2 release: entered only when phase 1 is all green. Lays the packages back out according
#                    to the "unpack to" column of archives.tsv.
#   Phase 3 verify:  **always runs**, not optional. Takes <package>.manifest and md5-checks every
#                    freshly unpacked file -- a correct md5 on the package as a whole does not mean
#                    what came out of it is correct (a full disk, or an interrupted unpack, does
#                    not necessarily make tar complain).
#
# Why it acts only when everything is present: half a set of output is harder to diagnose than
# none -- when it fails halfway with "xxx is missing", several GB are already laid down and there
# is no telling which of it came from this run. Better to complain first and start afterwards.
#
# The package list is archives.tsv at the repo root (6 TAB-separated columns; the format is
# described in that file's header).
#
# Directory roles (the single directory <repo>/archive/ is enough; packages downloaded from the
# cloud go straight in there):
#   restore.sh          this script         committed
#   *.manifest          per-package, per-file checksum list, committed, so a clone already has it
#   *.tar               the packages themselves, not committed; put them here after downloading
#                       them from cloud storage. To keep them elsewhere, export OCEAN_ARCHIVE=/path
#
# **Fills gaps only, never overwrites** (tar --skip-old-files). The packages overlap with the repo
# -- the data package carries data/3d/crop_fields.{py,sh} and
# data/fwv/{make_cases.py,wk_check.py,TK94/input.txt,TK94/gauges.txt} (the provenance of the case,
# though the repo holds a copy too), and the legacy package carries the source of the historical
# lines. Without that flag, anyone who had edited input.txt would have it silently replaced by the
# version from packaging day on a single restore, and silently is the problem.
# "Inputs outside the packages" are a different matter: they are **in no package** (they are all in
# the repo), but web-demo will not run without them, so they are checked here too. WARNING: before
# adding anything to that list, make sure it really is in no package -- writing a packaged file in
# there deadlocks: phase 1 exits 1 because it is "missing", when the very thing that would produce
# it is the unpack.
#
# Usage:
#   ./restore.sh                    restore every package in archives.tsv
#   ./restore.sh web                unpack only what web-demo needs (= the data package, 48.6 G;
#                                   the demo really reads about 3.1 G, but a package unpacks whole)
#   ./restore.sh <name> [<name>...] restore the named packages
#   ./restore.sh --check            scan only, unpack nothing
#   ./restore.sh --skip-input-check do not check the "inputs outside the packages"
#   ./restore.sh --verify-all       also verify, file by file, packages that were already in place
#                                   (it has to read all of the content, tens of GB upwards, hence
#                                   not the default)
#
# Exit codes: 0 = all green (or already in place), 1 = something missing/corrupt, 2 = usage error

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ARC_DIR="${OCEAN_ARCHIVE:-$REPO/archive}"
ARC_LIST="$REPO/archives.tsv"

DO_EXTRACT=1
DO_INPUT=1
DO_VERIFY_ALL=0
declare -a WANT=()

for a in "$@"; do
  case "$a" in
    --check)             DO_EXTRACT=0 ;;
    --skip-input-check)  DO_INPUT=0 ;;
    --verify-all)        DO_VERIFY_ALL=1 ;;
    -h|--help)           sed -n '2,51p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)                  echo "unknown option: $a (-h for usage)" >&2; exit 2 ;;
    web)                 WANT+=("data_") ;;      # everything web-demo needs is in the data package
    *)                   WANT+=("$a") ;;
  esac
done

c_ok=$'\033[32m'; c_bad=$'\033[31m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
[ -t 1 ] || { c_ok=; c_bad=; c_dim=; c_off=; }
ok()   { echo "  ${c_ok}✓${c_off} $*"; }
bad()  { echo "  ${c_bad}✗${c_off} $*"; }
skip() { echo "  ${c_dim}·${c_off} $*"; }

# Selection: no argument = everything; with arguments, matched by substring (so data_ alone works,
# without the date or the .tar)
selected() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  local n="$1" w
  for w in "${WANT[@]}"; do [[ "$n" == *"$w"* ]] && return 0; done
  return 1
}

# Already in place = the probe path is a non-empty directory (the same test setup.sh uses).
# WARNING: .gitkeep does not count -- that is the directory skeleton from the repo (so a clone shows
# where things belong); without excluding it the skeleton alone would make every package look
# "already in place", and nobody would be warned that the data was never unpacked.
in_place() {
  local p="$REPO/$1"
  [ -d "$p" ] || return 1
  [ -n "$(ls -A "$p" 2>/dev/null | grep -v '^\.gitkeep$')" ]
}

PROBLEMS=0
declare -a TODO_NAME=() TODO_DEST=() TODO_PROBE=() INPLACE_NAME=()

# ----------------------- phase 1: scan -----------------------

echo "repo     : $REPO"
echo "packages : $ARC_DIR"
echo
echo "[1/3] scan"

if [ ! -f "$ARC_LIST" ]; then
  bad "archives.tsv is missing -- it should be in the repo; the package list depends on it"
  exit 1
fi

echo "--- large-file packages ---"
n_row=0
while IFS=$'\t' read -r a_name a_dest a_probe a_size a_md5 a_url || [ -n "${a_name:-}" ]; do
  case "${a_name:-}" in ''|'#'*) continue ;; esac
  selected "$a_name" || continue
  n_row=$((n_row + 1))

  if in_place "$a_probe"; then
    skip "$a_probe already in place, skipping $a_name"
    INPLACE_NAME+=("$a_name")
    continue
  fi

  local_tar="$ARC_DIR/$a_name"
  if [ ! -f "$local_tar" ]; then
    bad "package $a_name missing ($a_size) -- and $a_probe is not there either"
    case "${a_url:-}" in
      '')        echo "      column 6 of archives.tsv has no address; ask the repo owner" ;;
      gdrive:*)  echo "      python -m gdown ${a_url#gdrive:} -O $ARC_DIR/$a_name" ;;
      *)         echo "      curl -fL -o '$ARC_DIR/$a_name' '$a_url'" ;;
    esac
    PROBLEMS=$((PROBLEMS + 1))
    continue
  fi

  if [ -n "${a_md5:-}" ]; then
    got="$(md5sum "$local_tar" | awk '{print $1}')"
    if [ "$got" != "$a_md5" ]; then
      bad "$a_name md5 mismatch (expected $a_md5, got $got)"
      echo "      an incomplete download, or the cloud copy was replaced. Delete and re-download, or check archives.tsv"
      PROBLEMS=$((PROBLEMS + 1))
      continue
    fi
  fi

  ok "$a_name ($a_size) to be unpacked into ${a_dest:-.}/"
  TODO_NAME+=("$a_name"); TODO_DEST+=("${a_dest:-.}"); TODO_PROBE+=("$a_probe")
done < "$ARC_LIST"

# Not one row matched = the package name is misspelled or the list changed. Do not let it run all
# the way green reporting "already in place".
if [ "$n_row" = 0 ]; then
  bad "no matching row in archives.tsv (selection: ${WANT[*]:-everything})"
  echo "      the packages that exist:"
  grep -v '^#' "$ARC_LIST" | cut -f1 | grep -v '^$' | sed 's/^/        /'
  exit 2
fi

# Inputs outside the packages -- all in the repo, in no package; missing means the clone is
# incomplete.
# WARNING: the mesh metadata under data/3d/cropped_0.05/ (coords / lbo / slice_y0.30 /
# chunk_010_times / stats_*) used to be listed here, but the repackaging of 20260822 folded them
# into data_*.tar -- leaving them here would deadlock (phase 1 exits 1 saying "missing", when only
# unpacking would produce them). Their integrity is the job of the per-file data_*.manifest check in
# phase 3.
CK=results/web/model/hpm_fw_aU_h128/2026-08-12_15-31-45   # the demo's default weights, in the repo
INPUTS=(
  "$CK/checkpoints/best.pt|model weights"
  "$CK/.hydra/config.yaml|archived hyperparameters (vis_adp.sh reads it)"
  "code/web-demo/server/target/release/wave-demo|backend binary"
  "code/web-demo/web/dist/index.html|frontend bundle"
)

if [ "$DO_INPUT" = 1 ]; then
  echo "--- inputs outside the packages ---"
  for item in "${INPUTS[@]}"; do
    p="${item%%|*}"; why="${item#*|}"
    if [ -s "$REPO/$p" ]; then
      ok "$p"
    else
      bad "$p missing -- $why"
      PROBLEMS=$((PROBLEMS + 1))
    fi
  done
fi

echo
if [ "$PROBLEMS" -gt 0 ]; then
  echo "${c_bad}scan failed: $PROBLEMS problems, nothing was unpacked.${c_off}"
  exit 1
fi
echo "${c_ok}scan passed.${c_off}"

# ----------------------- phase 2: release -----------------------

echo
if [ "$DO_EXTRACT" = 0 ]; then
  echo "[2/3] release -- --check mode, skipped (verification is skipped with it)"
  exit 0
fi

echo "[2/3] release"
if [ ${#TODO_NAME[@]} -eq 0 ]; then
  skip "no package to unpack (all already in place)"
else
  for i in "${!TODO_NAME[@]}"; do
    name="${TODO_NAME[$i]}"; dest="${TODO_DEST[$i]}"; probe="${TODO_PROBE[$i]}"
    echo "  unpacking $name -> $dest/"
    mkdir -p "$REPO/$dest"
    if ! tar -xf "$ARC_DIR/$name" -C "$REPO/$dest" --skip-old-files; then
      bad "$name failed to unpack"
      exit 1
    fi
    if in_place "$probe"; then
      ok "$probe"
    else
      bad "unpacked, but $probe is still empty -- do the paths inside the package not match the 'unpack to' column of archives.tsv?"
      exit 1
    fi
  done
fi

# Per-file verification -- the paths in a manifest are relative to the repo root, so it has to run
# at the repo root.
# Run unconditionally: reaching this point means the disk really was touched, and skipping the
# verification would leave the job unfinished.
echo
echo "[3/3] per-file verification"
declare -a CHECK_NAME=("${TODO_NAME[@]:-}")
[ "$DO_VERIFY_ALL" = 1 ] && CHECK_NAME+=("${INPLACE_NAME[@]:-}")
n_check=0
for name in "${CHECK_NAME[@]:-}"; do
  [ -n "$name" ] || continue
  n_check=$((n_check + 1))
  man="$ARC_DIR/${name%.tar}.manifest"
  if [ ! -f "$man" ]; then
    bad "$(basename "$man") is missing -- cannot verify $name"
    exit 1
  fi
  if (cd "$REPO" && md5sum -c "$man" > /dev/null 2>&1); then
    ok "$(basename "$man") all files match"
  else
    bad "$(basename "$man") has files that do not match:"
    (cd "$REPO" && md5sum -c "$man" 2>&1 | grep -v ': OK$' | head -10)
    exit 1
  fi
done
[ "$n_check" = 0 ] && skip "no package was unpacked this time, so there is nothing to verify (to verify the ones already in place too, add --verify-all)"

# The demo's runtime output directories; they have to exist even when empty
mkdir -p "$REPO/results/web/priors" "$REPO/results/web/vis"
echo
if [ "$DO_INPUT" = 1 ] && [ ${#WANT[@]} -eq 0 ]; then
  echo "${c_ok}Everything in place.${c_off} web-demo: cd code/web-demo && ./start.sh"
else
  echo "${c_ok}The selected packages are in place.${c_off} (No full check was done; for that run ./restore.sh --check)"
fi
