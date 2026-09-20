# Environment and cluster operations

> This file covers **how to get the environment up and how to run on the cluster** only.
> For the project itself (the six model lines, the data assets, the results) see [README.md](README.md);
> for the interactive web demo see [code/web-demo/README.md](code/web-demo/README.md).

**Runs on the cluster only** (the data is under `/nfs/hpc/share`, the scripts are SLURM scripts,
torch needs cu130); running `setup.sh` on any other machine errors out immediately.

---

## Quick start

```bash
cd /nfs/hpc/share/$USER
git clone https://github.com/Picomp-lab/Ocean_OpenFOAM.git models
cd models && ./setup.sh
```

Once that finishes:

```bash
source activate.sh                # for interactive use; $REPO is available afterwards
cd code && sbatch run.sh          # training (the sbatch scripts source it themselves, so no need to activate first)
```

**The one thing `setup.sh` cannot solve is the data** -- 254 GB that is not in the repository.
All it does is tell you what is missing, whether you can recompute it yourself, and who to ask
(step 5 covers that separately).

---

## setup.sh

```bash
./setup.sh            # one command, end to end: installs whatever is missing
./setup.sh --check    # probe only, install nothing (a non-zero exit = something is missing)
```

**That is the only switch.** Everything else is automatic: a missing conda environment or
FUNWAVE-TVD gets installed, and a package sitting in `archive/` is unpacked automatically.
**Not one package in `archives.tsv` may be absent** -- an absent one counts as a missing part and
the script exits non-zero (every step's checks still run to completion first). Three environment
variables allow fine-tuning and normally need no attention: `OCEAN_ENV` (environment location),
`OCEAN_ARCHIVE` (the large-package directory, default `<repo root>/archive`), and `SRUN_WAIT`
(seconds to wait for a compute node; 0 keeps everything on the login node).

Five steps: platform/SLURM -> conda environment (default `/nfs/hpc/share/$USER/.conda/envs/ocean`,
changeable with `OCEAN_ENV=`; it also generates `.env.local`) -> python dependencies -> wandb ->
directories and data (including automatic unpacking from `archive/` and an automatic clone of
FUNWAVE-TVD). web-demo is not among them; see below.

Probing and pip installation go by default through `srun -p share` onto a compute node (measured:
scheduled in twenty-odd seconds, and compute nodes have outbound network) -- on a login node
`RLIMIT_NPROC=400` per user is shared across the whole node, so when it is full numpy will not
start and a perfectly good package is judged broken. **preempt is not used**: it gets preempted and
requeued, which only gets in the way of a check that takes tens of seconds. When `share` cannot be
scheduled it waits 180 seconds (changeable with `SRUN_WAIT=`) and then falls back to the login
node, rather than leaving you hanging.

### web-demo: clone it and it runs

The frontend `web/dist/` (108 K), the **backend binary** `server/target/release/wave-demo` (4.5 M
on disk, about 1.6 M in git), and the demo's default weights (`results/web/model/`, `best.pt`
8.6 M) are **all in the repository**, so there is nothing to build and nothing to configure:

```bash
./code/web-demo/start.sh
```

`setup.sh` does not touch it -- it neither verifies nor builds it. To change and rebuild it, the
full instructions (including that the frontend must come before the backend, that it must go to a
compute node, and the two `srun` commands) are in
[code/web-demo/README.md](code/web-demo/README.md).
Only the one most likely to break quietly is repeated here: **after changing `server/src/` or
`web/src/` you must rebuild and commit the new artifacts yourself** -- nobody checks for you, and
the source and the binary will drift apart without a sound.

Why the binary is allowed into the repository: with unchanged sources, `cargo build --release` is
**byte-for-byte reproducible** (measured twice, identical md5), so it produces no phantom diffs.
The frontend is the same -- on 2026-08-23 `vite build` was re-run on cn-e02 and the output was
byte-identical to what is in the repository. The price is that the binary is platform-locked: it
needs `GLIBC_2.28` and Linux x86-64, and is only meaningful on this cluster.

---

## Dependency list

All of it is in the single file **`requirements.txt`**, installed in one pass by `setup.sh`, which
**hard-codes no package name** (to add a package, edit that file, not the script). The torch family
relies on one `--extra-index-url` line in the file pointing at pytorch's cu130 index; the rest comes
from PyPI.

The torch on PyPI that also calls itself `2.11.0` is a cu12 wheel; install that and
`torch.cuda.is_available()` is `False` on a compute node, so torch has to come from pytorch's index.

**Why it says `torch==2.11.0` and not `torch==2.11.0+cu130`** (measured 2026-08-20 with
`pip install --dry-run`):

- With `torch==2.11.0`: both indexes offer a candidate, but PEP 440 ranks a local version above the
  same base version, so `2.11.0+cu130` > `2.11.0` and pip must take the cu130 one. The measured
  result is exactly what the older `--index-url` form gave -- guaranteed by the spec, not by luck.
- With `torch==2.11.0+cu130`: **it actively breaks things**. Once the wheel is installed, the version
  in its dist metadata is stripped back to `2.11.0` (`+cu130` survives only in `torch.__version__`),
  so pip decides every time that it is "not installed" and re-downloads 2 GB on every `setup.sh` run.

`setup.sh` carries a separate check comparing the build tags of torch and torchvision, and warns if
they have been mixed.

> Incidentally, the cu130 index mirrors 117 packages, `numpy` and `pillow` among them, which are
> also on our list. This is not a problem: `numpy==2.2.6` is the same wheel on both indexes
> (measured, identical sha256).

`polars` uses the `lts-cpu` variant -- the `share` partition mixes in older ivybridge machines,
where the avx512 instructions in the ordinary wheel cause an illegal instruction.

Versions: Python 3.10.20 / PyTorch 2.11.0+cu130 / CUDA 13.0.

### WARNING: do not `conda install` compiled packages into this environment; always use pip

The cluster is el8 and the system libstdc++ only reaches `GLIBCXX_3.4.25`, older than the `3.4.29`
that conda-forge build artifacts require. Mixing them in stops the C extensions of torch and numpy
from loading each other, and what surfaces is torch failing to find `NP_SUPPORTED_MODULES`, which
is thoroughly misleading. Following `requirements.txt` through PyPI avoids it entirely, and
`setup.sh` has a runtime health check (`import torch, numpy.fft, torchvision`) watching for a
regression.

> History: numpy used to be a conda-forge build; after the switch to PyPI on 2026-08-19 the root
> cause disappeared, and the `LD_LIBRARY_PATH` workaround was removed on 08-20. For what **the same
> version number with a different compiled artifact** means for reproducibility, see
> [README.md](README.md) section 6.7-2.

---

## activate.sh -- how the scripts find the environment

**No sbatch script ever hard-codes an environment path**; they all source `activate.sh` at the repo
root:

```bash
_d="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$_d/activate.sh" ] && [ "$_d" != / ]; do _d=$(dirname "$_d"); done
source "$_d/activate.sh"      # find conda, activate the environment; $REPO is available afterwards
```

`sbatch` copies the script into spool, so locating it relies on `$SLURM_SUBMIT_DIR` rather than `$0`.

The environment is located in this order, **with nobody's username anywhere in it**:

1. `$OCEAN_ENV` (set explicitly)
2. `OCEAN_ENV=` in `<repo>/.env.local` (generated by `setup.sh`, not committed)
3. `/nfs/hpc/share/$USER/.conda/envs/ocean` (the default)

### Self-check before committing: no hard-coded personal paths

Every script sources `activate.sh` at the repo root, so nobody should still be writing
`/nfs/hpc/share/<someone>/` or `/nfs/stak/users/<someone>/` -- clone it under another account and it
would not run. **After editing a script and before committing**, run this at the repo root:

```bash
git ls-files -z '*.sh' '*.py' '*.md' '*.yaml' '*.rs' '*.toml' \
  | xargs -0 grep -nIE -- '/nfs/(hpc/share|stak/users)/[a-z][a-z0-9_-]*/' \
  | grep -vE '/nfs/stak/a1/rhel5apps|/nfs/hpc/share/coast-lab|\$USER|^legacy/'
```

No output = clean. Three exemptions do not count as hard-coded personal paths: `rhel5apps` is the
university-wide shared conda installation, `coast-lab` is the lab's shared FUNWAVE data
(`gen_prior.sh` / `scan.sh` point at it), and `$USER` is a variable. `legacy/` is no longer
maintained and is not included.

---

## What lives outside the repository

| | how to get it |
|---|---|
| `data/` (49 G) | in cloud storage, `data_20260822.tar`. Can also be rebuilt with `data/3d/crop_fields.sh` + `code/gen_prior.sh` |
| the output of `legacy/` | in cloud storage, three packages (below). The repo keeps only the source |
| `results/train`, `results/vis` | in cloud storage, `results_20260822.tar`. The repo keeps only empty directories with a `.gitkeep` |
| `FUNWAVE-TVD/` (256 M) | a clean clone of the third-party solver; `./setup.sh` pulls it when absent |
| `$OCEAN_DATA` | `ocean_project/` for the POD/LSTM line; has a default |
| `$OCEAN_CASE` | the OpenFOAM case; has a default |

The repository itself is small (249 files); a clone gives you the code plus a skeleton of empty
directories each holding a `.gitkeep` that says which package its contents are in and how to
restore them.

Not being logged in to wandb **does not affect training**: `train.py` checks for itself before
starting and, if the check fails, writes the reason to the log and records nothing this time (it
never stops at an interactive prompt and burns a GPU job until it times out).
Projects: `hpm-wave` (the two HPM lines) and `tsolverpp` (3D Transolver++), user `cassan-osu`.
The local `wandb/` directory is a regenerable cache -- every run is already synced to the cloud, so
deleting it loses nothing.

### Large files in the cloud (`archive/`)

The historical visualisations and checkpoints come to nearly 1 G, and git does not delta-compress
binaries -- every new version is stored whole again -- so they live in cloud storage rather than in
the repository. The list is **`archives.tsv`** at the repo root (6 TAB-separated columns; the format
is described in that file's header):

| package | unpacks to | size | contents |
|---|---|---|---|
| `data_20260822.tar` | `data/` | 48.6 G | 3d 34.5 G + fwv 8.5 G + 2d 5.6 G, 12122 files |
| `legacy_20260822.tar` | `legacy/` (**excluding hpm**) | 12.7 G | fno 12.6 G (`processed_data` + `outputs`), `transolver++/results`, `tsolverpp/outputs` |
| `legacy_hpm_vis_20260820.tar` | `legacy/hpm/vis/` | 486 M | 334 mp4 files, 46 sets of historical visualisations |
| `legacy_hpm_outputs_20260820.tar` | `legacy/hpm/outputs/` | 428 M | `checkpoints/*.pt` + `.hydra/*.yaml` for 46 runs |
| `results_20260822.tar` | `results/` | 194 M | `train/` 87 M + `vis/` 56 M + the record of those two web-demo runs |

The restore logic exists in `restore.sh` alone, and step 5 of `setup.sh` simply calls it (`--check`
is passed through):

```bash
./setup.sh                      # installs the environment too; a missing package counts as a missing part, exit code non-zero (every step is still checked)
./archive/restore.sh            # packages only: full scan -> unpack only when complete -> verify each file against the manifest afterwards
                                # one missing and it lists every problem, exit 1, not a byte unpacked
./archive/restore.sh --check    # inspect only, unpack nothing
./archive/restore.sh web        # unpack only what web-demo needs (= the data package)
./archive/restore.sh data_      # name the ones you have (matched by substring; the date and .tar can be omitted)
```

**When you have only some of the packages**, use the last form -- with no argument it "acts only
when everything is present", and one missing package means nothing is unpacked.

`restore.sh` uses `tar --skip-old-files` and never overwrites an existing file -- several packages
overlap with the repository (`data/fwv/TK94/input.txt`, for instance), and without that flag a
single restore would silently overwrite the repository's version.

Behaviour (identical through both entry points, because it is the same code): **a probe path that
exists and is non-empty is skipped** (`.gitkeep` does not count; idempotent, so it can be re-run
freely); otherwise it looks for the package in `archive/` and unpacks it **only after the md5
verifies**; when it is not in `archive/` either it **reports a missing part** and prints a download
command you can copy straight out (from column 6 of `archives.tsv`, supporting `gdrive:<FILE_ID>`
and direct `https://` links; when column 6 is empty it says to ask the repo owner).

**It never downloads over the network on its own** -- nearly 1 G, so when to fetch it is a human's
decision. But **an incomplete set of packages makes `setup.sh` fail** (non-zero exit): `data_*.tar`
is the training data and nothing runs without it, and while the others only affect digging through
history, they count as missing parts too, so that "finished installing" and "installed completely"
are not confused.

Beside each package there is also a `.manifest` (per-file md5), so once you have one you can verify
file by file rather than just the package as a whole:

```bash
cd <repo root> && md5sum -c $OCEAN_ARCHIVE/legacy_hpm_vis_20260820.manifest
```

---

## SLURM quick reference

| partition | hardware | use in this project |
|---|---|---|
| `dgxh` | H100 80GB / H200 143GB | all GPU training |
| `ampere` | **A40 48G** (no A100) | inference, rendering, animation; rejects pure-CPU jobs (QOS `MinTRES=gres/gpu=1`) |
| `eecs` / `share` | CPU (`eecs` also has RTX2080 11G) | data preparation, POD, LSTM, prior generation and calibration |

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,Elapsed,State,ExitCode,Reason,MaxRSS
tail -f code/logs/hpm_<jobid>.log
```

WARNING: `code/logs/` must exist **before submitting** -- `#SBATCH --output` takes effect before the
script runs, and when the directory is missing SLURM discards the log entirely while the job still
reports `COMPLETED` (measured). `setup.sh` creates it.

To find out which partition is fastest to submit to right now, do not rely on
`sbatch --test-only` alone (it gives a worst-case priority queue simulation, ignores backfill, and
does not check whether pending jobs can actually start). Look directly:

```bash
sinfo -p <part> -N -o '%N|%t|%C|%G'        # are there idle nodes
squeue -p <part> -t PD -o '%i|%u|%r|%b'    # what reason the PENDING ones are stuck on
```

Idle nodes present and every PENDING job stuck on `Dependency` (especially
`DependencyNeverSatisfied`, which will never run) -- submit, and it starts immediately.
