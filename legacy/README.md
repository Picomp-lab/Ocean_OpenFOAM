# legacy — retired lines

Everything here is **kept as a record; do not modify it**. The code that actually runs
lives in `code/` at the repository root. What each line attempted, how it turned out and
why it was dropped is in §2–§5 of the root `README.md`.

| directory | what it is | date | outcome |
|---|---|---|---|
| `pod-lstm/` | POD + POD-LSTM, plus the 2D Transolver++ code | 05-03 | usable in coefficient space; autoregression in field space failed |
| `transolver++/` | Transolver++ on a 2D point cloud | 05-11 | single step usable, diverges after 100 |
| `fno/` | FNO on a 2D regular grid | 05-12 | best single step, diverges after 96 |
| `tsolverpp/` | Transolver++ on a 3D point cloud | 05-17 | never converged (ep25/100) |
| `hpm/` | the previous generation of HPM (including `fwv/`) | 06–08 | superseded by the rewrite in `code/` |
| `env_info/` | one-off point-count / band-distribution probes | | not in the repo; ships in `legacy_20260822.tar` |

## What is in the repo, and what is not

**Only source code and small evidence files stay in the repo.** Everything bulky was
moved out on 2026-08-22/23 and now lives in packs listed in the root `archives.tsv`:

| pack | contents | size |
|---|---|---|
| `legacy_20260822.tar` | everything under `legacy/` **except `hpm/`** — `fno/processed_data` 6.4 G (7 files), `fno/outputs` 6.2 G (87), `transolver++/results` 354 M, `tsolverpp/outputs` 129 M, `pod-lstm/`, `env_info/` | 12.7 G / 219 files |
| `legacy_hpm_vis_20260820.tar` | `hpm/vis/` — the old HPM line's rendered videos | 486 M / 334 files |
| `legacy_hpm_outputs_20260820.tar` | `hpm/outputs/` — the old HPM line's hydra run directories | 428 M / 254 files |

So a fresh clone gives you roughly this, and nothing more:

```
legacy/fno/           76 K    source only
legacy/pod-lstm/     156 K    source only
legacy/tsolverpp/    2.9 M    source + three small mp4
legacy/transolver++/ 3.9 M    source + results/figs
legacy/hpm/           60 M    source + fwv/ (see below)
```

`legacy/hpm/fwv/` is the exception: since 2026-08-23 its rendered outputs **are** tracked
— `fwv/vis/` (32 mp4, 44.5 M) and `fwv/hpm_fw_ss_R4/` (8 mp4, 10.6 M), joining the
`vis_align/` clips and the `toffset_scan/*.json` that were already tracked. They were the
only files in the whole repository that lived in neither git nor a pack, and at 55 M they
are small enough to version rather than orphan.

To restore any of it:

```bash
# download the pack from the cloud drive into archive/ first (see archives.tsv column 6)
./archive/restore.sh legacy_20260822        # or: ./archive/restore.sh   for everything
```

`restore.sh` scans first and refuses to unpack anything unless every required piece is
present, then verifies each restored file against the pack's `.manifest`. It never
overwrites an existing file, so the source code already in your working tree is safe.

⚠️ Older revisions of this file said `./setup.sh --archives`. That flag does not exist —
`setup.sh` takes only `--check`, and anything else exits 2.

## Outputs that were never here

- `results/train/` and `results/vis/` used to sit under the repository root. They were
  moved out too and ship in `results_20260822.tar` (194 M / 69 files); the repo keeps only
  empty directories with a `.gitkeep` explaining where the contents went.
  (`results/fwv/`, the ADP scan line's outputs, was deleted on 2026-08-20 — see root
  README §1.5.)
- `legacy/*/wandb/` — the local wandb run directories (`hpm/wandb` 335 M,
  `tsolverpp/wandb` 17 M) were moved to `../archive/legacy/` and are **not in any pack
  yet**. The same runs exist in the cloud under the `hpm-wave` project on wandb.ai, since
  all of them ran in online mode.
- The POD / LSTM line's data lives outside the repository; point `$OCEAN_DATA` at it
  (default `~/hpc-share/ocean_project`).
- The OpenFOAM cases use `$OCEAN_CASE` (23 G of CFD data, never in the repository).

## If you really want to rerun one of these

1. Environment as usual: `./setup.sh` at the repository root; every script does
   `source <repo>/activate.sh`.
2. ~~numpy must be imported before `import torch`~~ — **retracted (2026-08-20)**. Calling
   it "an import-order quirk in torch 2.11" was wrong: the real cause was that the numpy
   in the environment was a conda-forge build requiring a newer libstdc++ than el8 ships,
   and torch, loaded first, had already taken the SONAME slot. Swapping numpy for the PyPI
   build on 2026-08-19 removed the root cause, and import order no longer matters. The one
   rule that still holds is **never `conda install` a compiled package into the
   environment** (see `SETUP.md`).
3. Paths were all changed to `$REPO` / `$OCEAN_DATA` / `$OCEAN_CASE`, but **this was never
   verified by an actual run** — only `bash -n` and `sbatch --test-only`.
