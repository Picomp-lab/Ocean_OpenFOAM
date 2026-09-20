# Wave-field surrogate model: an exploration record

> Covers the four historical lines under `legacy/` (`pod-lstm/`, `transolver++/`, `fno/`, `tsolverpp/` and `hpm/`), `code/` (the two HPM lines, **the current main effort**), plus `data/` (data assets) and `results/` (output).
> Purpose: to record **what was done, how it turned out, how to reproduce it, and where things are**.
>
> **How the three documents divide up**: this one = the project itself (data / six lines / results);
> [SETUP.md](SETUP.md) = environment and cluster operations;
> [code/web-demo/README.md](code/web-demo/README.md) = the interactive web demo.
>
> `legacy/hpm/` is the previous implementation of `code/` (the same HPM line, before the refactor); this document describes `code/`.

---

## Environment

After cloning, run `./setup.sh` first -- it creates the conda environment, installs the
dependencies, and checks whether the data is complete.
For the full account (setup.sh's switches, why the dependencies are split in two, how activate.sh
locates the environment, a SLURM quick reference) see **[SETUP.md](SETUP.md)**.

The interactive web demo has its own document: **[code/web-demo/README.md](code/web-demo/README.md)**.

---

## Directory layout

```
code/        the current main lines (the two HPM ones); the only code being changed
legacy/      historical lines, no longer maintained: hpm/ fno/ transolver++/ tsolverpp/ pod-lstm/ env_info/
results/     output. Only the demo weights in web/model/ travel with the repo; the rest is in results_*.tar
data/        data assets, 48.6 G (3d 34.5 G + fwv 8.5 G + 2d 5.6 G) -- **not distributed with the repo**,
             it is in data_*.tar; the repo keeps only the record of "how it was produced", such as
             crop_fields.* / make_cases.py / input.txt
archive/     restore.sh + each package's .manifest (committed); the .tar files downloaded from cloud
             storage go here too (not committed). For the list and the restore procedure see archives.tsv
```

---

## 0. Quick navigation

| # | direction | directory | date | dimension | status | short-range accuracy | long autoregression |
|---|---|---|---|---|---|---|---|
| 1 | POD + LSTM | `legacy/pod-lstm/` | 05-03 | 2D | finished, judged a failure | 10-25% in coefficient space | 51-164% in field space, failed |
| 2 | Transolver++ | `legacy/transolver++/` | 05-11 | 2D point cloud, 149,758 | finished (early stop ep180) | alpha 3.5% / Ux 16.3% / Uz 26.4% | 111% at 100 steps, diverges |
| 3 | FNO | `legacy/fno/` | 05-12 | 2D grid 4096x128 | hit the time limit | alpha 2.3% / Ux 8.0% / Uz 14.5% | 119% at 96 steps, diverges |
| 4 | Transolver++ 3D | `legacy/tsolverpp/` | 05-17 | 3D point cloud, 574,163 | did not converge (ep25/100) | alpha RMSE 0.036 | RMSE 0.27 at 93 steps, degrades |
| 5 | **pure HPM** | `code/` (`window=6`) | 08-11 | 3D point cloud, 574,163 | **50 ep completed** | val 4-step rollout nRMSE **0.2131** | not measured (lt belongs to fwv) |
| 6 | **HPM + FUNWAVE** | `code/` (`window=0`) | 08-04 | 3D point cloud, 574,163 | ep47/50 (hit the 24h wall) | val 4-step rollout nRMSE **0.1403** | **1000 frames without diverging** OK |

**The one-sentence conclusion**: all four earlier lines can do a single step, but every one of them
diverges under autoregression; this generation of HPM, through **a residual base (its own previous
frame / a FUNWAVE prior) + multi-step BPTT + a Delta=0 baseline for every metric**, stabilised a
long rollout for the first time.
The difficulty ordering is always `alpha < p_rgh < Ux < Uz` (`Uy` and `nut` are ineffective).

---

## 1. Data assets (`models/data/`, 48 GB in total)

The 3D and 2D data come from the same OpenFOAM case: `$OCEAN_CASE` (/nfs/hpc/share/coast-lab/OpenFOAM/BA_TingKirby1994_3D_spilling/BA02_9M_Smag_old/0_org)
(interFoam two-phase flow, wave generation + seabed topography, quasi-2D, t = 0-100 s, dt = 0.05 s).

```
models/data/                               48G
├── 2d/                                   5.6G   <- 1.1  2D slices (used by POD / Transolver++ / FNO)
├── fwv/                                  8.5G   <- 1.5  the FUNWAVE reference case TK94 (this one only)
└── 3d/                                    34G
    ├── crop_fields.py / crop_fields.sh          <- generation scripts (read the OpenFOAM volume fields + cellSet)
    └── cropped_0.05/                     34G    <- 1.2  **the only setting currently in use**
```

The number in the name is the **spanwise (y) slice thickness**

| setting | y range | thickness | points | `chunk_XXX_data.npy` | total size |
|---|---|---|---|---|---|
| `cropped_0.05` | [0.2750, 0.3250] | 0.05 m | **574,163** | (100, 574163, 6) f32 = 1.28 GB | 34 G |
| ~~`cropped_0.1`~~ | [0.2500, 0.3500] | 0.10 m | 1,245,500 | (100, 1245500, 6) f32 = 2.78 GB | 28 G |
| ~~`cropped_0.3`~~ | [0.1500, 0.4500] | 0.30 m | 3,732,705 | (100, 3732705, 6) f32 = 8.34 GB | 84 G |

**How to rebuild it**: the crop region comes from an OpenFOAM cellSet. Change the y bounds of that box in `$OCEAN_CASE/system/topoSetDict`, run `topoSet`, then run `crop_fields.sh` (changing `--output`):

```
// topoSetDict: currently the 0.05 setting
box (-2.5 0.275 -0.41) (16.5 0.325 0.16);
//        ^y lower        ^y upper   --  the 0.1 setting uses 0.25/0.35, the 0.3 setting uses 0.15/0.45
```

```bash
topoSet -case $OCEAN_CASE                          # generate subdomainCells from the new box
cd data/3d && sbatch crop_fields.sh <target dir>   # with no argument it writes to cropped_0.05
```

### 1.1 2D slices `data/2d/`

Exported by `postProcessing/sample` (the mid-y cross-section `*_ySlice.raw`), with the script `legacy/transolver++/prepare_data.py` (Polars reads the `.raw`).

| file | shape | description |
|---|---|---|
| `coords_2d.npy` | (149758, 2) float64 | (x, z), x in [-8.7, 16.45], z in [-0.4, 0.15] |
| `fields.npy` | (1001, 149758, 5) float64 | `[alpha.water, Ux, Uz, p_rgh, nut]`, 5.6 GB |
| `times.npy` | (1001,) | t = 0 -> 50 s, dt = 0.05 s |
| `mesh_distribution.png` / `mesh_detail.png` / `alpha_snapshots.png` | | mesh and data inspection plots |

### 1.2 3D crop `data/3d/cropped_0.05/` (35 GB)

**Main data** (produced by `crop_fields.py`; channel order `[alpha.water, Ux, Uy, Uz, p_rgh, nut]`):

| file | shape | description |
|---|---|---|
| `coords.npy` | (574163, 3) float32 | cell centres; x in [-2.50, 16.45], y in [0.275, 0.325], z in [-0.40, 0.148] |
| `chunk_XXX_data.npy` | (100, 574163, 6) float32 | 100 frames per chunk, about **1.38 GB** |
| `chunk_XXX_times.npy` | (100,) float64 | chunk k covers t in (5k, 5k+5] |
| `chunk_010_*` | (1000, ...) | **the exception**: 1000 frames covering t = 50-100 s, for the long-rollout demonstration |

chunks 0-9 -> t = 0-50 s. The conventional split is **train = 1-7, val = 8, test = 9** (chunk 0 contains the start-up transient and is not used).

**Derived assets** (all in the same directory, grouped by purpose):

| path | contents | who uses it |
|---|---|---|
| `lbo/lbo_eigenvectors.npy` | (574163, 128) float32, 294 MB | **HPM's spectral basis**, the first `freq_num=64` columns |
| `lbo/lbo_eigenvalues.npy`, `laplacian_info.txt` | eigenvalues + metadata | see below |
| `prior_ktuned/prior_XXX_{data,valid,times,meta}.npy/json` | priors 001-010, each chunk (100, 574163, 5), about 1.15 GB | **the base of the fwv line** |
| `toffset_scan/c00X.json` | per-chunk t-offset calibration results | read directly by `gen_prior.sh`, zero transcription |
| `slice_y0.30/{slice_cell_map,slice_tri,slice_xz}.npy`, `sub_mesh.vtu` | triangulation and indices of the y=0.3 slice | the tri rendering in `vis.py` |
| `full_cell_ids.npy` | cropped cell -> full-mesh cell id | writing back / comparing against the full mesh |
| `stats_*.npy` | [mean, std] of shape (2, F) | normalisation for training/inference, see 1.3 |
| `slice_sanity_chunk6_f85.png` | slice self-check plot | a one-off verification |

`laplacian_info.txt` records how the spectral basis was obtained:
`cellSet=subdomainCells`, N=574,163, 1,590,604 internal faces after cropping (28,093,814 in the full mesh),
k=128, distance-weighted, eigenvalue range [2.71e-04, 1.03].
WARNING: **once the LBO decomposition is recomputed, the sign ambiguity of the eigenvectors silently degrades old checkpoints** (with no error) --
the comment on `persistent=False` in `hpm_model.py` is there for exactly this.

`prior_XXX_meta.json` is self-describing; chunk 1 as an example:

```json
{"chunk":1, "n_cells":574163, "n_frames":100,
 "channels":["alpha","Ux","Uy","Uz","p_rgh"],
 "x_offset":15.05, "t_offset":0.05, "beta":-0.531, "alpha":"sharp_heaviside", "pnh":true,
 "mglob":1575, "nglob":30, "dx":0.02, "dy":0.02, "plot_intv":0.05,
 "valid_ratio":0.890, "cells_always_valid":510767, "cells_never_valid":62876,
 "invalid_outside":32600, "invalid_below_bed":0, "dry_cells_total":63070,
 "dry_x_min":13.23, "dry_x_max":16.43, "horiz_points_outside_grid":114, "frames_missing":0}
```

The key points: **the prior is invalid for about 11% of the cells** (dry regions + outside the domain, concentrated nearshore at x>13.2),
and `frames_missing: 0` shows that no frame alignment was dropped; `prior_XXX_valid.npy` is a per-frame, per-point bool mask for diagnostics.

### 1.3 The naming rule for `stats_*.npy`

The filename is generated by `signature()` in `schema.py` and **encodes which channel combination these statistics belong to**:

```
stats_{chunk range}_{channel list}.npy
   stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.npy        pure HPM line (5 channels, raw velocity)
   stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.nut.npy    pure HPM line (6 channels, including nut)
   stats_c1-7_alpha.Ux.Uz.p_rgh.npy           fwv line, non-alphaU
   stats_c1-7_alpha.Uxw.Uzw.p_rgh.npy         fwv line, **alphaU weighted** ('w' suffix)
   stats_c1-7_u{0,1}_nut{0,1}.npy             the old naming (u1 = alpha-weighted velocity), abandoned
   stats_c6_*.npy                             statistics from chunk 6 only, for debugging
```

### 1.4 Which line uses which data

| line | data | note |
|---|---|---|
| POD / POD-LSTM | `$OCEAN_CASE/postProcessing/sample` (the original raw) | does not go through `models/data` |
| Transolver++ 2D | `data/2d/{coords_2d,fields,times}.npy` | point cloud directly |
| FNO | `data/2d/*` -> `legacy/fno/processed_data/` (interpolated onto a regular grid, 6.3 GB) | one extra interpolation step |
| Transolver++ 3D | `data/3d/cropped_0.05/chunk_*` + `stats.npy` (absent) | see 5.4-2 |
| pure HPM | `cropped_0.05/chunk_*` + `lbo/` + `stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.npy` | |
| HPM + FUNWAVE | all of the above **+ `prior_ktuned/`** + `stats_c1-7_alpha.Uxw.Uzw.p_rgh.npy` | |

Also: `models/FUNWAVE-TVD/` (256 MB) is the source of the Boussinesq solver -- a clean clone of a third-party repository (`Version_3.6`, **not one byte changed**), not committed; `./setup.sh` pulls it when absent. The actual output used for the prior is at `/nfs/hpc/share/coast-lab/FUNWAVE/TingKirby1994_3D_spilling_2/output`.

### 1.5 The FUNWAVE reference case `data/fwv/` (8.5 GB, TK94 only)

```
data/fwv/
├── TK94/                  the reference case (AMP_WK=0.0635, SLP=1:35); the binary is built here
├── make_cases.py          builds new cases from TK94 (changing only the AMP_WK and SLP lines of input.txt)
└── wk_check.py            wave-generation parameter self-check
```

TK94 shares its parameters with the CFD ground truth and is the default case for `vis_adp.sh` and web-demo. All 8.5 G of it is output;
**the inputs are committed, the output is not** -- `input.txt` + `gauges.txt` are tracked, with a precise exception for them in `.gitignore`;
`LOG.txt`, the `funwave--gnu-parallel-single` binary and the field output are all excluded.

> **This used to be a parameter scan over 11 cases**: five wave heights `H0381`~`H0610` (with `SLP` fixed at 1:35),
> plus six varied slopes, `S325` = 1:32.5 and `S375` = 1:37.5. On 2026-08-20 this direction was
> dropped -- the case data (93.5 G) and every prior and rendered artifact under `results/fwv/` (124 G)
> were deleted together, and the repository keeps only the inputs of TK94.
>
> **To redo the scan**: `make_cases.py` copies TK94 and changes only the `AMP_WK` / `SLP` lines (everything
> else is byte-identical, so there are no hidden variables between cases); run FUNWAVE to obtain
> `output/`, then generate the prior with `vis_adp.sh STAGE=prior`. Neither the raw output nor the
> priors are distributed with the repo, so they have to be produced.

The strict principle of `make_cases.py`: **only the `AMP_WK` and `SLP` lines change; everything else is byte-identical to TK94**,
`gauges.txt` is copied verbatim, and the executable is symlinked to the one built for TK94 -- so the
difference between cases is exactly the two numbers in the filename, with no other hidden variable.

---

## 2. POD / POD-LSTM (`legacy/pod-lstm/`, output in `$OCEAN_DATA`)

> Sections 2-5 are retired lines, recorded for the record only. Their output is not distributed with
> the repo (see 7.2); before re-running any of them, read [legacy/README.md](legacy/README.md).

First POD compresses the field into modal coefficients, then an LSTM predicts how the coefficients
evolve, and finally the physical field is reconstructed.
`pod_decomposition.py` (sklearn PCA) processes 1000 frames x 149,758 points, keeping 300 modes per
variable, producing about 1.8 GB (modes / coefficients / singular values / energy spectra).

**Energy convergence** (modes needed to reach that energy):

| variable | 90% | 95% | 99% | share of mode 1 |
|---|---|---|---|---|
| `p_rgh` | 5 | 11 | 49 | 40.2% |
| `alpha.water` | 21 | 51 | 230 | 26.5% |
| `Ux` | 27 | 67 | 270 | 26.4% |
| `Uz` | 32 | 100 | >300 | 25.5% |
| `nut` | 189 | >300 | >300 | 16.6% |

This table is the basis for the channel choices of every later line: pressure is low-rank, the free
surface and the velocities are intermediate, and `nut` is essentially incompressible -- later models
either down-weight `nut` (`loss_weight=0.1`) or turn it off entirely.

The LSTM is trained in coefficient space, 218 dimensions (alpha 51 + Ux 67 + Uz 100, each at 95% energy),
and the best configuration is v9 (3 layers / hidden 128 / window 40 / 470,490 parameters, early stop ep51).

| | alpha | Ux | Uz |
|---|---|---|---|
| coefficient space, single step | 10.4% | 11.9% | 24.6% |
| coefficient space, autoregressive | 76.3% | 78.2% | 87.3% |
| field space, single step, rel L2 | 51.5% | 139.4% | 163.8% |
| field space, autoregressive, rel L2 | 51.3% | 141.0% | 163.3% |

**Judged a failure**: about 10% error in coefficient space is amplified past 50% in field space by
the reconstruction.
Note: the reconstruction figures above come from `lstm_results_v6_nop_wm` rather than the best v9;
a fair comparison would need a re-run.

---

## 3. Transolver++ 2D (`legacy/transolver++/`)

Operator learning directly on the original 149,758 mesh points, with no interpolation. The core is
Eidetic Physics-Attention: each point is projected into a token, softly assigned by an
input-dependent-temperature Gumbel-Softmax to 64 physical slices, attended to across slices, then
de-sliced back to the points, reducing the complexity from O(N^2) to O(G^2).

Configuration: 4 layers / hidden 128 / 4 heads / 64 slices, 330,903 parameters; train t in [10,45),
test t in [45,50]; early stop at ep180 (limit 200, patience 30), 1.6 h on an H200, best val 0.034274 @ ep150.

| field | normalised MSE | single-step rel L2 |
|---|---|---|
| alpha | 0.00295 | 3.47% |
| Ux | 0.02866 | 16.30% |
| Uz | 0.07125 | 26.37% |

Autoregressive rel L2 over 100 steps: 7.2% -> 20.8% (5 steps) -> 33.5% (10) -> 50.4% (20) -> 86.6% (50) -> 111.2% (100).

**Conclusion**: it reaches single-step accuracy of the same order as FNO with about 1/400 of the
parameters, so the point-cloud-plus-slice-attention route is viable; but the autoregressive error
grows roughly linearly and it is unusable past 50 steps.

**Known defects** (to be dealt with before re-running):

1. Line 220 of `transolver_pp.py` uses `math.log` while the top of the file does not `import math`, so constructing the model raises `NameError`.
2. `configs/default.yaml` (5 in / 4 out) does not match the 3-channel rollout in `results/` -- the saved results were produced by the single-frame model, before the change to a sliding-window version. Reproducing the table above means going back to the single-frame version.

---

## 4. FNO (`legacy/fno/`)

The unstructured mesh is interpolated onto a 4096x128 regular grid (Delaunay barycentric
coordinates, with the weights precomputed once and reused), and a standard 2D FNO does "the past 5
frames -> the next frame": the input is 5 frames x 4 fields + a topography mask, 21 channels in
total, the output is 4 channels, and the loss is computed inside the mask only.

| run | channels | parameters | trained to | best val MSE |
|---|---|---|---|---|
| `2026-05-12/22-31-06` | 3 fields, 16 input channels | 134.2 M | ep~175 (4 h limit) | ~0.0119 (best.pt is ep102) |
| `2026-05-13/17-54-45` | 4 fields, 21 input channels | ~134 M | ep30 (killed) | 0.00932 (best.pt is ep21) |

Autoregressive rollout (96 steps from t=45 s, relative L2):

| model | Step 1 | Step 49 | Step 96 |
|---|---|---|---|
| 3 fields (ep102) | alpha 2.4% / Ux 8.2% / Uz 14.6% / total 4.3% | total 87% | total 94% |
| 4 fields (ep21) | alpha 2.3% / Ux 8.0% / Uz 14.5% / total 3.7% | total 96% | total 119% |

**Conclusion**: the best single-step accuracy of the four earlier lines, but the rollout fails within
about 50 steps (2.5 s).
The train loss keeps falling (1e-3 -> 3e-4) while val stalls at 1.2e-2 -- overfitting compounded by
the mismatch between single-step training and multi-step inference.
`modes1=256` was tried; it doubled memory and time with no improvement.

---

## 5. Transolver++ 3D (`legacy/tsolverpp/`)

Changed from the 2D version to: coordinates (x, y, z), 574,163 points, 6 fields; a 21-dimensional
temporal input with `window=6` (weighted history plus first and second differences); the output
changed to a residual (`pred = fields[:,:,-1] + Delta`); weighted MSE (`p_rgh` / `nut` weight 0.1);
`rollout_steps=4` multi-step unrolling (the first 3 steps detached); split by chunk (train 1-7, val
8-9). The model is hidden 256 / 6 layers / 8 heads / 64 slices -> 2,491,075 parameters.

**Not a single training run on this line ever finished**: about 45 min per epoch, and a 20 h limit
allows only about 25 epochs, while the configuration asks for 100.

| run | configuration | reached | best val (weighted MSE) |
|---|---|---|---|
| `2026-05-19/03-21-52` | default | E024 | 0.205489 @ E020 |
| `2026-05-19/12-58-01` | `dropout=0.2 weight_decay=1e-3` | E025 | 0.204876 @ E025 |

Rollout animation (midplane, 377,215 points, 93 steps ~= 4.65 s, alpha RMSE): chunk 6 (in the
training range) starts at 0.0389 / ends at 0.2714 / mean 0.2096; chunk 9 (the validation range)
0.0357 / 0.2760 / 0.2161.

**Conclusion**: the two runs agree almost exactly, so the bottleneck is training time rather than
regularisation; the training and validation metrics are close, which is underfitting. An RMSE of
0.27 shows up as a blurred interface, but it is stable compared with the outright divergence of the
2D version -- **residual output and multi-step training were both inherited by the HPM line**.

**Known defects**:

1. `cd <repo>/tsolver3d` in `vis.sh` is a leftover from a rename and should be `tsolverpp` (it currently works by the luck of staying in the submit directory).
2. `data/3d/cropped_0.05/stats.npy` no longer exists (see 1.3). When `dataset.py` finds it missing it scans every chunk in the directory, recomputes (reading about 13 GB) and writes it back. `stats_c1-7_alpha.Ux.Uy.Uz.p_rgh.nut.npy` can be copied in its place, but note that what `dataset.py` computes itself includes chunks 0 and 10, and that the variants carrying `u1` / `Uxw` are alpha-weighted velocities and must not be mixed in.
3. `train.sh` hard-codes `--nodelist=dgxh-3`, which queues indefinitely when that node is busy; `epochs: 100` in `config.yaml` does not match the 20 h limit (it would need about 75 h).
4. Two early failures kept for the record: one version where train/val loss did not move for 40 epochs (no gradient flow); and a 12,991,637-parameter version that went OOM on an H200 -- a single `gumbel_softmax` needs 28.5 GiB, the bottleneck being `(B, heads, N, slice_num)`.

---

## 6. HPM (`models/code/`) -- the current main effort, two lines

`code/` is **the only directory being iterated on**. It converges every earlier lesson into one
setup: a residual base + multi-step BPTT + **a Delta=0 baseline for every metric**.

> `code/web-demo/` is an interactive demo page hanging off this line (it runs the whole pipeline for
> an audience) and is **outside the scope of this document**; it has its own `README.md`.

The two lines share all of the code, and **the only structural difference is what the residual is added to**:

| | base | state | input width | R | scheduled sampling | status |
|---|---|---|---|---|---|---|
| **pure HPM** | its own previous frame | W=6 sliding window | F (`window=6`) | 4 | none (`ss=false`, p constantly 0, fully autoregressive) | earlier stage |
| **HPM + FUNWAVE (fwv)** | **prior(t)** | single-frame feedback slot | 2F (`window=0`) | 4 | p 1.0->0.1 (first 60% of epochs) | **active** |

They are distinguished by the single parameter `data.window` (`0` = fwv, `>=3` = pure HPM). The two
are not freely combinable: only `window>0` has history frames available as a base. `build_policy`
asserts this, so a wrong combination fails outright instead of silently running as something else.

### 6.1 The model itself, `hpm_model.py`

HPM = Holistic Physics Mixer (ICML 2025), whose core is the **Calibrated Spectral Mixer**:

1. Use the **LBO eigenvectors** precomputed from the OpenFOAM graph Laplacian as a fixed spectral basis (1.2; the first `freq_num=64` columns);
2. A learnable gate network predicts a "frequency preference" per point -> `eigens = gate * basis`;
3. Forward transform into the spectral domain -> LayerNorm + a linear mix in the spectral domain -> inverse transform back to the physical points.

Two engineering optimisations: the spectral basis is stored **once** as (N, G) and broadcast (rather
than once per head or per block), and `persistent=False` keeps it out of the checkpoint -- memory
went from 50 GB to **42.8 GB**, with bit-identical output. The persisted basis in the early
checkpoints has all been stripped out (see 7.4), and the loading path no longer does any
compatibility handling.

Model size: `n_hidden=128 / 6 layers / 8 heads / freq_num=64 / mlp_ratio=2` -> about **740 thousand
parameters** (740,852 on the fw line, 742,780 on the pure line).

### 6.2 The pure HPM line (`window=6`, fully autoregressive)

- Input is coords(3) + temporal features 3F (macro-weighted history + first difference + second difference), structurally the same as the 3D Transolver++ of section 5;
- The base is the last frame of the window, and the model only outputs Delta;
- **`ss=false` -> p is constantly 0, and every training step is fed the model's own prediction** (= the deployment condition), so there is no notion of exposure bias and no cold-start problem either (the window is itself the state);
- R=4 true BPTT (strictly ordered within a sequence, random start point between sequences).

**Results**:

| run | channels | reached | best val (4-step rollout nRMSE) |
|---|---|---|---|
| `hpm_bl_h128` (nut on, 6 channels) | alpha, Ux, Uy, Uz, p_rgh, nut | ep8 (A40, 1.8 h/ep, interrupted) | 0.1881 @ ep6 |
| `hpm_no-nut_h128` (nut off, 5 channels) | alpha, Ux, Uy, Uz, p_rgh | **all 50 ep completed** (H100, 1678 s/ep) | **0.2131 @ ep16** |

WARNING: the two have **different channel sets** (6 vs 5), so the aggregate nRMSE is not in the same space and they cannot be compared directly.

`hpm_no-nut_h128` per channel (final round ep49, model / Delta=0 baseline; a check mark = better than the baseline):

| alpha | Ux | Uy | Uz | p_rgh |
|---|---|---|---|---|
| 0.071 / 0.189 OK | 0.206 / 0.353 OK | **1.000 / 1.031** | 0.361 / 0.604 OK | 0.071 / 0.286 OK |

- alpha and p_rgh reach about 1/3 of the baseline, which is good;
- **`Uy ~= 1.0`**: in a quasi-2D case Uy is itself close to noise, so the model cannot learn it and should not -- this is precisely the basis for turning Uy off on the fwv line;
- **val rises back from 0.2131 at ep16 all the way to 0.2367 at ep49 while train is still falling** -> clear overfitting; 50 epochs is too many for this line, and early stopping or more regularisation is the next step.

### 6.3 The HPM + FUNWAVE line (`window=0`, built on fwv as a skeleton)

**The core idea**: rather than have the network extrapolate in time from nothing, let
**FUNWAVE (a Boussinesq wave solver) supply the physical prior field prior(t) at time t first**, and
have the network learn only the correction Delta from "prior -> CFD ground truth". The base changes
from "its own previous frame" to "a field resupplied at every step by an external physical model", so
the error no longer accumulates purely on its own.

#### The prior production pipeline (three stages, all in `code/`)

```
stage 1  scan.sh      -> scan_toffset.py            per-chunk t-offset calibration  -> toffset_scan/c00X.json
stage 2  gen_prior.sh -> gen_prior.py + lift.py     Nwogu profile lifting into 3D   -> prior_ktuned/
stage 3  at training time, dataset.py drops Uy and takes 4 channels
```

- **`lift.py` (the lifting operator)**: lifts the 2D state (eta, u, v) into 3D following Nwogu (1993):
  `Ux = u + (za-z)dA/dx + 1/2(za^2-z^2)dB/dx` (a quadratic profile), `Uz = -(A + zB)` (continuity),
  `p = rho*g*eta - rho[A_dot(eta-z) + 1/2 B_dot(eta^2-z^2)]`, with `alpha` a sharp Heaviside. **Zero free parameters**; the profile is analytic in z, so no interpolation is done in the z direction; dry cells -> NaN, neither zeroed nor extrapolated.
- **`scan_toffset.py` (temporal registration)**: the time origins of FUNWAVE and CFD are not aligned, so the optimal frame offset is scanned chunk by chunk.
  Measured: `best_k[1..9] = [1,3,5,5,4,3,2,4,6]` -- **not monotonic**, so there is no "drift model" to apply and it has to be calibrated chunk by chunk.
  chunk 9 had previously been judged invalid (Uz nRMSE = 1.02 > 1 under a fixed offset); after recalibration k=+6 -> 0.957,
  so **it was a registration error, not a data problem**, and chunk 9 was restored as a valid test chunk.
  Each `c00X.json` also carries per-channel `k_rmse / k_corr / k_subframe / contrast / flat` diagnostics,
  and `Uy` is always `flat: true` (the curve is flat, so it is not counted) -- the correct signal for a quasi-2D case.
- The x-direction offset `XOFF=15.05` is treated as a known input and not calibrated.

#### Prior quality diagnostics -> channel selection

(chunks 1-9, `prior_ktuned`, raw space, averaged across chunks)

| channel | nRMSE_prior | corr | verdict |
|---|---|---|---|
| alpha | 0.410 | 0.918 | effective |
| p_rgh | 0.647 | 0.806 | effective |
| Ux | 0.896 | 0.521 | weakly effective |
| Uz | 0.930 | 0.403 | weakly effective |
| **Uy** | 1.002 | -0.001 | **ineffective** -- worse than a constant baseline; the prior is adding noise -> turned off |
| **nut** | — | — | **structurally there is no prior**: FUNWAVE is inviscid Boussinesq -> turned off |

So the fwv line keeps only 4 channels: `alpha, Ux, Uz, p_rgh`.

#### The feedback slot, SS, and alphaU

`window=0` means the model has no history, so a **feedback slot** is introduced:

- `feedback=none` -> input = `[prior]`, F channels (a plain "prior field -> CFD field" single-step mapping, with no rollout and no cold start);
- `feedback=self` -> input = `[prior | x_f*m]`, 2F channels; masking is done by `x_f*m` (an arithmetic identity, so no extra mask column is needed).

Once there is a slot there is exposure bias, hence SS: `p` = the probability of feeding GT, annealed
1.0 -> 0.1 (over the first 60% of epochs), with **val always at p=0, matching deployment**.
`alpha_weighted=true` on `Ux/Uz` (multiplying by the water volume fraction in physical space) is the
**current best arm** -- velocity in the air region should not enter the evaluation.

#### Results

| run | reached | best val (4-step pure rollout nRMSE) |
|---|---|---|
| `hpm_fw_aU_h128 / 2026-08-04_14-37-31` | ep47/50 (hit the 24 h wall) | 0.1512 @ ep22 |
| `hpm_fw_aU_h128 / 2026-08-12_15-31-45` | ep47/50 (hit the 24 h wall **again**) | **0.1403 @ ep19** |

Per channel (final round ep47, model / Delta=0 baseline):

| alpha | Ux | Uz | p_rgh |
|---|---|---|---|
| 0.221 / 0.370 OK | 0.436 / 0.619 OK | 0.602 / 0.729 OK | 0.318 / 0.562 OK |

**All four channels beat the "do not learn, just use the prior" baseline** -- the minimum evidence that this line works.

A 100-frame cold-start rollout on chunk 9 (test), RMSE on the y=0.3 slice (107,466 cells), using best.pt from ep19:

| field | teacher forcing (final frame / mean) | rollout (final frame / mean) | gap = exposure bias |
|---|---|---|---|
| alpha | 0.0415 / 0.0365 | 0.1614 / 0.1309 | +0.120 |
| alphaUx | 0.0551 / 0.0390 | 0.1405 / 0.1116 | +0.085 |
| alphaUz | 0.0163 / 0.0145 | 0.0499 / 0.0465 | +0.034 |
| p_rgh | 34.07 / 27.09 | 152.34 / 119.70 | +118.3 |

The error distribution of the same rollout (`DIFF=both`, re-run 2026-08-16, S = GT full scale):

| field | S | bias | MAE | max\|Delta\| | MAE% | max% |
|---|---|---|---|---|---|---|
| alpha | 1.000 | -0.0044 | 0.0298 | 1.073 | 2.98% | 107.3% |
| alphaUx | 2.700 | -0.0278 | 0.0524 | 2.224 | 1.94% | 82.4% |
| alphaUz | 1.255 | -0.0006 | 0.0174 | 0.734 | 1.39% | 58.5% |
| p_rgh | 1734.8 | -10.11 | 53.15 | 1305 | 3.06% | 75.3% |

**The bias is negative in all four fields but very small** (-0.04% to -1.03%) -- there is no
systematic over- or under-prediction, and the error is locally structural rather than an overall
drift. Meanwhile max|Delta| reaches the order of the full scale (alpha 107%, i.e. some cells are
completely wrong) while the MAE is only 1.4-3.1%: **the error is highly concentrated in a small
number of cells**. Exactly where those cells fall (whether they are the breaking region discussed
under "error decomposition" below) has to be read off the video and has not yet been checked frame
by frame -- the videos are in
`results/vis/pred/hpm_fw_aU_h128/2026-08-12_15-31-45_diffboth/`.

**The long rollout (the single most important result)**: **1000 consecutive frames (50 s of physical
time) on chunk 10 without collapsing**, at **722 ms/frame** of inference (574,163 cells, one GPU).
Each of the two runs has a
`results/vis/lt/hpm_fw_aU_h128/<timestamp>/longterm_chunk10_alpha_tri.mp4`.
**This is the first time any of the lines achieved long-term stability.**

#### Error decomposition

The autoregressive rollout error of the fw line is about **0.125**, which splits in two:

- **a floor of about 0.10** = the FUNWAVE prior failing in the breaking region. Boussinesq theory
  itself breaks down where the wave overturns, and this part is not something the model can fix;
- **an accumulation of about 0.04** = exposure bias, which is what SS addresses and all it addresses.

**This decomposition sets the priority**: further SS tuning can win back 0.04 at most; breaking
through the 0.10 means changing the prior.

### 6.4 Engineering conventions

- **One yaml + CLI overrides**: `config.yaml` is the single configuration file, and its header carries a **reproduction table** (one command per run that was performed). The discipline is "editing the yaml = defining a new baseline; run variants from the CLI".
- **`override_dirname`**: a CLI override automatically enters the hydra directory name and the wandb run name, while editing the yaml does not -- so running two variants by editing the yaml produces a name collision.
- **`schema.py` as the single source of truth**: channel enable / update_rule (delta | frozen) / loss_weight / alpha_weighted all derive from here, including the **stats filename signature** (1.3). The files on disk are always 6-channel and columns are selected **by name** -> **an ablation needs no regenerated data**.
- **The `run.sh pure` shortcut**: switching to the pure HPM line takes four groups of parameters (window/feedback/SS/channels) and is wrapped up; forgetting `rollout.ss=false` is caught by a `build_policy` assertion.
- **How `vis.sh` locates things**: explicitly with `CONFIG=... CKPT=...`, or resolved from `results/train/` with `RUN=runname [TS=timestamp]` (omit TS and it takes the newest one containing best.pt); it refuses to overwrite a non-empty output directory unless `FORCE=1`.
- **`vis.py` sub-commands**: `gt` (plain data inspection) / `align` (registration check, before training) / `lift` (render the prior only, computed live from FUNWAVE) / `pred` (two rows, GT|pred, shared by both lines, with a tf/rollout gap self-check) / `nofb` (three rows for the no-feedback arm) / `lt` (long rollout, streaming, fwv only). **`--diff` hangs off `pred` alone** (the other sub-commands have no GT, so there is no Delta to compute); through `vis.sh` it is `DIFF=abs|pct|both`, see "Error visualisation" in 6.6.

### 6.5 Directions that were ruled out

| direction | why it was ruled out |
|---|---|
| Clifford / geometric algebra | the domain has no rotational symmetry (gravity anisotropy + sloping bed + bounded domain); `mlp_trans_weights` shows no learnable structure after training |
| a pointwise PDE time-derivative residual | Courant C~=1.6 (near the air region), so the residual form does not hold |
| a differentiable solver | MULES is not differentiable |
| global invariant constraints | an open dissipative domain does not conserve |
| symmetry / equivariance constraints | the domain is anisotropic |
| a mask channel (2F+1) | masking is already done by `x_f*m` (an arithmetic identity); a mask column only states it explicitly, which is efficiency rather than necessity |

### 6.6 How to run it

```bash
cd <repo>/code && mkdir -p logs

# --- the fwv line (default = the current best arm, hpm_fw_aU_h128) ---
sbatch run.sh                                    # prior + self-feedback + SS(R=4) + alphaU
sbatch run.sh rollout.R=8                        # a variant (enters the directory/run name automatically)
sbatch run.sh rollout.feedback=none rollout.R=1 \
      data.channels.1.alpha_weighted=false \
      data.channels.3.alpha_weighted=false       # the no-feedback baseline arm

# --- the pure HPM line ---
sbatch run.sh pure                               # equivalent to the four groups of parameters, see run.sh
sbatch run.sh pure data.channels.5.enabled=false # nut ablation

# --- prior production (fwv line only, once before training) ---
sbatch scan.sh                                   # stage 1: per-chunk t-offset calibration
sbatch --dependency=afterok:<scan_jobid> gen_prior.sh   # stage 2: lift into prior_ktuned/

# --- visualisation ---
RUN=hpm_fw_aU_h128 sbatch vis.sh                 # default SUB=pred, chunk 9
SUB=lt RUN=hpm_fw_aU_h128 CHUNK=10 sbatch vis.sh # long rollout (fwv only)
python vis.py align --fw-dir <fw>/output --chunk 9        # registration check before training

# --- error rows (DIFF, pred only; not rendered by default, see "Error visualisation" below) ---
DIFF=both STYLE=tri RUN=hpm_fw_aU_h128 \
  FEATURE=hpm_fw_aU_h128/<timestamp>_diffboth sbatch --partition=gpu vis.sh
```

#### The output paths of `vis_adp.sh` can be overridden

`OUTROOT` / `PRIORROOT` / `VISROOT` / `LIFTROOT` / `CKDIR` are all written as `${VAR:-default}`, so
**passing nothing gives exactly the original behaviour** (`results/fwv/{priors,vis,lift}` +
`results/train/...`). Pass them and the output goes elsewhere, with the two not interfering:

```bash
STAGE=prior CHUNK=10 CASES='TK94' \
  PRIORROOT=$REPO/results/web/priors sbatch ... vis_adp.sh
```

These variables were added so that the demo page in `code/web-demo/` writes its output into its own
`results/web/` rather than mixing it into the ADP scan line's `results/fwv/` (that directory was
emptied on 2026-08-20 and is rebuilt if it is run again).
Running ADP by hand needs none of them.

#### Error visualisation (`--diff` / `DIFF=`)

`pred` renders two rows by default (GT | pred). Passing `DIFF=` appends error rows below,
Delta = pred - GT, animated frame by frame, so the spatial distribution of the error can be seen
rather than just a single RMSE number:

| `DIFF=` | what it draws | colour scale | purpose |
|---|---|---|---|
| `abs` | Delta, in physical units | adaptive +-p99\|Delta\| (tuned with `DIFF_PCT=`) | to see structure; however small the error, it fills the frame |
| `pct` | Delta% = Delta/S x 100 | fixed +-100% | side-by-side comparison across runs / checkpoints |
| `both` | render both (4 rows in total) | | |

Companion parameters: `PCT_SCALE=range|rms|p99` (the denominator S of Delta%, default `range` = GT
full scale), `DIFF_PCT=99` (the abs row only), and `ROW_H=` (the height of each row in inches).
The error rows use their own symmetric coolwarm colour scale (red = too high, blue = too low,
white = accurate) rather than sharing the main one -- the main scale would squash negative errors
into its lower end and render them invisible.

Three things to note:

1. The difference between `abs` and `pct` is the colour scale, not the denominator -- S appears once in the numerator and once in the denominator of the colouring position and cancels, so under an adaptive colour scale the two rows are the same picture; anchoring the `pct` row at +-100% is exactly what makes it comparable across runs.
2. `DIFF=both` is 4 rows, and the default `row_h=10.8` -> 4320 px, past the 4096 hardware-decode line of many players. When `ROW_H` is not given explicitly, `vis.sh` drops it to 10.0 (= 4000 px).
3. On alpha, `both` is essentially redundant (S=1.0 and the p99 adaptive scale is +-0.894, only 11% away from +-100%), so `abs` is enough; what really separates them is Ux (+-0.475 vs +-2.70) and p_rgh (+-556 vs +-1735).

### 6.7 WARNING: known problems

1. **No random seed is fixed**. Re-running the same configuration moves `best_val` appreciably (three measurements: 0.1549 / 0.1512 / 0.1403), and the epoch at which the best appears moves a great deal too (ep10 / ep22 / ep19), while the convergence endpoint of the last 15 epochs almost coincides.
   **"Just run it again" does not currently constitute comparable evidence**; to use it as evidence the seed has to be fixed first.
2. **The numbers in the tables above were produced in the old environment and are not bit-comparable with the current one**. On 2026-08-19 the numpy in `ocean` was switched from a conda-forge build to a PyPI build (**the same version number, 2.2.6; only the compiled artifact differs** -- BLAS went from conda's openblas to the one bundled in the wheel). torch / torchvision and the other 16 dependencies were untouched, the heavy GPU work is entirely in torch's hands, and numpy only takes part in data preprocessing and statistics, so the effect is expected to be in the last floating-point bit; but strictly speaking, numbers across that environment boundary can only be compared qualitatively, not treated as an exact reproduction.
   Getting new numbers means re-running a round in the current environment (compounded with item 1: the seed has to be fixed first).
3. **alphaU and non-alphaU nRMSE are not in the same space** and cannot be compared directly (the normalised quantity changes from `Ux` to `alphaUx`, with a Delta=0 baseline of 0.896 vs 0.619, see 1.3). Cross-arm comparison is only valid on a metric in a shared space -- the raw-space slice-RMSE of `vis.py pred`, or the nearshore shape region by region. By the same token, **the val of the pure line and the fw line cannot be compared directly either**.
4. **50 epochs x ~1780 s ~= 24.7 h hits the 24 h SLURM wall** -- **both** fw runs stopped at ep47.
   The last 15 epochs are already flat, so dropping to 30 would finish without losing anything. The pure line at 1678 s/ep just barely completes 50.
5. `update_rule: flux_div` is a reserved placeholder and raises `NotImplementedError`.
6. Never combine `update_rule=delta` with `loss_weight=0.0`: an unsupervised head will contaminate the rollout. To remove a channel, use `frozen` or `enabled: false`.

---

## 7. Results and output (`models/results/` and each line's output)

### 7.1 The current state of `results/`

Since 2026-08-22, **the only thing under `results/` that travels with the repo is the demo's default
weights** (`web/model/`, 8.6 M); everything else was moved out and packaged:

| contents | where it is now |
|---|---|
| `train/` 87 M, `vis/` 56 M | `results_20260822.tar` (194 M / 69 files) |
| `web/` demo-page state (intermediate fields / mp4 / submission records) | not packaged -- one click rebuilds it, see `code/web-demo/README.md` |
| `fwv/` output of the ADP scan line, 124 G | deleted entirely on 2026-08-20 (see 1.5); running `vis_adp.sh` again rebuilds it |

What the repository keeps is a skeleton of empty directories with a `.gitkeep`, each saying which
package its contents are in and how to restore them (`./archive/restore.sh`; for the list see
[archives.tsv](archives.tsv)). After a restore the structure is:

```
results/
├── train/<runname>/<override_dirname>/<timestamp>/  70M
│   ├── .hydra/{config,overrides,hydra}.yaml         <- the **actual** configuration of this run (the source of truth)
│   ├── checkpoints/{best,latest}.pt                 <- 9.0 MB each (no longer contains the LBO basis)
│   └── train.log                                    <- usually empty; the real log is in code/logs/
└── vis/<sub>/<runname>/<timestamp>/                 44M
    ├── pred:  compare_chunk9_<field>_pred_{tri,scatter}.mp4
    └── lt:    longterm_chunk10_alpha_tri.mp4
```

The name of the last directory level is decided by `FEATURE=` (default `<runname>/<timestamp>`). When
re-running the same ckpt under a different rendering setting, **add a suffix and open a new directory**
rather than overwriting with `FORCE=1` -- the error-row run, for instance, used
`FEATURE=hpm_fw_aU_h128/2026-08-12_15-31-45_diffboth` (12M, one 4-row tri video for each of the four fields).

The runs that currently exist:

| runname | line | timestamp | best_val | note |
|---|---|---|---|---|
| `hpm_fw_aU_h128` | fwv | `2026-08-04_14-37-31` | 0.1512 @ ep22 | has pred + lt videos |
| `hpm_fw_aU_h128` | fwv | `2026-08-12_15-31-45` | **0.1403 @ ep19** | has pred + lt videos, plus a `_diffboth` set of error rows; the current best |
| `hpm_bl_h128` | pure | `...window-6.../2026-08-11_22-58-29` | 0.1881 @ ep6 | only reached ep8 |
| `hpm_no-nut_h128` | pure | `...enabled-false.../2026-08-12_16-24-16` | 0.2131 @ ep16 | 50 ep completed, no vis |

That long `override_dirname` directory name is not noise; it is exactly **this run's diff against the
default configuration** (for example `data.window-6_rollout.feedback-none_rollout.ss-false...` is
recognisable at a glance as the pure HPM line).

**Intermediate output**: `vis.py pred` also saves `compare_chunk9_<field>_preds.npy`
((100, 574163, 4) float32 = **918 MB each**, so that "re-rendering with a different colour scheme
does not require re-running inference") and the `_rmse{,_tf}.npy` curves. This set of intermediates
has been cleared and only the mp4 files remain -- when re-rendering is needed, just run `vis.sh` again.

### 7.2 Where the output of the older lines is

| line | location of the output | size | contents |
|---|---|---|---|
| POD | `$OCEAN_DATA/pod_results/` | 1.8 G | modes / coefficients / energy spectra / `mode_summary.txt` |
| POD-LSTM | `$OCEAN_DATA/lstm_results_v*/` x9 | small | `best_model.pt`, `results_summary.json`, `var_info.json`, curve plots |
| field reconstruction | `$OCEAN_DATA/reconstruction_results/` | medium | predicted/true field npy + `reconstruction_errors.json` + snapshot plots |
| Transolver++ 2D | `legacy/transolver++/results/` | 354 M | `best_model.pt`, `training_history.json`, `rollout_{pred,gt}.npy`, `figs/` |
| FNO | `legacy/fno/outputs/` (inside `legacy_20260822.tar`) | **16 G** | `best.pt` + `epoch_*.pt` (134 M parameters -> about 1.6 GB per ckpt) + `visualizations/` |
| FNO intermediate data | `legacy/fno/processed_data/` (same package) | 6.3 G | the interpolated regular-grid data |
| Transolver++ 3D | `legacy/tsolverpp/outputs/<date>/<time>/checkpoints/` | 129 M | `best.pt` / `latest.pt`; the mp4 files sit directly in the `legacy/tsolverpp/` root |
| the previous HPM generation | `legacy/hpm/`: `outputs/` 429 M and `vis/` 487 M each have their own package (`legacy_hpm_*.tar`); `wandb/` 372 M was moved out unpackaged; **`fwv/` 58 M stays in the repository** | | the implementation and output from before the refactor. The 40 rendered mp4 files in `fwv/` were taken into the repository on 2026-08-23 -- they are the only things in the whole tree that were in neither git nor a package |

**Where the logs are** (to find the complete output of one run): `legacy/fno/logs/`,
`legacy/tsolverpp/logs/`, `legacy/transolver++/tsolver_pp_*.log`, `code/logs/`
(`hpm_<jobid>.log` + `.err`), `$OCEAN_DATA/*.log`. Hydra's `train.log` is essentially empty --
**the real log is SLURM's**.

### 7.3 The standard procedure for tracing a run

1. Find the run under `results/train/<runname>/<override_dirname>/<timestamp>/`;
2. Read `.hydra/overrides.yaml` -- **what was changed this time**; read `.hydra/config.yaml` -- **the complete actual configuration** (more reliable than reading `config.yaml` in the repository, which changes over time);
3. Look at `code/logs/hpm_<jobid>.log` for the per-epoch curves and the per-channel nRMSE against the baseline;
4. `RUN=<runname> TS=<timestamp> sbatch vis.sh` to reproduce the visualisation (`vis.sh` fetches the ckpt and config from the same path itself).

### 7.4 Disk and cleanup

On 2026-08-22/23 the data and output were all moved out of the repository into five packages (see
[archives.tsv](archives.tsv)). A clone now gives only code plus an empty directory skeleton:

| | in the repo | in a package |
|---|---|---|
| `data/` | 19 K (6 tracked scripts and inputs) | `data_20260822.tar` 48.6 G |
| `legacy/` (excluding hpm) | 7 M (source only) | `legacy_20260822.tar` 12.7 G |
| `legacy/hpm/` | 60 M (source + the 40 mp4 files in `fwv/`) | `legacy_hpm_{vis,outputs}_*.tar`, 914 M together |
| `results/` | 8.7 M (demo weights) | `results_20260822.tar` 194 M |
| `code/` | 296 M -- of which 290 M is `web-demo`'s cargo/npm build cache, regenerable | — |
| `FUNWAVE-TVD/` | 256 M, a third-party clone, not committed | — |
| `.git` | 478 M WARNING | — |

WARNING: those 478 M of `.git` are essentially all of the 306 mp4 files in `hpm/vis/` (436 M, in the
history since commit `d8c9d1c`). They are no longer in the working tree, but **the history objects
remain and a clone still pulls them** -- a real slim-down would mean rewriting history with
filter-repo.

**Checkpoint slimming (done 2026-08-13; the tool has since been deleted)**: early checkpoints
persisted the LBO spectral basis (one copy per head, per block) into the weight file, making a single
ckpt 6.58 GiB of which 6.6 GB was the duplicated basis. `code/strip_ckpt.py` / `strip_ckpt.sh` were
written at the time to strip them in bulk: **66 files, 6.58 GiB -> 8.6 MiB**, and
`legacy/hpm/outputs` went from 409 G to 396 M (releasing 407.9 GB).

Re-checked 2026-08-18: of the **37 checkpoints that exist (including everything old in
`legacy/hpm/outputs`), the legacy keys are all 0**, so the tool had served its purpose and was
deleted along with `strip_legacy_basis()` in `hpm_model.py` -- the resume path in `train.py` and the
loading path in `vis.py` were both changed to a plain `load_state_dict(..., strict=True)`, verified
to read the current checkpoints.
To get it back: `git checkout <the commit before the deletion> -- code/strip_ckpt.py`.

**The principle behind `.gitignore`**: output and data never enter version control (`*.pt`, `*.npy`,
`processed_data/`, `logs/`, `outputs/`, `wandb/`, `data/*`, `vis*/`, `FUNWAVE-TVD/`, `.env.local`).
That makes `.hydra/config.yaml` the only thing that can trace a historical configuration -- do not
delete it.

**Deliberate exceptions** (all re-admitted level by level in `.gitignore`, each with a comment giving the reason):

| exception | why |
|---|---|
| `data/fwv/TK94/{input.txt,gauges.txt}` + `make_cases.py` / `wk_check.py` | the only record of "what was actually run" (1.5) |
| `data/3d/crop_fields.{py,sh}` | "how the data was produced" |
| `results/web/model/**` (except `.hydra/hydra.yaml`) | the demo's default weights; without them a fresh clone has an empty dropdown |
| `code/web-demo/{web/dist,server/target/release/wave-demo}` | so a fresh clone can run `./start.sh` directly, without building |
| `legacy/hpm/fwv/{vis,hpm_fw_ss_R4}/` | 40 rendered mp4 files, 55 M (taken in on 2026-08-23) |
| `archive/restore.sh` + `archive/*.manifest` | the restorer and the per-file checksum lists; a clone has to have them |
| the `.gitkeep` of each placeholder directory | the directory skeleton, saying which package the contents are in |

---

## 8. Cross-comparison and overall conclusions

### 8.1 How the six lines evolved

```
POD-LSTM      dimensionality reduction + sequence model  -> reconstruction amplifies the error, judged a failure
   |  (drop the reduction; learn on the mesh directly)
Transolver++  point-cloud attention, single step         -> good single step, rollout diverges
FNO           spectral method on a regular grid, single step -> best single step, rollout diverges faster
   |  (introduce residual output + multi-step rollout training)
tsolverpp 3D  residual + R=4 BPTT                        -> the rollout no longer explodes but underfits, never finishes training
   |  (switch to an LBO spectral-basis backbone + a Delta=0 baseline)
pure HPM      residual added to its own previous frame   -> every channel is stably better than the baseline
   |  (replace the base with an external physical prior)
HPM+FUNWAVE   residual added to prior(t)                 -> 1000 frames of long rollout without diverging OK
```

### 8.2 Three phenomena that kept recurring

1. **Good single step, collapsing rollout** (the four earlier lines). The root cause is the mismatch between the training objective (single-step MSE) and the inference mode (autoregression). The HPM line addressed it head-on with "a residual base + multi-step BPTT + a val metric that is pure rollout".
2. **A constant difficulty ordering**: `alpha` < `p_rgh` < `Ux` < `Uz` (`Uy` and `nut` ineffective).
   `Uz` is small in magnitude (std 0.058 in 3D alphaU space vs 0.182 for `Ux`; 0.101 vs 0.267 in raw space) and fine in scale, giving it the worst signal-to-noise ratio.
3. **`nut` and `Uy` should both be turned off, but for completely different reasons**: `nut` has **poor low-rank structure** (POD says it takes 189 modes to reach 90% energy), whereas `Uy` is **a signal that is itself close to noise** (a quasi-2D case; the model's nRMSE of 1.000 ~= the baseline's 1.031, and even the prior calibration curve is flat).

### 8.3 The real methodological advance

It is not that the model was swapped several times, but that **the criterion changed**:

| period | val metric | the problem with it |
|---|---|---|
| POD-LSTM / Transolver++ / FNO | single-step MSE or coefficient-space error | disconnected from the deployment condition (autoregression), so it cannot show that the rollout will collapse |
| tsolverpp | weighted MSE, still the training loss | no "do not learn" control |
| **HPM** | **R-step pure rollout nRMSE (p=0 = the deployment condition) + a Delta=0 baseline per channel** | answers directly whether the model is better than doing nothing |

---

## 9. Hydra output conventions

How to install the environment and how to use the cluster are all in **[SETUP.md](SETUP.md)**. Only
one convention, needed when reading results, is kept here.

The output of `fno`, `transolver++` and `tsolverpp` is `outputs/<date>/<time>/`; that of `code` is
`results/train/<runname>/<override_dirname>/<timestamp>/`. Both carry `.hydra/config.yaml` (the
actual configuration), `.hydra/overrides.yaml` (the CLI overrides) and `checkpoints/`.
**To investigate any historical run, look at its `.hydra/overrides.yaml` first** (see 7.3 for details).
