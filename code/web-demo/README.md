# wave-demo

Interactive demo of the wave-field surrogate model. **Runs on a login node of the OSU
cluster** — click once, the whole pipeline runs on the cluster under a locked default
configuration, and the rendered videos come back to the page.

Location: `models/code/web-demo/`

## First-time setup

`results/web/` is the runtime state directory, and **none of it is in the repository
except the demo's default weights**. Those weights
(`results/web/model/hpm_fw_aU_h128/2026-08-12_15-31-45/`, `best.pt` 8.6 M plus two hydra
yamls) travel with the repo — clone it and the dropdown already has something to pick.
The empty `priors/` and `vis/` directories are in the repo too, each holding a
`.gitkeep`, so **there is nothing to create**.

The backend binary (`server/target/release/wave-demo`) and the frontend bundle
(`web/dist/`) are **both committed, so there is nothing to build** — clone and run
`./start.sh`. Only a change under `server/src/` requires a rebuild, and the new binary
must be committed along with it:

```bash
module load rust/1.92
cd code/web-demo/server && cargo build --release
```

**History starts empty.** When `submissions.json` does not exist the backend treats it as
an empty list (`load()` in `pipeline.rs`); the first record appears only after you click
once, and the intermediate fields and videos are generated at that point too. If you want
to see what history looks like, run it yourself — no samples are shipped.

## Running it

```bash
cd <models>/code/web-demo
./start.sh      # starts it; prints pid / node / socket / log / forwarding command
./stop.sh       # stops it and cleans up the socket and pid file
```

`start.sh` refuses to start a second instance (two processes would fight over the same
socket and the same `submissions.json`); if the process is on **a different login node**,
`stop.sh` tells you which one to go to — `submit` round-robins across submit-a/b/c, and
this trap is easy to fall into.

To view it from your own machine (a socket is not a port, so the forwarding syntax
differs, and you **must target the node the service is on**):

```bash
# start.sh prints the full command — just copy it. The hostname must be the node it
# reported (submit is round-robin), and it must be the external domain
# <short-name>.hpc.engr.oregonstate.edu — internal short names do not resolve outside.
ssh -L 8788:<models>/results/web/wave-demo.sock $USER@<node>.hpc.engr.oregonstate.edu
# then open http://localhost:8788 locally
```

Debugging on the cluster cannot use `curl localhost:8788` (that is TCP; the service
listens on a socket). Use
`curl --unix-socket <models>/results/web/wave-demo.sock http://x/api/health`.

### ⚠️ Build the frontend on a compute node, not the login node

The login node's `ulimit -v` is 15,000,000 KB (about 14.3 GiB), and the **soft and hard
limits are identical, so you cannot raise it yourself**. `npm run build` dies on its very
first step, `vue-tsc -b`, with
`WebAssembly.instantiate(): Out of memory: Cannot allocate Wasm memory`.

A compute node has no such limit, and `module load nodejs/v23.9` gives you a current Node
there. Do not rely on the bare `/usr/bin/node` on a compute node — that one is v10.24.0
(the login node's v22.23.1 comes from a different system image); the module is the point.

```bash
# frontend first — the backend bakes web/dist/ in at compile time (rust_embed)
srun -p share -n1 -c4 --mem=8G -t 00:20:00 bash -lc \
  'module load nodejs/v23.9 && cd code/web-demo/web && npm ci && npm run build'
# then the backend
srun -p share -n1 -c4 --mem=8G -t 00:30:00 bash -lc \
  'module load rust/1.92 && cd code/web-demo/server && cargo build --release'
```

Verified on cn-e02, 2026-08-23: `vite build` finishes in 1.4 s and reproduces the
committed bundle byte for byte (`index-Cx6eP9a1.css` 18587 B, `index-Dw6tlNiT.js`
91055 B). Both artifacts are committed, so remember to `git add` them afterwards.

You do not have to go through this every time you touch the frontend: run `npm run dev`
locally (5174) together with the port forward above, and `/api` hits the backend on the
cluster directly, with hot reload working as usual.

## All state lives in one directory: results/web/

```
results/web/
├── model/hpm_fw_aU_h128/2026-08-12_15-31-45/   ← weights, **do not delete**
│   ├── checkpoints/best.pt
│   └── .hydra/config.yaml
├── priors/            ← intermediate fields, ~12 GB per case   deleted ⇒ recomputed (~6 min)
├── vis/               ← mp4, ~26 MB per case                   deleted ⇒ recomputed (~1h38m)
├── submissions.json   ← submission log (job id / state / elapsed)
├── wave-demo.log      ← backend log, appended
├── wave-demo.pid      ← "<pid> <short hostname>"
└── wave-demo.sock     ← Unix socket, 0600, removed on exit
```

**Everything this demo reads and writes stays inside `<models>`**; nothing is scattered
elsewhere. The one exception is the SLURM logs of the jobs it submits — `vis_adp.sh`
hard-codes `#SBATCH --output` to `code/logs/visadp_%x_%j.log`, the same directory used
when running ADP by hand.

**Delete what is under `priors/` and `vis/` and the next submission recomputes from
scratch** — that is how you demonstrate the pipeline really runs, without touching the
experiment outputs under `results/train` / `results/vis`.

`wave-demo.log` records **what the backend did** (what sbatch returned on submit, why
reconciliation with the cluster failed); what happened inside a job is under `code/logs/`,
findable by job id. Do not confuse the two.

The hostname in the pid file is necessary — with three login nodes, a pid alone does not
tell you where to go to stop it.

## Sharing toggle

This runs on a **shared login node** with a dozen other users on it at the same time. The
key point: `127.0.0.1` means "this machine", not "this user" — **a TCP port has no owner
and no chmod**, so any UID on the same host can connect. "Bound to loopback = private" is
simply false.

Hence two separate channels:

| | transport | who can connect |
|---|---|---|
| you | Unix socket, `0600` | **only your UID** — the kernel enforces file permissions; everyone else gets `EACCES` |
| visitors | TCP `0.0.0.0:8788`, controlled by the toggle in the top-right of the UI | anyone who can reach it over the network while it is on; turning it off makes the listener disappear |

When it is off there is **no line for 8788 in `ss -ltn`**, and `curl` from the same host
gets connection refused — it genuinely does not exist, this is not "the port is hard to
guess".

The visitor channel is **read-only**: `POST /api/submit` and `POST /api/share` both return
403. Otherwise anyone with the link could submit SLURM jobs as you and burn your quota.
Visitors see `visitor: true` from `/api/share`, and the frontend hides the toggle entirely
based on that.

`WAVE_SHARE_ADDR` changes the bind address. Whether it is reachable from off campus
depends on campus network policy; only same-host reachability has been verified.

## Idle auto-stop

The default is **auto-stop after 60 idle minutes** (`WAVE_IDLE_MIN=0` disables it). This
is a process on a shared login node; it should not be left running unattended. If sharing
is on, that is turned off with it.

**Only human actions count; background polling does not.** While a job is running the
frontend polls `/api/submissions` every 15 seconds — that is a machine acting. Counting it
would let a single forgotten tab keep the service alive forever, making auto-stop
meaningless. So:

| | counts as activity |
|---|---|
| opening / reloading the page, submitting, opening the detail modal, flipping the share toggle | ✅ |
| a real click or keypress → the frontend posts `POST /api/active` (throttled to once a minute) | ✅ |
| background polling of `/api/submissions` (carries the `x-wave-poll: 1` header) | ❌ |

The same applies to visitors — someone watching counts as someone using it.

It also stops while a job is still running and nobody is watching. That is fine: the
start-up reconciliation on the next `./start.sh` picks the state back up.

## What it does

**This version only runs the locked default configuration; the frontend has no input
control at all.** Click "compute with the default configuration" once, the backend runs
the pipeline on the cluster, the submission log on the right shows which stage it is on,
and when it finishes you click "details" to see the results.

The default configuration is the training case TK94 (Ting & Kirby 1994, spilling
breaker):

```
wave height H  12.7 cm (AMP_WK 0.0635 m)     slope 1:35 · T = 2 s · d = 40 cm
checkpoint     hpm_fw_aU_h128 / 2026-08-12_15-31-45   best_val 0.1403 @ ep19
interval       chunk 10 · t 50–100 s · 1000 frames
channels       alpha · Ux · Uz · p_rgh       (window=0 / feedback=self / R=4 / αU weighted)
```

### Pipeline

One submission = **two** SLURM jobs chained with `afterok`, submitted atomically in a
single shell invocation (submitting them separately would leave an orphaned prior job
behind whenever the second submission fails):

| stage | how | partition | measured |
|---|---|---|---|
| ~~FUNWAVE~~ | **not run** — TK94's output is already on the cluster | — | — |
| gen_prior | `vis_adp.sh STAGE=prior` | `eecs` CPU | **5–6 min** |
| HPM rollout + render | `vis_adp.sh STAGE=vis` | `preempt,ampere` GPU | **1 h 38 min** |

The second stage is "HPM inference + plotting": `vis.py lt` loads the checkpoint, finishes
a 1000-frame rollout and renders. It is **not two jobs**.

**Not one line of the computation lives in this repository** — it all reuses `vis_adp.sh`
on the cluster, and **writes no new files into `models/`** (other than vis_adp.sh's own
outputs).

### Outputs kept separate from experiment data (done)

The app sits at `models/code/web-demo/`, next door to `vis_adp.sh`. All outputs go into
**one directory, `results/web/`** (layout under "All state lives in one directory"
above), leaving experiment outputs alone. This works because `vis_adp.sh` writes
`OUTROOT` / `PRIORROOT` / `VISROOT` / `LIFTROOT` / `CKDIR` as `${VAR:-default}`, and
`pipeline.rs` points the first few at `results/web/` when it submits; pass nothing and
`vis_adp.sh` behaves exactly as before.

**Not done yet: unlocking the frontend configuration.** The controls are already drawn;
see below.

### Checkpoints are discovered by scanning

The backend **scans `results/web/model/` on every start**, recognizing the layout
`<run_name>/<timestamp>/checkpoints/*.pt` (the same as `results/train/`, so copying a
training output directory over wholesale is enough to be picked up), and serves the result
to the frontend through `/api/config` to fill the dropdown. Copy another set of weights
under `model/`, restart once, and it shows up as an option — the frontend hard-codes no
weight name.

If nothing is found the start-up log warns and the dropdown shows "(no .pt found under
model/)".

### Unlocking the parameters later

The frontend controls are **already really drawn** (slider, radio group, two dropdowns,
channel chips); each one just carries a `lockedTry()`: change it and the DOM value snaps
back to the default with a warning explaining why that item is locked for now. To unlock,
delete the `lockedTry` call and wire up a real v-model — the layout needs no change at all.

On the backend side, change the constants in `pipeline.rs` and `/api/config`. Unlocking
**arbitrary wave heights** needs three more things: composing the case directory name from
the parameters, copying TK94's `input.txt` and changing only the `AMP_WK` / `SLP` lines,
and adding a FUNWAVE job on the `share` partition (measured 4 h 03 min single-core, job
20902929). Note that the numbers written into `input.txt` must be byte-identical to those
of the existing case (the two formats `{:.4}` / `{:.9}`), otherwise the old and new cases
look the same while their parameters differ.

## API

| path | description |
|---|---|
| `GET /api/config` | the locked configuration plus wave-height bounds; the frontend draws the slider from it |
| `POST /api/submit` | submit one computation under the default configuration; **takes no body**. 502 = submission to the cluster failed |
| `GET /api/submissions` | the submission log, **refreshing the state of still-live jobs along the way**. With the `x-wave-poll: 1` header it does not count as "someone is using it" |
| `GET/POST /api/share` | read/change the visitor channel toggle. Visitors read `visitor: true` and POST is always 403 |
| `POST /api/active` | the frontend reporting "a human did something", for idle auto-stop. Throttled to once a minute |
| `GET /api/video/{stage}/{case}/{chunk}/{field}` | mp4, supports Range |
| `GET /api/health` | which node it is on, the cluster paths, the cache location |

## Every path is relative

No absolute path is hard-coded anywhere. Once the whole tree is committed with `models/`
as its root, it runs wherever you clone it:

- the **binary** walks 5 levels up from its own location to find `<models>`
  (`<models>/code/web-demo/server/target/release/wave-demo`)
- **`start.sh` / `stop.sh`** take `../..` from the script's own location and export
  `WAVE_ROOT` explicitly
- every other path (`code/vis_adp.sh`, `results/web/…`, `data/fwv/…`) is composed from
  that root

Verified: run the binary directly after `cd /` and it still resolves `models = <repo>`.

`WAVE_ROOT` can override this, to point the state somewhere else (during testing, say).

## Environment variables

| variable | default |
|---|---|
| `WAVE_IDLE_MIN` | `60` — auto-stop after this many idle minutes; `0` = never |
| `WAVE_ROOT` | `<models>`, derived from the executable's location (`start.sh` passes it explicitly) |
| `WAVE_SOCK` | `<WAVE_ROOT>/results/web/wave-demo.sock` |
| `WAVE_PUBLIC_HOST` | `<short hostname>.hpc.engr.oregonstate.edu`, used for the forwarding command and share links |
| `WAVE_PUBLIC_DOMAIN` | `hpc.engr.oregonstate.edu` (replaces only the domain suffix) |
| `WAVE_SHARE_ADDR` | `0.0.0.0:8788` (only used when sharing is on) |
| `WAVE_STORE` | `<WAVE_ROOT>/results/web/submissions.json` |
| `WAVE_LOG` | `<WAVE_ROOT>/results/web/wave-demo.log` |
| `WAVE_PID` | `<WAVE_ROOT>/results/web/wave-demo.pid` |
| `RUST_LOG` | `wave_demo=info` |

`WAVE_ROOT` is resolved to an **absolute path**: shell scripts and `fs::read` use the same
string, and a leftover `~` would be expanded by the former but not the latter.

## Design decisions

**No polling.** Job state only needs looking at while a job is running. So: fetch the
submission log once when the page opens; while your own jobs are running, query **those
job ids** every 15 seconds; stop once they are all done. Left sitting there untouched, the
demo page is completely silent toward the cluster.

**Both prior and vis get `FORCE=1`.** When a job is preempted and requeued on `preempt`,
`vis_adp.sh` would see a half-written mp4, treat it as "already there" and skip it —
shipping a truncated video. Forcing a recompute is the only clean answer, and it costs
only prior's 6 minutes.

**The submission log is persisted.** `submissions.json`, written to a temporary file and
renamed. It survives page reloads and service restarts, and keeps the most recent 50
entries.

**Anything that can end up in a remote shell has its character set nailed down before it
gets there.** Case names allow only `[A-Za-z0-9_]`, channels go through a whitelist,
chunk is restricted to 0–10, job ids must be digits. This is not a matter of getting the
quoting right — one botched quote is arbitrary command execution. The validators are
`valid_case_name` / `valid_field` / `valid_job_id` in `wave.rs`.

⚠️ They currently have **no test coverage**. An earlier version of this file claimed a
test `wave::tests::校验挡住注入` guarded them; no such test exists (as of 2026-08-23 the
crate's 18 tests cover path resolution, job-state merging, Range parsing and checkpoint
scanning). Worth adding.

**Fetching mp4 does not go through `sh::run`.** That function decodes as UTF-8 lossy,
which chews through binary with replacement characters. Video goes through
`sh::read_file`, which takes the bytes off local disk as they are.

⚠️ An earlier version of this file described an `ssh::fetch` (`ssh host cat --`) writing a
`.part` file and renaming it. None of that exists: the service runs on the login node
itself, so the data is local and there is no download — `main.rs` opens with "no ssh
anywhere". Some comments still say "ssh" where they mean one local `bash -ls`.

**Range is mandatory.** Videos at the surrogate-model resolution are 4–8 MB; without 206
the progress bar cannot be dragged.

**A submission always renders all four channels.** One submission means running the
checkpoint through this set of conditions completely; there is no reason to make the user
pick channels first.

## Known gotchas

**Videos occasionally fail to decode.** What gets rendered is 3840×1080 H.264, and the
browser's decoder can **neither produce frames nor report an error** (`readyState` stuck
at 0, `error` is null). Seen once in practice: the same file played fine minutes earlier,
then the entire Chrome instance could not decode it — not through its own built-in player,
not even a 1920×540 downscaled version; only restarting the browser fixed it. **This is
not a server-side problem** (`fetch()` on the same URL returns 206 in 6 ms, and feeding
the bytes to the video element as a blob fails just the same).

The frontend has a 12-second watchdog for this: on timeout it overlays an explanation plus
an "open the mp4 in a new tab" link, instead of spinning forever. A real fix means
lowering the render resolution (`row_h` on the `vis.py` side), but that would require
re-rendering every output, which is out of scope for this prototype.

**Every run rewrites a 12 GB prior.** `FORCE=1` means prior is recomputed and rewritten
every time, overwriting `results/web/priors/TK94/` (1000 frames of chunk 10, 11.5 G).
`/nfs/hpc/share` still has 1.3 T, so this is not a near-term problem, but nothing cleans
up automatically.

**Preemption and requeue restart the whole stage.** The third step goes to
`preempt,ampere`; when it lands on preempt and gets preempted, `--requeue` reschedules it
and those 1 h 38 min start over (because `FORCE=1` was passed). If you need the result in
a hurry, change that line in `pipeline.rs` to submit to `ampere` only.
