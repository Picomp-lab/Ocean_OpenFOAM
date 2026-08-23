#!/bin/bash
# ============================================================
# setup.sh — clone 之后跑这一个脚本，把环境装齐 / 检出缺什么
#
#   ./setup.sh              # 一条命令跑完：缺什么补什么
#   ./setup.sh --check      # 只检测不装，退出码非 0 表示有缺的
#
# 一趟走完五步: 平台 / conda 环境 / python 包 / wandb / 目录与数据。
# 环境和 FUNWAVE-TVD 缺了自动装，大文件包放进 archive/ 会自动解压（这一步整个转交
# archive/restore.sh，清单解析和解压逻辑只有那一份）。
# **archives.tsv 里的包一个都不能少** —— 内容没就位、archive/ 下又没有包，就算缺件，
# 脚本以非 0 退出（照样把这一趟能查的全查完再退，不会半路截断）。
#
# 环境变量（都有默认值，正常不用管）:
#   OCEAN_ENV=/path/to/env   conda 环境位置（默认见下面 "第 2 步"）
#   OCEAN_ARCHIVE=/path      大文件包(.tar)的存放目录（默认 <仓库根>/archive）
#   SRUN_WAIT=90             等计算节点的秒数；设 0 就全部留在登录节点跑
#
# 探测和 pip 安装默认走 srun 丢到 share 分区的计算节点（排队通常几十秒，不用 preempt
# —— 那个会被抢占 requeue）—— 登录节点每用户 RLIMIT_NPROC=400 且全节点共享，
# import numpy 时 OpenBLAS 起不了线程就会失败，好包会被误判成坏的。
#
# 依赖清单在 requirements.txt —— 加包改那个文件，
# 这里不写死任何包名。**只能在集群上跑**，别的机器直接报错退出。
#
# 这个脚本只做三件事: 检测、安装、提醒。不动任何数据，不投作业。
# ============================================================

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 参数 ----
# 只有一个开关。其余一律自动: 缺什么补什么，装不了的就提醒。
DO_INSTALL=1
for a in "$@"; do
  case "$a" in
    --check)     DO_INSTALL=0 ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数: $a（只支持 --check；--help 看用法）"; exit 2 ;;
  esac
done

# ---- 输出 ----
if [ -t 1 ]; then R=$'\e[31m' G=$'\e[32m' Y=$'\e[33m' B=$'\e[1m' N=$'\e[0m'
else R= G= Y= B= N=; fi
MISSING=0 WARNED=0
ok()   { printf '  %s✔%s %s\n' "$G" "$N" "$*"; }
bad()  { printf '  %s✘%s %s\n' "$R" "$N" "$*"; MISSING=$((MISSING+1)); }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; WARNED=$((WARNED+1)); }
info() { printf '    %s\n' "$*"; }
# py_run 是在 $( ) 里被调用的，普通 warn 会被当成探测结果吞掉 —— 这条走 stderr
warn_raw() { printf '  %s!%s %s\n' "$Y" "$N" "$*" >&2; }
step() { printf '\n%s== %s ==%s\n' "$B" "$*" "$N"; }

# ============================================================
# 1. 在哪台机器上
# ============================================================
step "1/5 平台"
ON_CLUSTER=0
CONDA_SH_CLUSTER=/nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
[ -f "$CONDA_SH_CLUSTER" ] && [ -d /nfs/hpc/share ] && ON_CLUSTER=1

if [ "$ON_CLUSTER" = 1 ]; then
  ok "OSU 工程学院集群（$(hostname -s)）"
  case "$(hostname -s)" in
    submit*) info "登录节点。重活一律 sbatch —— 这里有 ulimit -v ≈15 GB 的地址空间限制" ;;
  esac
  command -v sbatch >/dev/null 2>&1 \
    && ok "SLURM 可用（$(sbatch --version 2>/dev/null)）" \
    || warn "PATH 里没有 sbatch —— 非交互式 ssh 会这样，命令外面包一层 bash -lc"
else
  printf '  %s✘%s 不在 OSU 集群上（%s %s）\n' "$R" "$N" "$(uname -s)" "$(hostname -s)"
  cat <<'ERR'

    这套东西只在 OSU 工程学院集群上跑得起来:
      · 数据 152 GB 在 /nfs/hpc/share 下，本机没有也拷不动
      · 训练 / 渲染脚本是 SLURM 的（sbatch、分区、--gres）
      · torch 要 cu130，配的是集群那批 H100 / H200 / A40

    先 ssh 上去再跑:
      ssh <user>@submit.hpc.engr.oregonstate.edu
      cd ~/hpc-share/models && ./setup.sh

    只想改前端（code/web-demo/web/）的话不用这个脚本，
    直接 cd code/web-demo/web && npm install && npm run dev。
ERR
  exit 1
fi

# ============================================================
# 2. conda 环境
# ============================================================
step "2/5 conda 环境"

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
  bad "找不到 conda"
  info "集群上它在 $CONDA_SH_CLUSTER"
  info "别处装个 miniforge，或者 CONDA_SH_OVERRIDE=/path/to/conda.sh ./setup.sh"
  exit 1
fi
ok "conda: $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"

# 环境位置: 集群上放 /nfs/hpc/share（home 有配额，装不下 torch），别处走默认 envs 目录
if [ -n "${OCEAN_ENV:-}" ]; then ENV_PREFIX="$OCEAN_ENV"
elif [ "$ON_CLUSTER" = 1 ];  then ENV_PREFIX="/nfs/hpc/share/$USER/.conda/envs/ocean"
else                              ENV_PREFIX="$(conda info --base)/envs/ocean"
fi

if [ -x "$ENV_PREFIX/bin/python" ]; then
  ok "已有环境: $ENV_PREFIX"
elif [ "$DO_INSTALL" = 0 ]; then
  bad "环境不存在: $ENV_PREFIX（去掉 --check 会自动建）"
else
  echo "  建环境（python 3.10 + ffmpeg，几分钟）: $ENV_PREFIX"
  mkdir -p "$(dirname "$ENV_PREFIX")"
  conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.10 ffmpeg || {
    bad "conda create 失败"; exit 1; }
  ok "环境建好了"
fi

if [ -x "$ENV_PREFIX/bin/python" ]; then
  conda activate "$ENV_PREFIX" || { bad "activate 失败: $ENV_PREFIX"; exit 1; }
  ok "python $("$ENV_PREFIX/bin/python" -V 2>&1 | awk '{print $2}')"
  PY="$ENV_PREFIX/bin/python"

  # 把环境位置写进 .env.local（不进版本库）——  activate.sh 会读它，
  # 于是各 sbatch 脚本不用写死任何人的路径。
  if [ ! -f "$REPO/activate.sh" ]; then
    bad "缺 activate.sh —— 版本库里应该有，各 sbatch 脚本靠它找环境"
  elif [ "$DO_INSTALL" = 1 ]; then
    printf 'OCEAN_ENV=%s\n' "$ENV_PREFIX" > "$REPO/.env.local"
    ok ".env.local 已写（activate.sh 读它定位环境）"
  elif [ -f "$REPO/.env.local" ]; then
    ok ".env.local: $(sed -n 's/^OCEAN_ENV=//p' "$REPO/.env.local")"
  else
    warn "没有 .env.local —— 各 sbatch 脚本会回落到 /nfs/hpc/share/\$USER/.conda/envs/ocean"
    info "去掉 --check 跑一次就会生成"
  fi
else
  PY=""
fi

# ---- 重活丢到计算节点 ----
# 登录节点每用户 RLIMIT_NPROC=400 且全用户共享，import 大包时 OpenBLAS 起不了线程，
# 好好的包会被误判成坏的。计算节点上 nproc 二十多万、地址空间不限、也能出网装包。
# 只投 share，**不碰 preempt** —— preempt 会被抢占 requeue，一个本该几十秒的检测
# 变成半路被踢掉再重排，徒增麻烦。share 排不上（QOSGrpCpuLimit 是全组配额，占满
# 时会 PENDING）就等 SRUN_WAIT 秒后退回登录节点。SRUN_WAIT=0 可强制留在登录节点。
USE_SRUN=0
if [ "${SRUN_WAIT:-90}" -gt 0 ] && command -v srun >/dev/null 2>&1; then USE_SRUN=1; fi
SRUN_BASE=(-p share -n1 -c2 -J hpc_setup)
SRUN_WAIT=${SRUN_WAIT:-90}           # 等不到就退回登录节点（本地那条路已经能自己
                                     # 判断"没测准"，不会误报），别把人吊在这儿

# 退回登录节点时用的单线程环境
NOTHREAD=(env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
          NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1)

# 在计算节点上跑 python；stdin 是脚本，$@ 是脚本参数。跑不成就退回登录节点。
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
      warn_raw "share 分区 ${SRUN_WAIT}s 没排上（多半撞了 QOSGrpCpuLimit），退回登录节点（限单线程）"
    else
      warn_raw "计算节点上没跑成（srun rc=$rc），退回登录节点（限单线程）"
    fi
  fi
  printf '%s' "$script" | "${NOTHREAD[@]}" "$PY" - "$@" 2>/dev/null
}

# 同上，跑 pip。输出直接往外走，不捕获。
pip_run() {
  local args a
  args=""
  for a in "$@"; do args="$args $(printf '%q' "$a")"; done
  if [ "$USE_SRUN" = 1 ]; then
    srun "${SRUN_BASE[@]}" --mem=8G -t 00:40:00 \
      bash -lc "source '$CONDA_SH' && conda activate '$ENV_PREFIX' && exec python -m pip$args" && return 0
    warn "计算节点上装不成，退回登录节点再试一次"
  fi
  "$PY" -m pip "$@"
}

# ============================================================
# 3. python 包
# ============================================================
step "3/5 python 包"

# 包名和版本全在 requirements.txt 里，脚本本身不写死任何包 —— 要加依赖改那个文件。
# torch 系也在里面，靠文件里的 --extra-index-url 指向 pytorch 的 cu130 源。
REQ="$REPO/requirements.txt"
[ -f "$REQ" ] || { bad "缺 ${REQ#"$REPO"/} —— 版本库里应该有这个文件"; exit 1; }

if [ -z "$PY" ]; then
  bad "没有可用的 python，跳过包检测"
else
  [ "$USE_SRUN" = 1 ] && echo "  探测丢到计算节点（share 分区）—— 排队通常几十秒，等不到就退回本地"

  # 一次 srun / 一个 python 进程把 requirements 全探完，顺带报 torch 的
  # cuda 版本和 torch/torchvision 构建是否配套。输出:
  #   OK|导入名|版本            能 import
  #   MISSING|导入名|pip 行     没装
  #   BROKEN|导入名|说明|pip 名 装了但 import 抛错
  #   INFO|文字                 直接打印
  #   TVDIFF|文字               torch 与 torchvision 构建不一致
  probe_pkgs() {
  need_pip=0
  probe=$(py_run "$REQ" <<'PYEOF'
import importlib, importlib.metadata as md, re, sys

# numpy 第一个 import，用来先把机器资源探明白：登录节点 RLIMIT_NPROC 只有 400
# 且全用户共享，挤满时 numpy 的 OpenBLAS 开不出线程，C 扩展直接起不来。先在这里
# 撞上，后面所有 import 失败就都能标成"没测准"而不是"包坏了"。
import socket
print("WHERE|%s" % socket.gethostname())
try:
    import numpy  # noqa: F401
except Exception as e:
    # numpy 自己都起不来 —— 登录节点资源被挤爆时会这样。后面所有 import 失败
    # 都不作数，bash 那边看到这行会把结果标成"没测准"。
    print("NUMPYFAIL|%s" % str(e).replace("|", "/")[:80])

# pip 名 != 导入名的几个
ALIAS = {"pillow": "PIL", "pyyaml": "yaml", "hydra-core": "hydra",
         "polars-lts-cpu": "polars", "imageio-ffmpeg": "imageio_ffmpeg"}

for path in sys.argv[1:]:
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):      # 空行 / 注释 / --index-url
            continue
        dist = re.split(r"[=<>!~;\[ ]", line)[0].strip()
        imp = ALIAS.get(dist.lower(), dist.lower().replace("-", "_"))
        try:
            m = importlib.import_module(imp)
            ver = getattr(m, "__version__", None)
            if ver is None:                       # xxhash 这类没有 __version__
                try:
                    ver = md.version(dist)
                except md.PackageNotFoundError:
                    ver = "?"
            print("OK|%s|%s||%s" % (imp, ver, path))
        except Exception as e:
            msg = str(e).replace("|", "/")[:90]
            try:
                v = md.version(dist)
                print("BROKEN|%s|%s 已装(%s) 但 import 抛 %s: %s|%s|%s"
                      % (imp, dist, v, type(e).__name__, msg, dist, path))
            except md.PackageNotFoundError:
                print("MISSING|%s|%s||%s" % (imp, line, path))

# torch 的 cuda / GPU 情况；顺带看 torch 与 torchvision 是不是同一批构建
# (+cu130 这个本地版本号只在模块的 __version__ 里，dist 元数据被剥掉了)
try:
    import torch
    print("INFO|torch %s / cuda %s / GPU 可见: %s"
          % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
    try:
        import torchvision
        a = torch.__version__.partition("+")[2] or "无"
        b = torchvision.__version__.partition("+")[2] or "无"
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
                "$(hostname -s)"*) [ "$USE_SRUN" = 1 ] && info "（这次是在登录节点跑的）" ;;
                *) info "（在 $ran_on 上跑的）" ;;
              esac ;;
      NUMPYFAIL)
              untrusted=1
              warn "这台机器上 numpy 都起不来，本轮包检测不作数（不代表包有问题）"
              info "$imp"
              info "登录节点资源被挤爆时就是这样。隔一会儿重跑，或让它走 srun（别加 --local）。" ;;
      OK)     ok "$imp $detail" ;;
      INFO)   info "$imp" ;;
      TVDIFF) warn "torch / torchvision 不是同一批构建（$imp）"
              info "torchvision 多半是从 PyPI 装的（cu12 构建）。要统一:"
              info "  $PY -m pip install --force-reinstall --no-deps torch torchvision \\"
              info "       --extra-index-url https://download.pytorch.org/whl/cu130" ;;
      BROKEN)
              # 资源型报错 != 包坏了。登录节点挤爆时 numpy 的 C 扩展起不来，
              # 连锁报出来的是 PyCapsule_Import 这些。
              case "$detail$untrusted" in
                *PyCapsule_Import*|*"Resource temporarily unavailable"*|\
                *"CPU dispatcher tracer"*|*1)
                  warn "$imp 没测准（$(printf '%s' "$detail" | cut -c1-60)…）"
                  if [ "$hint_shown" = 0 ]; then
                    hint_shown=1
                    info "这类报错是机器资源不够导致的，不是包坏了。重跑一次，别加 --local，"
                    info "让它去计算节点上测。（下面同类的就不再重复这段了）"
                  fi ;;
                *)
                  bad "$detail"
                  info "包在但 import 挂了。先卸干净再重装:"
                  info "  $PY -m pip uninstall -y $dist && ./setup.sh" ;;
              esac ;;
      MISSING)
              bad "$imp 缺失 → $detail"
              need_pip=1 ;;
    esac
  done <<< "$probe"
  }

  pre_missing=$MISSING
  probe_pkgs

  if [ "$DO_INSTALL" = 1 ] && [ "$need_pip" = 1 ]; then
    # 一次装完。torch 系靠文件里的 --extra-index-url 走 pytorch 的 cu130 源，
    # 其余走 PyPI。计算节点能出网（实测 pypi / npm registry 都通），所以丢过去装。
    echo "  pip install -r ${REQ#"$REPO"/}"
    pip_run install -r "$REQ" && ok "依赖装好了" || bad "pip 装失败，看上面的报错"
    # 重新探一遍: 装之前那一轮的计数已经不作数了
    printf '  %s-- 装完重新检测 --%s\n' "$B" "$N"
    MISSING=$pre_missing
    probe_pkgs
  fi

  # 混装体检：torch 和 numpy 的 C 扩展能不能一起加载。requirements.txt 钉的是
  # PyPI 版 numpy，正常情况下这句必过；挂了通常意味着有人往环境里 conda 装了
  # 编译过的包，把 libstdc++ 的依赖搅乱了。
  if "${NOTHREAD[@]}" "$PY" -c "import torch, numpy.fft, torchvision" >/dev/null 2>&1; then
    ok "torch + numpy.fft + torchvision 能一起用"
  else
    err=$("${NOTHREAD[@]}" "$PY" -c "import torch, numpy.fft, torchvision" 2>&1 | tail -1)
    bad "torch 和 numpy 的扩展打架了"
    info "$(printf '%s' "$err" | cut -c1-110)"
    info "把 numpy 换回 requirements.txt 指定的 PyPI 版:"
    info "  conda remove --force -p $ENV_PREFIX numpy && $PY -m pip install -r ${REQ#"$REPO"/}"
  fi

  # ffmpeg: vis.py 用 matplotlib 的 ffmpeg writer 出 mp4，要 PATH 上有真的 ffmpeg
  if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
  else
    bad "PATH 里没有 ffmpeg —— vis.py 存 mp4 会失败"
    [ "$DO_INSTALL" = 1 ] && conda install -y -p "$ENV_PREFIX" -c conda-forge ffmpeg \
      && { ok "ffmpeg 装好了"; MISSING=$((MISSING-1)); }
  fi
fi

# ============================================================
# 4. wandb
# ============================================================
step "4/5 wandb"
# train.py 默认开着 wandb（config.yaml: wandb.enabled=true, project=hpm-wave），
# 但 train.py 的 wandb_ready() 会先判断「能不能用」—— 没装 / 没登录 / init 失败都只在
# log 里写一行，训练照常跑。所以这里没登录只是提醒，不算缺件。
if [ -n "${WANDB_API_KEY:-}" ]; then
  ok "WANDB_API_KEY 已设"
elif grep -qs 'api\.wandb\.ai' "$HOME/.netrc"; then
  ok "已登录（~/.netrc 里有 api.wandb.ai）"
else
  warn "wandb 没登录 —— 本次不记录，训练照常（train.py 的 wandb_ready 会跳过）"
  info "要记录的话三选一:"
  info "  wandb login                    # 在登录节点上做，计算节点不一定通外网"
  info "  export WANDB_MODE=offline      # 只落本地 wandb/ 目录，事后 wandb sync"
  info "  sbatch run.sh wandb.enabled=false"
fi
[ "${WANDB_MODE:-}" = offline ] && info "当前 WANDB_MODE=offline"

# ============================================================
# 5. 目录与数据
# ============================================================
step "5/5 目录与数据"
# 只建 code/logs —— results/ 及其子目录带 .gitkeep 进了版本库，clone 就有，不用管。
# logs/ 非建不可: #SBATCH --output 在脚本执行**之前**就生效，目录不在时 SLURM 把日志
# 整个丢掉而作业照样 COMPLETED（实测，见 code/vis.sh 抬头）。脚本内那句 mkdir -p logs
# 只是兜底，救不了第一次提交。--check 只看不建。
if [ -d "$REPO/code/logs" ]; then ok "code/logs/"
elif [ "$DO_INSTALL" = 1 ]; then mkdir -p "$REPO/code/logs" && ok "code/logs/（已建）"
else warn "缺 code/logs/（去掉 --check 会自动建；不建的话第一次 sbatch 的日志会丢）"; fi

# 训练数据本体（config.yaml: data.dir = data/3d/cropped_0.05）不在这里查 —— 它就是
# archives.tsv 里 data_*.tar 的探测路径，下面那个循环连 md5 和解压一起管了。这里只管
# 循环不管的 prior_ktuned/（config.yaml: data.prior_dir），它是 gen_prior.sh 现算的，
# 不打包。
PRIOR="$REPO/data/3d/cropped_0.05/prior_ktuned"
if [ -d "$PRIOR" ] && [ -n "$(ls -A "$PRIOR" 2>/dev/null)" ]; then
  ok "prior_ktuned/"
else
  warn "缺 prior_ktuned/ —— 数据到位后 cd code && sbatch gen_prior.sh 生成"
  info "整份 data/ 都想自己重造的话（不从云端拿包）:"
  info "  data/3d/crop_fields.sh   从 OpenFOAM 算例（\$OCEAN_CASE）的体场裁一份出来"
  info "  code/gen_prior.sh        再生成 fwv 线要的 prior_ktuned/"
fi

# ---- 大文件（历史可视化 / checkpoint / 训练数据）----
# 这一段整个交给 archive/restore.sh —— 清单解析、md5、解压、逐文件核 manifest 只有
# 那一份实现，这里不再抄一遍（抄的那份漏了 --skip-old-files，会把版本库里跟包重叠的
# 文件静默盖掉）。它的规矩是**全有才动手**: 少一个包就列全问题 exit 1，一个字节都不解。
# 幂等: 探测路径非空的包它自己会跳过，所以重跑 setup.sh 不会重解。
RESTORE="$REPO/archive/restore.sh"
if [ ! -x "$RESTORE" ]; then
  bad "缺 archive/restore.sh（或没有执行位）—— 版本库里应该有，大文件靠它还原"
else
  # --check 传下去: 只扫描不解压。其余参数它自己从 OCEAN_ARCHIVE / archives.tsv 读。
  [ "$DO_INSTALL" = 1 ] && r_args=() || r_args=(--check)
  "$RESTORE" "${r_args[@]+"${r_args[@]}"}" 2>&1 | sed 's/^/  /'
  r_rc=${PIPESTATUS[0]}
  case "$r_rc" in
    0) ok "大文件全部就位" ;;
    1) bad "大文件没齐（见上面 restore.sh 的扫描结果）"
       info "包放进 ${OCEAN_ARCHIVE:-$REPO/archive}/ 再跑一次就会自动解。"
       info "它是**全有才动手** —— 手上只有一部分包时，指名解那几个:"
       info "  ./archive/restore.sh data_    # 按包名子串挑，日期和 .tar 可以不写"
       info "  ./archive/restore.sh web      # 只解 web-demo 要的那个（= data 包）"
       info "  ./archive/restore.sh          # 全部（缺一个就什么都不解）" ;;
    *) bad "restore.sh 退出码 $r_rc（用法错误？）" ;;
  esac
fi

# FUNWAVE-TVD —— 第三方 Boussinesq 求解器（fwv 线 prior 的来源）。不进版本库，
# 是别人仓库的干净 clone，按需拉。tag 钉死，免得上游动了对不上。
FW_DIR="$REPO/FUNWAVE-TVD"
FW_URL=https://github.com/fengyanshi/FUNWAVE-TVD.git
FW_TAG=Version_3.6
if [ -d "$FW_DIR/.git" ]; then
  ok "FUNWAVE-TVD（$(cd "$FW_DIR" && git describe --tags 2>/dev/null || echo '?')）"
  if [ -n "$(cd "$FW_DIR" && git status --porcelain 2>/dev/null)" ]; then
    info "有本地改动 —— 想让别人也拿到，存成 patch 提交进来:"
    info "  cd FUNWAVE-TVD && git diff > ../funwave.patch"
  fi
elif [ "$DO_INSTALL" = 1 ]; then
  echo "  git clone $FW_URL（$FW_TAG，约 250 M）"
  if git clone --depth 1 --branch "$FW_TAG" "$FW_URL" "$FW_DIR"; then
    ok "FUNWAVE-TVD clone 好了"
    if [ -f "$REPO/funwave.patch" ]; then
      (cd "$FW_DIR" && git apply "$REPO/funwave.patch") \
        && ok "funwave.patch 已打上" || warn "funwave.patch 打不上，自己看看"
    fi
  else
    bad "clone 失败（登录节点连不上 github 就换计算节点再试）"
  fi
else
  info "没有 FUNWAVE-TVD（去掉 --check 会自动 clone $FW_TAG，约 250 M）"
fi

# ============================================================
step "小结"
printf '  环境: %s\n' "${ENV_PREFIX:-无}"
[ "${untrusted:-0}" = 1 ] && printf '  %s⚠ 本轮包检测没测准（numpy 都起不来），上面的结果只能参考%s\n' "$Y" "$N"
printf '  缺件: %s%d%s   提醒: %s%d%s\n' \
  "$([ "$MISSING" -gt 0 ] && echo "$R" || echo "$G")" "$MISSING" "$N" \
  "$([ "$WARNED" -gt 0 ] && echo "$Y" || echo "$G")" "$WARNED" "$N"
cat <<TIP

  每次开新 shell:
    source $REPO/activate.sh          # 找 conda + 激活环境，并导出 \$REPO

  然后:
    cd code && sbatch run.sh          # 训练（默认 dgxh，可能要等 2-3 天）
    cd code && sbatch vis.sh          # 出图
  sbatch 脚本自己会 source activate.sh，不用先激活。
  六条路线怎么回事、数据放在哪，看 README.md。
TIP

[ "$MISSING" -gt 0 ] && exit 1
exit 0
