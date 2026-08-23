#!/bin/bash
# 大文件包的还原器 —— **两段式，全有才动手**。
#
#   阶段一 扫描: 只读。逐项核对该有的东西在不在、包的 md5 对不对。
#                任何一项缺失或损坏 -> 把问题一次性全列出来, 直接 exit 1, 不解压。
#   阶段二 释放: 阶段一全绿才进。按 archives.tsv 的「解压到」把包铺回去。
#   阶段三 核对: **每次都跑**, 不是可选项。拿 <包名>.manifest 把刚解出来的文件
#                逐个 md5 核一遍 —— 包整体 md5 对了不等于解出来的东西对
#                （磁盘满了、解到一半被打断, tar 未必报错）。
#
# 为什么要全有才动手: 半套产物比没有更难查 —— 跑到一半报「缺 xxx」时前面已经
# 铺了几 G 下去, 分不清哪些是这次解的。宁可先骂一顿再开始。
#
# 包清单在仓库根的 archives.tsv（6 列 TAB, 格式说明写在那个文件头部）。
#
# 目录分工（<仓库>/archive/ 一个目录就够, 从云端下下来的包直接丢这儿）:
#   restore.sh          本脚本            进版本库
#   *.manifest          各包的逐文件校验单  进版本库, clone 就有
#   *.tar               包本体            不进版本库, 从云端硬盘下载后放这里
#                       换个地方放就 export OCEAN_ARCHIVE=/path
#
# **只补缺, 不覆盖**（tar --skip-old-files）。包跟版本库有重叠 —— data 包里带着
# data/3d/crop_fields.{py,sh} 和 data/fwv/{make_cases.py,wk_check.py,TK94/input.txt,
# TK94/gauges.txt}（算例的出身证明, 但版本库里也有一份）, legacy 包里带着历史线的
# 源码。不加这个开关的话, 谁改过 input.txt, 一次还原就被打包那天的旧版本静默盖掉,
# 而且盖了不报错。
# 「包外必需输入」是另一码事: 它们**不在任何包里**（都在版本库里）, 但缺了 web-demo
# 跑不起来, 所以一并体检。⚠️ 往这个清单里加东西前先确认它真的不在包里 —— 包里的
# 文件写进去会造成死锁: 阶段一因为「缺」而 exit 1, 而它恰恰要靠解包才能出现。
#
# 用法:
#   ./restore.sh                    还原 archives.tsv 里所有包
#   ./restore.sh web                只解 web-demo 要的那个包（= data 包, 48.6 G;
#                                   demo 真正读到的约 3.1 G, 但包只能整个解）
#   ./restore.sh <包名> [<包名>...]  指名还原
#   ./restore.sh --check            只扫描, 什么都不解
#   ./restore.sh --skip-input-check 不体检「包外必需输入」
#   ./restore.sh --verify-all       连「已就位」没解的包也一并逐文件核对
#                                   （要把内容全读一遍, 几十 G 起步, 所以不是默认）
#
# 退出码: 0 = 全绿（或已就位）, 1 = 有缺失/损坏, 2 = 用法错误

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
    -h|--help)           sed -n '2,41p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)                  echo "未知选项: $a（-h 看用法）" >&2; exit 2 ;;
    web)                 WANT+=("data_") ;;      # web-demo 要的都在 data 包里
    *)                   WANT+=("$a") ;;
  esac
done

c_ok=$'\033[32m'; c_bad=$'\033[31m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
[ -t 1 ] || { c_ok=; c_bad=; c_dim=; c_off=; }
ok()   { echo "  ${c_ok}✓${c_off} $*"; }
bad()  { echo "  ${c_bad}✗${c_off} $*"; }
skip() { echo "  ${c_dim}·${c_off} $*"; }

# 选中判定: 不给参数 = 全要; 给了就按子串匹配（可以只写 data_ 不写日期和 .tar）
selected() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  local n="$1" w
  for w in "${WANT[@]}"; do [[ "$n" == *"$w"* ]] && return 0; done
  return 1
}

# 已就位 = 探测路径是个非空目录（跟 setup.sh 的判定一致）。
# ⚠️ .gitkeep 不算数 —— 那是版本库里的目录骨架（让 clone 下来能看出东西该放哪），
# 不排掉的话骨架本身就会让每个包都被判成「已就位」，数据没解也不会有人提醒。
in_place() {
  local p="$REPO/$1"
  [ -d "$p" ] || return 1
  [ -n "$(ls -A "$p" 2>/dev/null | grep -v '^\.gitkeep$')" ]
}

PROBLEMS=0
declare -a TODO_NAME=() TODO_DEST=() TODO_PROBE=() INPLACE_NAME=()

# ─────────────────────── 阶段一 扫描 ───────────────────────

echo "仓库   : $REPO"
echo "包目录 : $ARC_DIR"
echo
echo "[1/3] 扫描"

if [ ! -f "$ARC_LIST" ]; then
  bad "缺 archives.tsv —— 版本库里应该有, 包清单靠它"
  exit 1
fi

echo "--- 大文件包 ---"
n_row=0
while IFS=$'\t' read -r a_name a_dest a_probe a_size a_md5 a_url || [ -n "${a_name:-}" ]; do
  case "${a_name:-}" in ''|'#'*) continue ;; esac
  selected "$a_name" || continue
  n_row=$((n_row + 1))

  if in_place "$a_probe"; then
    skip "$a_probe 已就位, 跳过 $a_name"
    INPLACE_NAME+=("$a_name")
    continue
  fi

  local_tar="$ARC_DIR/$a_name"
  if [ ! -f "$local_tar" ]; then
    bad "缺包 $a_name（$a_size）—— $a_probe 也不在"
    case "${a_url:-}" in
      '')        echo "      archives.tsv 第 6 列没填地址, 找仓库主人要" ;;
      gdrive:*)  echo "      python -m gdown ${a_url#gdrive:} -O $ARC_DIR/$a_name" ;;
      *)         echo "      curl -fL -o '$ARC_DIR/$a_name' '$a_url'" ;;
    esac
    PROBLEMS=$((PROBLEMS + 1))
    continue
  fi

  if [ -n "${a_md5:-}" ]; then
    got="$(md5sum "$local_tar" | awk '{print $1}')"
    if [ "$got" != "$a_md5" ]; then
      bad "$a_name md5 不符（期望 $a_md5, 实际 $got）"
      echo "      下载不完整或云端换了文件。删掉重下, 或核对 archives.tsv"
      PROBLEMS=$((PROBLEMS + 1))
      continue
    fi
  fi

  ok "$a_name（$a_size）待解到 ${a_dest:-.}/"
  TODO_NAME+=("$a_name"); TODO_DEST+=("${a_dest:-.}"); TODO_PROBE+=("$a_probe")
done < "$ARC_LIST"

# 一行都没匹配上 = 包名打错了或清单变了。别让它一路绿到底报「已就位」。
if [ "$n_row" = 0 ]; then
  bad "archives.tsv 里没有匹配的行（选择条件: ${WANT[*]:-全部}）"
  echo "      现有的包:"
  grep -v '^#' "$ARC_LIST" | cut -f1 | grep -v '^$' | sed 's/^/        /'
  exit 2
fi

# 包外必需输入 —— 都在版本库里, 不在任何包里, 缺了说明 clone 不完整。
# ⚠️ data/3d/cropped_0.05/ 下的网格元数据（coords / lbo / slice_y0.30 / chunk_010_times
# / stats_*）曾经列在这儿, 但 20260822 那次重新打包已经把它们收进 data_*.tar 了 ——
# 留在这儿会死锁（阶段一说「缺」就 exit 1, 而它们要靠解包才会出现）。它们的完整性
# 由阶段三的 data_*.manifest 逐文件核对负责。
CK=results/web/model/hpm_fw_aU_h128/2026-08-12_15-31-45   # demo 默认权重, 在版本库里
INPUTS=(
  "$CK/checkpoints/best.pt|模型权重"
  "$CK/.hydra/config.yaml|存档超参（vis_adp.sh 读它）"
  "code/web-demo/server/target/release/wave-demo|后端二进制"
  "code/web-demo/web/dist/index.html|前端产物"
)

if [ "$DO_INPUT" = 1 ]; then
  echo "--- 包外必需输入 ---"
  for item in "${INPUTS[@]}"; do
    p="${item%%|*}"; why="${item#*|}"
    if [ -s "$REPO/$p" ]; then
      ok "$p"
    else
      bad "缺 $p —— $why"
      PROBLEMS=$((PROBLEMS + 1))
    fi
  done
fi

echo
if [ "$PROBLEMS" -gt 0 ]; then
  echo "${c_bad}扫描不通过: $PROBLEMS 项有问题, 什么都没解压。${c_off}"
  exit 1
fi
echo "${c_ok}扫描通过。${c_off}"

# ─────────────────────── 阶段二 释放 ───────────────────────

echo
if [ "$DO_EXTRACT" = 0 ]; then
  echo "[2/3] 释放 —— --check 模式, 跳过（核对也一并跳过）"
  exit 0
fi

echo "[2/3] 释放"
if [ ${#TODO_NAME[@]} -eq 0 ]; then
  skip "没有要解的包（都已就位）"
else
  for i in "${!TODO_NAME[@]}"; do
    name="${TODO_NAME[$i]}"; dest="${TODO_DEST[$i]}"; probe="${TODO_PROBE[$i]}"
    echo "  解 $name -> $dest/"
    mkdir -p "$REPO/$dest"
    if ! tar -xf "$ARC_DIR/$name" -C "$REPO/$dest" --skip-old-files; then
      bad "$name 解压失败"
      exit 1
    fi
    if in_place "$probe"; then
      ok "$probe"
    else
      bad "解完了但 $probe 还是空的 —— 包内路径跟 archives.tsv 的「解压到」对不上?"
      exit 1
    fi
  done
fi

# 逐文件核对 —— manifest 里的路径是相对仓库根的, 所以要在仓库根上跑。
# 无条件执行: 走到这里说明确实动过盘, 不核对等于没做完。
echo
echo "[3/3] 逐文件核对"
declare -a CHECK_NAME=("${TODO_NAME[@]:-}")
[ "$DO_VERIFY_ALL" = 1 ] && CHECK_NAME+=("${INPLACE_NAME[@]:-}")
n_check=0
for name in "${CHECK_NAME[@]:-}"; do
  [ -n "$name" ] || continue
  n_check=$((n_check + 1))
  man="$ARC_DIR/${name%.tar}.manifest"
  if [ ! -f "$man" ]; then
    bad "缺 $(basename "$man") —— 没法核对 $name"
    exit 1
  fi
  if (cd "$REPO" && md5sum -c "$man" > /dev/null 2>&1); then
    ok "$(basename "$man") 全部一致"
  else
    bad "$(basename "$man") 有对不上的文件:"
    (cd "$REPO" && md5sum -c "$man" 2>&1 | grep -v ': OK$' | head -10)
    exit 1
  fi
done
[ "$n_check" = 0 ] && skip "这次没解任何包, 无需核对（要连已就位的一起核, 加 --verify-all）"

# demo 的运行时输出目录, 空的也得在
mkdir -p "$REPO/results/web/priors" "$REPO/results/web/vis"
echo
if [ "$DO_INPUT" = 1 ] && [ ${#WANT[@]} -eq 0 ]; then
  echo "${c_ok}全部就位。${c_off}web-demo: cd code/web-demo && ./start.sh"
else
  echo "${c_ok}选中的包已就位。${c_off}（没做全量体检, 完整检查跑 ./restore.sh --check）"
fi
