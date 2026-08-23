#!/bin/bash
# 起后端。跑在集群登录节点上。
#
# 状态全在 <models>/results/web/ 下：submissions.json / wave-demo.log / wave-demo.pid
set -euo pipefail
cd "$(dirname "$0")"                       # → <models>/code/web-demo

# 从脚本自身位置推出 models 根，不写死任何绝对路径。整棵树 clone 到哪都能跑。
ROOT="${WAVE_ROOT:-$(cd ../.. && pwd)}"
export WAVE_ROOT="$ROOT"
PID="$ROOT/results/web/wave-demo.pid"
BIN=server/target/release/wave-demo

[ -x "$BIN" ] || { echo "还没编译：module load rust/1.92 && (cd server && cargo build --release)"; exit 1; }

# 已经在跑就别起第二个 —— 两个进程会抢同一个 socket 和同一份 submissions.json
if [ -f "$PID" ]; then
    read -r oldpid oldhost < "$PID"
    if [ "$oldhost" = "$(hostname -s)" ] && kill -0 "$oldpid" 2>/dev/null; then
        echo "已经在本机跑着了（pid $oldpid）。要重起先 ./stop.sh"; exit 1
    fi
    echo "pid 文件是陈的（$oldpid @ $oldhost），忽略"
fi

nohup "$BIN" >/dev/null 2>&1 &
sleep 2
if [ -f "$PID" ]; then
    read -r p h < "$PID"
    echo "已启动：pid $p @ $h"
    SOCK="${WAVE_SOCK:-$ROOT/results/web/wave-demo.sock}"
    echo "socket : $SOCK"
    echo "日志   : $ROOT/results/web/wave-demo.log"
    echo
    # hostname -s 给的 submit-b 只在集群内部能解析，hostname -f 给的
    # submit-b.ib.coehpc 是 InfiniBand 内部名 —— 从外面连要用对外域名。
    PUBLIC="${WAVE_PUBLIC_HOST:-$(hostname -s).${WAVE_PUBLIC_DOMAIN:-hpc.engr.oregonstate.edu}}"
    IP=$(hostname -I 2>/dev/null | cut -d' ' -f1)
    echo "服务只监听上面这个 socket，不监听任何端口。要从自己机器上看，"
    echo "在**本地**另开一个终端跑（连的必须是这一台，不能是轮询的 submit）："
    echo
    echo "  ssh -L 8788:$SOCK $USER@$PUBLIC"
    [ -n "$IP" ] && echo "  解析不了的话用 IP：ssh -L 8788:$SOCK $USER@$IP"
    echo
    echo "保持那个终端开着，然后浏览器开 http://localhost:8788"
else
    echo "启动失败，看日志：$ROOT/results/web/wave-demo.log"; exit 1
fi
