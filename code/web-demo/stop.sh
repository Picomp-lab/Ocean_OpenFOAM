#!/bin/bash
# 停后端。靠 pid 文件找进程 —— 不用 pkill，那个模式会把 ssh 会话自己也匹配掉。
set -euo pipefail
cd "$(dirname "$0")"                       # → <models>/code/web-demo
ROOT="${WAVE_ROOT:-$(cd ../.. && pwd)}"
PID="$ROOT/results/web/wave-demo.pid"

[ -f "$PID" ] || { echo "没有 pid 文件，应该没在跑"; exit 0; }
read -r pid host < "$PID"

if [ "$host" != "$(hostname -s)" ]; then
    echo "它跑在 $host 上，不是这台（$(hostname -s)）。"
    echo "去那台停：ssh $host '$(pwd)/stop.sh'"
    exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
    echo "pid $pid 已经不在了，清掉 pid 文件"; rm -f "$PID"; exit 0
fi

# TERM 让它自己收拾 socket 和 pid（main.rs 里接了 SIGTERM）
kill "$pid"
for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$pid" 2>/dev/null; then
    echo "TERM 之后还活着，强杀"; kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID" "${WAVE_SOCK:-$ROOT/results/web/wave-demo.sock}"
fi
echo "已停止（pid $pid @ $host）"
