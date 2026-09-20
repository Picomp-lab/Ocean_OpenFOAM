#!/bin/bash
# Stop the backend. Finds the process through the pid file -- not pkill, whose pattern would
# also match the ssh session itself.
set -euo pipefail
cd "$(dirname "$0")"                       # -> <models>/code/web-demo
ROOT="${WAVE_ROOT:-$(cd ../.. && pwd)}"
PID="$ROOT/results/web/wave-demo.pid"

[ -f "$PID" ] || { echo "No pid file; it is probably not running"; exit 0; }
read -r pid host < "$PID"

if [ "$host" != "$(hostname -s)" ]; then
    echo "It is running on $host, not this machine ($(hostname -s))."
    echo "Stop it over there: ssh $host '$(pwd)/stop.sh'"
    exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
    echo "pid $pid is already gone; clearing the pid file"; rm -f "$PID"; exit 0
fi

# TERM lets it clean up the socket and the pid itself (main.rs handles SIGTERM)
kill "$pid"
for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$pid" 2>/dev/null; then
    echo "Still alive after TERM; killing hard"; kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID" "${WAVE_SOCK:-$ROOT/results/web/wave-demo.sock}"
fi
echo "Stopped (pid $pid @ $host)"
