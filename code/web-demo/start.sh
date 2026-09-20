#!/bin/bash
# Start the backend. Runs on a login node of the cluster.
#
# All state lives under <models>/results/web/: submissions.json / wave-demo.log / wave-demo.pid
set -euo pipefail
cd "$(dirname "$0")"                       # -> <models>/code/web-demo

# Derive the models root from the script's own location; no absolute path is hard-coded, so the
# whole tree runs wherever it is cloned.
ROOT="${WAVE_ROOT:-$(cd ../.. && pwd)}"
export WAVE_ROOT="$ROOT"
PID="$ROOT/results/web/wave-demo.pid"
BIN=server/target/release/wave-demo

[ -x "$BIN" ] || { echo "Not built yet: module load rust/1.92 && (cd server && cargo build --release)"; exit 1; }

# If one is already running, do not start a second -- two processes would fight over the same
# socket and the same submissions.json
if [ -f "$PID" ]; then
    read -r oldpid oldhost < "$PID"
    if [ "$oldhost" = "$(hostname -s)" ] && kill -0 "$oldpid" 2>/dev/null; then
        echo "Already running on this machine (pid $oldpid). To restart, run ./stop.sh first"; exit 1
    fi
    echo "The pid file is stale ($oldpid @ $oldhost); ignoring it"
fi

nohup "$BIN" >/dev/null 2>&1 &
sleep 2
if [ -f "$PID" ]; then
    read -r p h < "$PID"
    echo "Started: pid $p @ $h"
    SOCK="${WAVE_SOCK:-$ROOT/results/web/wave-demo.sock}"
    echo "socket : $SOCK"
    echo "log    : $ROOT/results/web/wave-demo.log"
    echo
    # The submit-b that hostname -s returns only resolves inside the cluster, and the
    # submit-b.ib.coehpc that hostname -f returns is an InfiniBand-internal name -- connecting
    # from outside needs the public domain.
    PUBLIC="${WAVE_PUBLIC_HOST:-$(hostname -s).${WAVE_PUBLIC_DOMAIN:-hpc.engr.oregonstate.edu}}"
    IP=$(hostname -I 2>/dev/null | cut -d' ' -f1)
    echo "The service listens only on the socket above; it listens on no port. To view it from"
    echo "your own machine, open another terminal **locally** and run (it must connect to this"
    echo "very host, not the round-robin submit):"
    echo
    echo "  ssh -L 8788:$SOCK $USER@$PUBLIC"
    [ -n "$IP" ] && echo "  If that does not resolve, use the IP: ssh -L 8788:$SOCK $USER@$IP"
    echo
    echo "Keep that terminal open, then open http://localhost:8788 in a browser"
else
    echo "Failed to start; see the log: $ROOT/results/web/wave-demo.log"; exit 1
fi
