#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE=".dictate.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "dictate is not running (no pid file)"
    exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
    kill -INT "$PID"
    for _ in $(seq 1 20); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "dictate did not stop gracefully, killing"
        kill -KILL "$PID"
    fi
    echo "dictate stopped"
else
    echo "dictate is not running (stale pid file)"
fi
rm -f "$PID_FILE"
