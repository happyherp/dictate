#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE=".dictate.pid"
LOG_FILE="dictate.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "dictate already running (PID $(cat "$PID_FILE"))"
    exit 0
fi

nohup .venv/bin/python -m dictate > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "dictate started (PID $!), logging to $LOG_FILE"
