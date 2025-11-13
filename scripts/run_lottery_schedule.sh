#!/bin/bash
set -euo pipefail

# This script runs the appropriate LotteryPython commands depending on the
# current weekday. It is designed for use with cron or launchd on macOS.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$PROJECT_ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

weekday="$(date +%u)"  # 1=Monday, 7=Sunday

case "$weekday" in
  2|5)
    log "Running Big Lottery update and result save tasks."
    "$PYTHON_BIN" -m lotterypython --update --type big
    "$PYTHON_BIN" -m lotterypython --type big --save-results
    ;;
  1|4)
    log "Running Super Lottery update and result save tasks."
    "$PYTHON_BIN" -m lotterypython --update --type super
    "$PYTHON_BIN" -m lotterypython --type super --save-results
    ;;
  *)
    log "No scheduled LotteryPython tasks for weekday $weekday."
    ;;
esac
