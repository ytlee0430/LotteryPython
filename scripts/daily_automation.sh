#!/bin/bash
#
# Daily Automation Shell Wrapper
#
# This script wraps the Python daily automation script for use with
# cron or launchd on macOS/Linux.
#
# Usage:
#   ./daily_automation.sh [options]
#
# Options are passed directly to daily_automation.py
#

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Log function
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# Change to project directory
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    log "Activated virtual environment"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    log "Activated virtual environment"
fi

# Run the Python automation script
log "Starting daily automation..."
"$PYTHON_BIN" "$SCRIPT_DIR/daily_automation.py" "$@"
exit_code=$?

log "Daily automation completed with exit code: $exit_code"
exit $exit_code
