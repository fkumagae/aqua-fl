#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${AQUAFL_PYTHON:-python3}"
mkdir -p "$PROJECT_ROOT/logs"

while true; do
  cd "$PROJECT_ROOT" || exit 1
  "$PYTHON_BIN" -m fluxos.clima.aqua_fl_demo > logs/update.log 2>&1
  sleep 60
done
