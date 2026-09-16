#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${AQUAFL_PYTHON:-python3}"
mkdir -p "$PROJECT_ROOT/logs"

while true; do
  cd "$PROJECT_ROOT" || exit 1

  "$PYTHON_BIN" -m fluxos.edgebox.edgebox_node >> logs/edgebox_loop.log 2>&1

  sleep 10
done
