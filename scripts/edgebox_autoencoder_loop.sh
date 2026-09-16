#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${AQUAFL_PYTHON:-python3}"
mkdir -p "$PROJECT_ROOT/logs"

while true; do
  cd "$PROJECT_ROOT" || exit 1

  if [ -f "$PROJECT_ROOT/dados/edgebox/edgebox_autoencoder_model.json" ]; then
    "$PYTHON_BIN" -m fluxos.edgebox.edgebox_autoencoder infer >> logs/edgebox_autoencoder.log 2>&1
  fi

  if [ -f "$PROJECT_ROOT/fluxos/edgebox/edgebox_site_autoencoder.py" ]; then
    "$PYTHON_BIN" -m fluxos.edgebox.edgebox_site_autoencoder >> logs/edgebox_site.log 2>&1
  fi

  if [ -f "$PROJECT_ROOT/fluxos/edgebox/edgebox_db_sync.py" ]; then
    "$PYTHON_BIN" -m fluxos.edgebox.edgebox_db_sync >> logs/edgebox_db.log 2>&1
  fi

  sleep 10
done
