#!/usr/bin/env bash
# Run the llm-lite inference API from repo root.
# Usage: from repo root, run:  ./api/run.sh   or   bash api/run.sh

set -e
cd "$(dirname "$0")/.."

if [ -n "$VIRTUAL_ENV" ]; then
  echo "Using venv: $VIRTUAL_ENV"
else
  if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
  fi
fi

if [ ! -f "models/transformer/ckpt.pt" ]; then
  echo "Warning: models/transformer/ckpt.pt not found. Train first:"
  echo "  python scripts/train_transformer_causal.py"
  echo ""
fi

echo "Starting API on http://127.0.0.1:8000"
exec uvicorn api.app.main:app --host 0.0.0.0 --port 8000 "$@"
