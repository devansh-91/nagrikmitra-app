#!/usr/bin/env bash
# Start the FastAPI service (routing engine + officer inbox API).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "No .venv found. Create one first:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
export $(grep -v '^#' .env 2>/dev/null | xargs) 2>/dev/null || true

uvicorn api:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
