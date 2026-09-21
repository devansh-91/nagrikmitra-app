#!/usr/bin/env bash
# Start the Nagarik Mitra MCP server (stdio transport) for an MCP client
# (Claude Desktop, etc.) to connect to.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "No .venv found. Create one first:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
export $(grep -v '^#' .env 2>/dev/null | xargs) 2>/dev/null || true

python mcp_server.py
