#!/usr/bin/env bash
# Rebuild the static evidence-board UI into public/ for Vercel deploy.
#
# Run this whenever runs/ledger.jsonl, runs/trials/*/metrics.json,
# runs/baseline/metrics.json, runs/discovery_ledger.jsonl, or trials/T*.json
# change and you want the deployed UI to reflect the latest data. Commit
# public/ alongside the data change so Vercel picks it up on the next push.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
  echo "build_ui.sh: needs Python >= 3.11 (StrEnum). Set PYTHON=/path/to/python3.11+ and retry." >&2
  exit 1
fi

"$PYTHON" -m autoalphafold3.ui.build --runs runs --out public
echo "build_ui.sh: wrote public/ — commit and push to redeploy on Vercel."
