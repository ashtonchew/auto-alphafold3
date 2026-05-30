#!/usr/bin/env bash
# scripts/up.sh — one-command setup for auto-AlphaFold3 with Raindrop Workshop tracing.
#
# Usage:
#   bash scripts/up.sh                    # set up everything and drop into a ready shell
#   bash scripts/up.sh <command>          # set up everything and run <command>
#
# This script is idempotent. Safe to run repeatedly. It will skip steps that are
# already done.
#
# What it does (in order):
#   1. Installs the Raindrop Workshop daemon if missing
#   2. Creates a local Python venv at .venv/ if missing
#   3. Installs project deps and the optional OpenTelemetry packages
#   4. Starts the Workshop daemon in the background if not running
#   5. Sets the RAINDROP_LOCAL_DEBUGGER env var for tracing
#   6. Runs your command (or drops you into a shell with the env preconfigured)
#
# If you do not want tracing, simply unset RAINDROP_LOCAL_DEBUGGER. The
# project still runs normally; trace logging becomes a silent no-op.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Install Workshop daemon if missing
if ! command -v raindrop &>/dev/null; then
    echo "[up.sh] Installing Raindrop Workshop daemon..."
    curl -fsSL https://raindrop.sh/install | bash
    # The installer typically drops the binary in ~/.raindrop/bin or ~/.local/bin
    export PATH="$HOME/.raindrop/bin:$HOME/.local/bin:$PATH"
fi

# 2. Create venv if missing
if [ ! -d ".venv" ]; then
    echo "[up.sh] Creating Python venv at .venv/..."
    if command -v python3.12 &>/dev/null; then
        python3.12 -m venv .venv
    elif command -v python3.11 &>/dev/null; then
        python3.11 -m venv .venv
    elif command -v python3.13 &>/dev/null; then
        python3.13 -m venv .venv
    else
        python3 -m venv .venv
    fi
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# 3. Install project deps + OpenTelemetry tracing packages (idempotent; quiet)
echo "[up.sh] Installing dependencies (quiet)..."
pip install --quiet -r requirements.txt 2>/dev/null || true
pip install --quiet opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http 2>/dev/null || true

# 4. Start Workshop daemon if not running
if ! curl -fsS http://localhost:5899/health &>/dev/null 2>&1; then
    echo "[up.sh] Starting Workshop daemon at localhost:5899..."
    nohup raindrop workshop >/tmp/raindrop-workshop.log 2>&1 &
    # Wait up to 15 seconds for the daemon to come up
    for i in $(seq 1 15); do
        sleep 1
        if curl -fsS http://localhost:5899/health &>/dev/null 2>&1; then
            break
        fi
    done
fi

# 5. Enable tracing for whatever runs after this
export RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/

# 6. Run the user's command, or drop into a shell
if [ $# -eq 0 ]; then
    echo ""
    echo "[up.sh] Setup complete."
    echo "[up.sh]   Workshop UI: http://localhost:5899"
    echo "[up.sh]   RAINDROP_LOCAL_DEBUGGER is set in this shell."
    echo "[up.sh]   Exit this shell to leave the environment."
    echo ""
    exec "${SHELL:-/bin/bash}"
else
    echo "[up.sh] Running: $*"
    exec "$@"
fi
