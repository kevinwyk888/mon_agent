#!/usr/bin/env bash
# Setup the Python virtual environment for mon-agent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="${1:-${MON_AGENT_VENV:-$HOME/.venvs/mon_agent_env}}"
MINI_SWE_AGENT_DIR="${MINI_SWE_AGENT_DIR:-$PROJECT_DIR/../mini-SWE-agent}"

if command -v module >/dev/null 2>&1; then
  module load python/3.11.5 || true
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 required. Try: module load python/3.11.5")
PY

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [[ -d "$MINI_SWE_AGENT_DIR" ]]; then
  echo "Installing mini-SWE-agent from local checkout: $MINI_SWE_AGENT_DIR"
  python -m pip install -e "$MINI_SWE_AGENT_DIR"
  python -m pip install -e "$PROJECT_DIR" --no-deps
  python -m pip install "vllm>=0.6.0"
else
  echo "Local mini-SWE-agent checkout not found."
  echo "Installing mon-agent and pinned mini-SWE-agent dependency from GitHub."
  python -m pip install -e "$PROJECT_DIR[vllm]"
fi

echo "========================================"
echo " Environment ready: $ENV_DIR"
echo " Activate: source $ENV_DIR/bin/activate"
echo "========================================"
