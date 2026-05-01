#!/usr/bin/env bash
# Setup the Python virtual environment for mon-agent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_DIR="${1:-${MON_AGENT_VENV:-$HOME/.venvs/mon_agent_env}}"

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

# Install mini-SWE-agent from sibling directory
python -m pip install -e "$PROJECT_DIR/../mini-SWE-agent"

# Install vLLM and other dependencies
python -m pip install vllm>=0.6.0 litellm openai

echo "========================================"
echo " Environment ready: $ENV_DIR"
echo " Activate: source $ENV_DIR/bin/activate"
echo "========================================"
