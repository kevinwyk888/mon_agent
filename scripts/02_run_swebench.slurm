#!/usr/bin/env bash
#SBATCH --job-name=mon-swebench
#SBATCH --account=alkontar0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/swebench-%j.out

# ============================================================
# Step 2: Run mini-SWE-agent on SWE-bench Lite.
#
# Usage (option A - pass URL directly):
#   sbatch --export=ALL,VLLM_URL=http://gl1234:8000/v1 scripts/02_run_swebench.slurm
#
# Usage (option B - pass the vLLM job ID, auto-reads hostfile):
#   sbatch --export=ALL,VLLM_JOB_ID=12345678 scripts/02_run_swebench.slurm
#
# Optional overrides:
#   SUBSET=lite|verified  (default: lite)
#   SPLIT=dev|test        (default: test)
#   SLICE=0:10            (default: all, use small slice for testing)
#   WORKERS=4             (default: 1)
#   CONFIG=path/to/yaml   (default: configs/swebench_lite.yaml)
# ============================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load python/3.11.5 || true
  module load singularity || true
fi

ENV_DIR="${MON_AGENT_VENV:-$HOME/.venvs/agent_env}"
source "$ENV_DIR/bin/activate"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Resolve vLLM server URL ---
if [[ -n "${VLLM_URL:-}" ]]; then
  OPENAI_BASE_URL="$VLLM_URL"
elif [[ -n "${VLLM_JOB_ID:-}" ]]; then
  HOSTFILE="${PROJECT_DIR}/logs/vllm_host_${VLLM_JOB_ID}.txt"
  if [[ ! -f "$HOSTFILE" ]]; then
    echo "ERROR: Hostfile not found: $HOSTFILE"
    echo "Is the vLLM job ${VLLM_JOB_ID} running?"
    exit 1
  fi
  OPENAI_BASE_URL="$(cat "$HOSTFILE")"
else
  echo "ERROR: Set VLLM_URL or VLLM_JOB_ID"
  echo "  sbatch --export=ALL,VLLM_URL=http://gl1234:8000/v1 scripts/02_run_swebench.slurm"
  echo "  sbatch --export=ALL,VLLM_JOB_ID=12345678 scripts/02_run_swebench.slurm"
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# --- Storage strategy for Singularity ---
# Layer cache → scratch (persistent, avoids re-downloading images)
# Sandbox working dir → node-local /tmp (fast SSD, no quota, auto-cleaned)
SCRATCH_BASE="/scratch/alkontar_root/alkontar0/rongbo"
export SINGULARITY_CACHEDIR="${SCRATCH_BASE}/singularity_cache"
export SINGULARITY_TMPDIR="/tmp/${USER}/singularity_tmp_${SLURM_JOB_ID:-0}"
export TMPDIR="/tmp/${USER}/mon_agent_${SLURM_JOB_ID:-0}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$TMPDIR"

# --- Configuration ---
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-1}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/swebench_lite.yaml}"
REGISTRY="${PROJECT_DIR}/configs/registry.json"

# --- Output to scratch ---
SCRATCH_DIR="/scratch/alkontar_root/alkontar0/rongbo/mon-agent"
RESULTS_DIR="${SCRATCH_DIR}/results/${SUBSET}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo " SWE-bench Evaluation"
echo " vLLM URL: ${OPENAI_BASE_URL}"
echo " Subset:   ${SUBSET}"
echo " Split:    ${SPLIT}"
echo " Config:   ${CONFIG}"
echo " Output:   ${RESULTS_DIR}"
echo " Workers:  ${WORKERS}"
echo "========================================"

# --- Build the command ---
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

CMD=(
  python -m mon_agent.runner
  --subset "$SUBSET"
  --split "$SPLIT"
  --output "$RESULTS_DIR"
  --workers "$WORKERS"
  -c "$CONFIG"
  -c "model.model_kwargs.api_base=${OPENAI_BASE_URL}"
)

# Optional: slice for quick testing (e.g., SLICE=0:5)
if [[ -n "${SLICE:-}" ]]; then
  CMD+=(--slice "$SLICE")
fi

export LITELLM_MODEL_REGISTRY_PATH="$REGISTRY"

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"

echo "========================================"
echo " Done! Results at: ${RESULTS_DIR}"
echo "========================================"
