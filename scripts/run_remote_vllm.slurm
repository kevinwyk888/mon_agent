#!/usr/bin/env bash
#SBATCH --job-name=mon-remote
#SBATCH --account=alkontar0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mon-remote-%j.out

# ============================================================
# CPU-only job: agent runs here, vLLM runs on RunPod (remote).
# No GPU needed — all inference is done via remote API.
#
# Usage:
#   # Run all 300 instances:
#   VLLM_API_BASE="https://<pod-id>-8000.proxy.runpod.net/v1" \
#     sbatch scripts/run_remote_vllm.slurm
#
#   # Run a slice:
#   VLLM_API_BASE="https://<pod-id>-8000.proxy.runpod.net/v1" \
#   SLICE=0:5 sbatch scripts/run_remote_vllm.slurm
# ============================================================

set -euo pipefail

# --- Validate remote API ---
: "${VLLM_API_BASE:?ERROR: Set VLLM_API_BASE before submitting, e.g. VLLM_API_BASE=https://<pod-id>-8000.proxy.runpod.net/v1}"

if command -v module >/dev/null 2>&1; then
  module load python/3.11.5 || true
  module load singularity || true
fi

AGENT_ENV_DIR="${MON_AGENT_VENV:-$HOME/.venvs/agent_env}"
source "$AGENT_ENV_DIR/bin/activate"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# --- Model config ---
MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"

# --- Storage ---
SCRATCH_BASE="/scratch/alkontar_root/alkontar0/rongbo"
NODE_TMP="/tmp/${USER}/mon_${SLURM_JOB_ID}"

export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf_cache/hub"
export SINGULARITY_CACHEDIR="${SCRATCH_BASE}/singularity_cache"
export SINGULARITY_TMPDIR="${NODE_TMP}/singularity_tmp"
export TMPDIR="${NODE_TMP}/tmpdir"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$TMPDIR"

# --- SWE-bench config ---
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-1}"
SLICE="${SLICE:-}"
CONFIG="${PROJECT_DIR}/configs/swebench_lite.yaml"
REGISTRY="${PROJECT_DIR}/configs/registry.json"
RESULTS_DIR="${SCRATCH_BASE}/mon-agent/results/${SUBSET}_remote_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# --- Verify remote vLLM is reachable ---
echo "Checking remote vLLM at ${VLLM_API_BASE} ..."
HEALTH_URL="${VLLM_API_BASE%/v1}/health"
if ! curl -sf --max-time 15 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "ERROR: Cannot reach ${HEALTH_URL}"
  echo "Make sure your RunPod pod is running and the port is exposed."
  exit 1
fi
echo "Remote vLLM is healthy."

echo "========================================"
echo " Mon-Agent Remote vLLM Run"
echo " Model:   ${MODEL_NAME}"
echo " API:     ${VLLM_API_BASE}"
echo " Subset:  ${SUBSET} / ${SPLIT}"
echo " Slice:   ${SLICE:-all}"
echo " Results: ${RESULTS_DIR}"
echo "========================================"

# --- Run SWE-bench agent ---
export OPENAI_API_KEY="${RUNPOD_API_KEY:-EMPTY}"
export LITELLM_MODEL_REGISTRY_PATH="$REGISTRY"

CMD=(
  python -m mon_agent.runner
  --subset "$SUBSET"
  --split "$SPLIT"
  --output "$RESULTS_DIR"
  --workers "$WORKERS"
  -c "$CONFIG"
  -c "model.model_name=hosted_vllm/${MODEL_NAME}"
  -c "model.model_kwargs.api_base=${VLLM_API_BASE}"
)

[[ -n "$SLICE" ]] && CMD+=(--slice "$SLICE")

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"
deactivate || true

# --- Summary ---
echo ""
echo "========================================"
echo " Done! Results: ${RESULTS_DIR}"
echo "========================================"
echo ""
echo "Files produced:"
find "$RESULTS_DIR" -type f \( -name "*.json" -o -name "*.jsonl" \) | sort
echo ""

STEP_LOG=$(find "$RESULTS_DIR" -name "*.steps.jsonl" | head -1)
if [ -n "$STEP_LOG" ]; then
  echo "Step log: ${STEP_LOG}"
  echo "Total steps: $(wc -l < "$STEP_LOG")"
fi
