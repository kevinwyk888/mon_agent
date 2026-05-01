#!/usr/bin/env bash
#SBATCH --job-name=mon-e2e
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e2e-%j.out

# ============================================================
# End-to-end test: vLLM + MonitoringAgent + SWE-bench Lite
# Uses Gemma-4-31B-it (TP=2) on 1 instance.
# ============================================================

set -euo pipefail

cleanup() {
  if [ -n "${VLLM_PID:-}" ]; then
    # Kill entire process tree (container children included)
    pkill -P "$VLLM_PID" 2>/dev/null || true
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

module load python/3.11.5 || true
module load singularity || true

AGENT_ENV_DIR="${MON_AGENT_VENV:-$HOME/.venvs/agent_env}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/vllm_container_lib.sh"

# --- Model ---
MODEL="google/gemma-4-31B-it"
PORT=8000

# --- Storage ---
SCRATCH_BASE="/scratch/alkontar_root/alkontar0/rongbo"
NODE_TMP="/tmp/${USER}/mon_${SLURM_JOB_ID}"

export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf_cache/hub"
export SINGULARITY_CACHEDIR="${SCRATCH_BASE}/singularity_cache"
export SINGULARITY_TMPDIR="${NODE_TMP}/singularity_tmp"
export TMPDIR="${NODE_TMP}/tmpdir"
export XDG_CACHE_HOME="${NODE_TMP}/xdg_cache"
export CUDA_CACHE_PATH="${NODE_TMP}/cuda_cache"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

set_container_runtime_paths
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$TMPDIR" \
         "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

RUNTIME="$(detect_container_runtime)"
set_default_vllm_container_paths
ensure_vllm_container_image "$RUNTIME"
set_vllm_container_env

RESULTS_DIR="${SCRATCH_BASE}/mon-agent/results/e2e_test_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo " E2E Test — $(date)"
echo " Node: $(hostname -s)"
echo " Model: ${MODEL}"
echo " Results: ${RESULTS_DIR}"
echo "========================================"

# ========== Step 1: Launch vLLM ==========
echo "[1/2] Starting vLLM..."
"$RUNTIME" exec --nv --cleanenv \
  --bind "$HF_HOME:$HF_HOME,$TMPDIR:$TMPDIR,$XDG_CACHE_HOME:$XDG_CACHE_HOME,$CUDA_CACHE_PATH:$CUDA_CACHE_PATH${VLLM_CONTAINER_EXTRA_BINDS:+,$VLLM_CONTAINER_EXTRA_BINDS}" \
  "$VLLM_CONTAINER_SIF" \
  vllm serve "$MODEL" \
    --download-dir "$HF_HUB_CACHE" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    > "${RESULTS_DIR}/vllm_server.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM (loading from scratch, ~5min)..."
for i in $(seq 1 2700); do
  if curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
    echo "vLLM ready after ${i}s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM died. Last 50 lines of vllm_server.log:"
    tail -50 "${RESULTS_DIR}/vllm_server.log" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

if ! curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
  echo "ERROR: vLLM did not start within 2700s"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

# ========== Step 2: Run 1 SWE-bench instance ==========
echo "[2/2] Running MonitoringAgent on 1 SWE-bench Lite instance..."

source "$AGENT_ENV_DIR/bin/activate"
export OPENAI_API_KEY="EMPTY"
export LITELLM_MODEL_REGISTRY_PATH="${PROJECT_DIR}/configs/registry.json"

python -m mon_agent.runner \
  --subset lite \
  --split test \
  --slice 0:1 \
  --output "$RESULTS_DIR" \
  --workers 1 \
  -c "${PROJECT_DIR}/configs/swebench_lite.yaml" \
  -c "model.model_name=hosted_vllm/${MODEL}" \
  -c "model.model_kwargs.api_base=http://127.0.0.1:${PORT}/v1" \
  -c "agent.step_limit=60" \
  2>&1 | tee "$RESULTS_DIR/run.log"
deactivate || true

# ========== Check outputs ==========
echo ""
echo "========================================"
echo " Results at: ${RESULTS_DIR}"
echo "========================================"
echo ""
echo "Files produced:"
find "$RESULTS_DIR" -type f -name "*.json" -o -name "*.jsonl" | sort
echo ""

# Show step logs if any
STEP_LOG=$(find "$RESULTS_DIR" -name "*.steps.jsonl" | head -1)
if [ -n "$STEP_LOG" ]; then
  echo "Step log preview (${STEP_LOG}):"
  head -5 "$STEP_LOG"
  echo "..."
  echo "Total steps: $(wc -l < "$STEP_LOG")"
fi

# ========== Cleanup ==========
echo ""
echo "Stopping vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "Done!"
