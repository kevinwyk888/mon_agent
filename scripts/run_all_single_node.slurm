#!/usr/bin/env bash
#SBATCH --job-name=mon-all
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/mon-all-%j.out

# ============================================================
# Single-node job: vLLM server + SWE-bench agent on same node.
# No cross-node networking needed.
#
# Usage:
#   sbatch scripts/run_all_single_node.slurm
#   SLICE=0:5 sbatch scripts/run_all_single_node.slurm
# ============================================================

set -euo pipefail

cleanup() {
  if [ -n "${VLLM_PID:-}" ]; then
    pkill -P "$VLLM_PID" 2>/dev/null || true
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if command -v module >/dev/null 2>&1; then
  module load python/3.11.5 || true
  module load singularity || true
fi

AGENT_ENV_DIR="${MON_AGENT_VENV:-$HOME/.venvs/agent_env}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${PROJECT_DIR}/scripts/vllm_container_lib.sh"

# --- Model config ---
MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
PORT=8000
TP_SIZE="${SLURM_GPUS_ON_NODE:-2}"

# --- Storage strategy ---
# Persistent (scratch): model weights, singularity image layer cache
# Ephemeral (node /tmp): all runtime caches, sandboxes — auto-cleaned after job
SCRATCH_BASE="/scratch/alkontar_root/alkontar0/rongbo"
NODE_TMP="/tmp/${USER}/mon_${SLURM_JOB_ID}"

# Model weights on scratch (persistent, avoid re-downloading)
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf_cache/hub"

# Singularity image layer cache on scratch (persistent, reusable across jobs)
export SINGULARITY_CACHEDIR="${SCRATCH_BASE}/singularity_cache"

# Everything else on node-local /tmp (ephemeral, no quota cost)
export SINGULARITY_TMPDIR="${NODE_TMP}/singularity_tmp"
export TMPDIR="${NODE_TMP}/tmpdir"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export XDG_CACHE_HOME="${NODE_TMP}/xdg_cache"
export CUDA_CACHE_PATH="${NODE_TMP}/cuda_cache"
export TORCHINDUCTOR_CACHE_DIR="${NODE_TMP}/torch_cache"

set_container_runtime_paths
mkdir -p "$HF_HUB_CACHE" "$SINGULARITY_CACHEDIR" \
         "$SINGULARITY_TMPDIR" "$TMPDIR" \
         "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR"

RUNTIME="$(detect_container_runtime)"
set_default_vllm_container_paths
ensure_vllm_container_image "$RUNTIME"
set_vllm_container_env

# --- SWE-bench config ---
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-1}"
SLICE="${SLICE:-}"
CONFIG="${PROJECT_DIR}/configs/swebench_lite.yaml"
REGISTRY="${PROJECT_DIR}/configs/registry.json"
RESULTS_DIR="${SCRATCH_BASE}/mon-agent/results/${SUBSET}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo " Mon-Agent Single Node Run"
echo " Model:   ${MODEL_NAME} (TP=${TP_SIZE})"
echo " Subset:  ${SUBSET} / ${SPLIT}"
echo " Slice:   ${SLICE:-all}"
echo " Results: ${RESULTS_DIR}"
echo "========================================"

# ========== Step 1: Launch vLLM server in background ==========
echo "[1/2] Starting vLLM server..."
"$RUNTIME" exec --nv --cleanenv \
  --bind "$HF_HOME:$HF_HOME,$TMPDIR:$TMPDIR,$XDG_CACHE_HOME:$XDG_CACHE_HOME,$CUDA_CACHE_PATH:$CUDA_CACHE_PATH,$TORCHINDUCTOR_CACHE_DIR:$TORCHINDUCTOR_CACHE_DIR${VLLM_CONTAINER_EXTRA_BINDS:+,$VLLM_CONTAINER_EXTRA_BINDS}" \
  "$VLLM_CONTAINER_SIF" \
  vllm serve "$MODEL_NAME" \
    --download-dir "$HF_HUB_CACHE" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP_SIZE" \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --limit-mm-per-prompt '{"image":0,"audio":0}' \
    > "$RESULTS_DIR/vllm_server.log" 2>&1 &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to start (model loading from scratch takes ~5min)..."
for i in $(seq 1 2700); do
  if curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
    echo "vLLM server ready after ${i}s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM process died. Last 50 lines of log:"
    tail -50 "$RESULTS_DIR/vllm_server.log" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

if ! curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
  echo "ERROR: vLLM server did not start within 2700s"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

# ========== Step 2: Run SWE-bench agent ==========
echo "[2/2] Running SWE-bench evaluation..."

source "$AGENT_ENV_DIR/bin/activate"
export OPENAI_API_KEY="EMPTY"
export LITELLM_MODEL_REGISTRY_PATH="$REGISTRY"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

CMD=(
  python -m mon_agent.runner
  --subset "$SUBSET"
  --split "$SPLIT"
  --output "$RESULTS_DIR"
  --workers "$WORKERS"
  -c "$CONFIG"
  -c "model.model_kwargs.api_base=http://127.0.0.1:${PORT}/v1"
)

[[ -n "$SLICE" ]] && CMD+=(--slice "$SLICE")

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"
deactivate || true

# ========== Cleanup ==========
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "========================================"
echo " Done! Results: ${RESULTS_DIR}"
echo "========================================"
