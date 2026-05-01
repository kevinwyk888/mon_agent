#!/usr/bin/env bash
#SBATCH --job-name=mon-vllm-smoke
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vllm-smoke-%j.out

set -euo pipefail

cleanup() {
  if [ -n "${VLLM_PID:-}" ]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

module load python/3.11.5 || true
module load singularity || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${PROJECT_DIR}/scripts/vllm_container_lib.sh"

MODEL="${MODEL_NAME:-google/gemma-4-31B-it}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
SCRATCH_BASE="/scratch/alkontar_root/alkontar0/rongbo"
NODE_TMP="/tmp/${USER}/mon_${SLURM_JOB_ID}"
RESULTS_DIR="${SCRATCH_BASE}/mon-agent/results/vllm_smoke_${SLURM_JOB_ID}"

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
         "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$RESULTS_DIR"

RUNTIME="$(detect_container_runtime)"
set_default_vllm_container_paths
ensure_vllm_container_image "$RUNTIME"
set_vllm_container_env

echo "========================================"
echo " VLLM Smoke Test — $(date)"
echo " Node:    $(hostname -s)"
echo " Model:   ${MODEL}"
echo " TP:      ${TP_SIZE}"
echo " Image:   ${VLLM_CONTAINER_SIF}"
echo " Results: ${RESULTS_DIR}"
echo "========================================"

echo "[1/3] Starting vLLM..."
"$RUNTIME" exec --nv --cleanenv \
  --bind "$HF_HOME:$HF_HOME,$TMPDIR:$TMPDIR,$XDG_CACHE_HOME:$XDG_CACHE_HOME,$CUDA_CACHE_PATH:$CUDA_CACHE_PATH${VLLM_CONTAINER_EXTRA_BINDS:+,$VLLM_CONTAINER_EXTRA_BINDS}" \
  "$VLLM_CONTAINER_SIF" \
  vllm serve "$MODEL" \
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
    > "${RESULTS_DIR}/vllm_server.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM health..."
for i in $(seq 1 1800); do
  if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "vLLM ready after ${i}s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM died before health check passed"
    tail -n 200 "${RESULTS_DIR}/vllm_server.log" || true
    exit 1
  fi
  sleep 1
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: vLLM did not start within 1800s"
  tail -n 200 "${RESULTS_DIR}/vllm_server.log" || true
  exit 1
fi

echo "[2/3] Running chat completion smoke test..."
CHAT_HTTP_CODE="$(
  curl -s -o "${RESULTS_DIR}/chat_response.json" -w "%{http_code}" \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"$MODEL"'",
      "messages": [{"role": "user", "content": "Reply with exactly two words."}],
      "max_tokens": 16,
      "temperature": 0.0
    }'
)"
echo "chat status: ${CHAT_HTTP_CODE}"
if [ "$CHAT_HTTP_CODE" != "200" ]; then
  cat "${RESULTS_DIR}/chat_response.json"
  exit 1
fi
python - "${RESULTS_DIR}/chat_response.json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
msg = data["choices"][0]["message"]["content"]
print("chat response:", msg)
PY

echo "[3/3] Running tool-choice API smoke test..."
TOOLS_HTTP_CODE="$(
  curl -s -o "${RESULTS_DIR}/tools_response.json" -w "%{http_code}" \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"$MODEL"'",
      "messages": [{"role": "user", "content": "Use the dummy tool to send the string hi."}],
      "max_tokens": 64,
      "temperature": 0.0,
      "tool_choice": "auto",
      "tools": [{
        "type": "function",
        "function": {
          "name": "dummy",
          "description": "Echoes a string.",
          "parameters": {
            "type": "object",
            "properties": {
              "text": {"type": "string"}
            },
            "required": ["text"]
          }
        }
      }]
    }'
)"
echo "tools status: ${TOOLS_HTTP_CODE}"
if [ "$TOOLS_HTTP_CODE" != "200" ]; then
  cat "${RESULTS_DIR}/tools_response.json"
  exit 1
fi
python - "${RESULTS_DIR}/tools_response.json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
msg = data["choices"][0]["message"]
tool_calls = msg.get("tool_calls", [])
print("tool call count:", len(tool_calls))
if tool_calls:
    print("first tool:", tool_calls[0]["function"]["name"])
PY

echo "Smoke test passed."
echo "Results: ${RESULTS_DIR}"
