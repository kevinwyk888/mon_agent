#!/usr/bin/env bash
#SBATCH --job-name=mon-vllm
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vllm-%j.out

# ============================================================
# Step 1: Deploy a vLLM server on a GPU node.
#
# Usage:
#   sbatch scripts/01_deploy_model.slurm
#   # or override model:
#   sbatch --export=ALL,MODEL_NAME=google/gemma-4-31B scripts/01_deploy_model.slurm
#
# After launch, check logs/vllm-<jobid>.out for the server URL.
# Then pass that URL to step 2.
# ============================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load singularity || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${PROJECT_DIR}/scripts/vllm_container_lib.sh"

# --- Configuration ---
MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE:-2}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# --- Use scratch for model cache ---
SCRATCH_DIR="/scratch/alkontar_root/alkontar0/rongbo"
HF_CACHE="${SCRATCH_DIR}/hf_cache"
NODE_TMP="/tmp/${USER}/mon_${SLURM_JOB_ID}"
mkdir -p "$HF_CACHE/hub"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export TMPDIR="${NODE_TMP}/tmpdir"
export XDG_CACHE_HOME="${NODE_TMP}/xdg_cache"
export CUDA_CACHE_PATH="${NODE_TMP}/cuda_cache"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
set_container_runtime_paths
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

RUNTIME="$(detect_container_runtime)"
SCRATCH_BASE="$SCRATCH_DIR"
set_default_vllm_container_paths
ensure_vllm_container_image "$RUNTIME"
set_vllm_container_env

HOSTNAME_SHORT="$(hostname -s)"

# --- Write hostfile so the client job knows where to connect ---
HOSTFILE="${PROJECT_DIR}/logs/vllm_host_${SLURM_JOB_ID}.txt"
echo "http://${HOSTNAME_SHORT}:${PORT}/v1" > "$HOSTFILE"

echo "========================================"
echo " vLLM Server"
echo " Host:  ${HOSTNAME_SHORT}"
echo " Port:  ${PORT}"
echo " Model: ${MODEL_NAME}"
echo " TP:    ${TP_SIZE}"
echo " Image: ${VLLM_CONTAINER_SIF}"
echo " URL:   http://${HOSTNAME_SHORT}:${PORT}/v1"
echo " Hostfile: ${HOSTFILE}"
echo "========================================"

"$RUNTIME" exec --nv \
  --bind "$HF_HOME:$HF_HOME,$TMPDIR:$TMPDIR,$XDG_CACHE_HOME:$XDG_CACHE_HOME,$CUDA_CACHE_PATH:$CUDA_CACHE_PATH" \
  "$VLLM_CONTAINER_SIF" \
  vllm serve "$MODEL_NAME" \
    --download-dir "$HF_CACHE/hub" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP_SIZE" \
    --disable-custom-all-reduce \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --limit-mm-per-prompt image=0,audio=0
