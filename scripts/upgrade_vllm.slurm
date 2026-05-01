#!/usr/bin/env bash
#SBATCH --job-name=mon-upgrade
#SBATCH --account=alkontar0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/upgrade-%j.out

set -euo pipefail

module load singularity || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${PROJECT_DIR}/scripts/vllm_container_lib.sh"

SCRATCH_BASE="/scratch/alkontar_root/alkontar0/rongbo"
NODE_TMP="/tmp/${USER}/mon_${SLURM_JOB_ID}"
RUNTIME="$(detect_container_runtime)"

export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf_cache/hub"
export TMPDIR="${NODE_TMP}/tmpdir"
export XDG_CACHE_HOME="${NODE_TMP}/xdg_cache"
export CUDA_CACHE_PATH="${NODE_TMP}/cuda_cache"
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

set_container_runtime_paths
mkdir -p "$HF_HUB_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
set_default_vllm_container_paths
ensure_vllm_container_image "$RUNTIME"
set_vllm_container_env

echo "=== Container Prep ==="
echo "Runtime: $RUNTIME"
echo "Image ref: $VLLM_CONTAINER_REF"
echo "SIF path: $VLLM_CONTAINER_SIF"
echo "HF cache: $HF_HUB_CACHE"
echo "Singularity cache: $SINGULARITY_CACHEDIR"
echo "Singularity tmp: $SINGULARITY_TMPDIR"

echo ""
echo "=== Verify container contents ==="
"$RUNTIME" exec "$VLLM_CONTAINER_SIF" python3 --version
"$RUNTIME" exec "$VLLM_CONTAINER_SIF" python3 -c "import vllm; print(f'vllm: {vllm.__version__}')"
"$RUNTIME" exec "$VLLM_CONTAINER_SIF" python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"

echo ""
echo "=== Verify Gemma 4 support and parser availability ==="
"$RUNTIME" exec \
  --bind "$HF_HOME:$HF_HOME,$TMPDIR:$TMPDIR,$XDG_CACHE_HOME:$XDG_CACHE_HOME" \
  "$VLLM_CONTAINER_SIF" \
  python3 - <<'PY'
from transformers import AutoConfig
import subprocess

cfg = AutoConfig.from_pretrained("google/gemma-4-31B-it")
print(f"transformers model_type: {cfg.model_type}")
help_text = subprocess.check_output(["vllm", "serve", "--help"], text=True)
if "gemma4" not in help_text:
    raise SystemExit("gemma4 parser not exposed by this vLLM container")
print("gemma4 parser is available in vllm serve --help")
PY

echo "Done!"
