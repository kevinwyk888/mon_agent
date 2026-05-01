#!/usr/bin/env bash
#SBATCH --job-name=mon-dl
#SBATCH --account=alkontar0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download-%j.out

# ============================================================
# Download model weights to scratch (no GPU needed).
#
# Usage:
#   sbatch scripts/download_model.slurm
#   MODEL_NAME=other/model sbatch scripts/download_model.slurm
# ============================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load python/3.11.5 || true
fi

ENV_DIR="${MON_AGENT_VENV:-$HOME/.venvs/agent_env}"
source "$ENV_DIR/bin/activate"

MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
SCRATCH_DIR="/scratch/alkontar_root/alkontar0/rongbo"
HF_CACHE="${SCRATCH_DIR}/hf_cache"
mkdir -p "$HF_CACHE/hub"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"

echo "========================================"
echo " Downloading: ${MODEL_NAME}"
echo " Cache dir:   ${HF_CACHE}/hub"
echo " Started:     $(date)"
echo "========================================"

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MODEL_NAME}',
    cache_dir='${HF_CACHE}/hub',
    resume_download=True,
)
print('Download complete!')
"

echo "========================================"
echo " Done: $(date)"
echo "========================================"
du -sh "${HF_CACHE}/hub/models--${MODEL_NAME//\//-}/"
