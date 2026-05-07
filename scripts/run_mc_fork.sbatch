#!/bin/bash
# SLURM batch script: run mon-agent MC-fork experiment + post-processing.
#
# Submit:
#     sbatch scripts/run_mc_fork.sbatch
#
# Override knobs at submit time, e.g.:
#     sbatch --export=ALL,SLICE=1:10,OUTPUT_DIR=results/mc_run_1 \
#            scripts/run_mc_fork.sbatch
#     sbatch --export=ALL,MC_SAMPLES=4,MC_TEMPERATURE=0.7,MC_FORK_EVERY=8 \
#            scripts/run_mc_fork.sbatch
#
# What this script does
# ---------------------
#   1. activates the project venv,
#   2. runs scripts/run_mc_fork.sh (the actual experiment),
#   3. computes y_global per instance,
#   4. merges main + all forks into <inst>.all_steps.jsonl with y_global.
#
# Required runtime context (provided by caller, not by this script):
#   - a reachable LiteLLM endpoint, e.g. exported MODEL_API_BASE pointing at a
#     vLLM server. Forward over an SSH tunnel or run vLLM in a sibling job.
#   - HuggingFace cache populated with the SWE-bench dataset.
#   - `module load singularity` on this cluster (loaded below).

#SBATCH --job-name=mon_mc
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=YOUR_UNIQNAME@umich.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "GPUs visible : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Workdir      : ${SLURM_SUBMIT_DIR:-$PWD}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load singularity 2>/dev/null || true

VENV="${VENV:-$HOME/.venvs/mon_agent_env}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Quiet HF tokenizers warning under threaded runs
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---------------------------------------------------------------------------
# Experiment knobs (mirror run_mc_fork.sh defaults; override via --export=ALL,...)
# ---------------------------------------------------------------------------
export OUTPUT_DIR="${OUTPUT_DIR:-results/mc_run_${SLURM_JOB_ID:-local}}"
export CONFIG="${CONFIG:-configs/swebench_lite.yaml}"
export SUBSET="${SUBSET:-lite}"
export SPLIT="${SPLIT:-dev}"
export SLICE="${SLICE:-1:100}"
export WORKERS="${WORKERS:-1}"

export MC_FORK_EVERY="${MC_FORK_EVERY:-5}"
export MC_SAMPLES="${MC_SAMPLES:-2}"
export MC_TEMPERATURE="${MC_TEMPERATURE:-0.3}"
export MC_TOP_P="${MC_TOP_P:-0.95}"
export MC_MAX_FORK_STEPS="${MC_MAX_FORK_STEPS:-30}"
export MC_MAX_FORK_STEPS_MIN="${MC_MAX_FORK_STEPS_MIN:-40}"
export MC_MAX_FORK_STEPS_TOTAL="${MC_MAX_FORK_STEPS_TOTAL:-60}"
export MC_FORK_COST_BUDGET="${MC_FORK_COST_BUDGET:-1.0}"
export MC_SNAPSHOT_CWD="${MC_SNAPSHOT_CWD:-/testbed}"

cat <<EOF
[config]
  OUTPUT_DIR              = ${OUTPUT_DIR}
  CONFIG                  = ${CONFIG}
  SUBSET / SPLIT / SLICE  = ${SUBSET} / ${SPLIT} / ${SLICE}
  WORKERS                 = ${WORKERS}
  MC_FORK_EVERY           = ${MC_FORK_EVERY}
  MC_SAMPLES              = ${MC_SAMPLES}
  MC_TEMPERATURE / TOP_P  = ${MC_TEMPERATURE} / ${MC_TOP_P}
  MC_MAX_FORK_STEPS (stat)= ${MC_MAX_FORK_STEPS}
  MC_MAX_FORK_STEPS_MIN   = ${MC_MAX_FORK_STEPS_MIN}
  MC_MAX_FORK_STEPS_TOTAL = ${MC_MAX_FORK_STEPS_TOTAL}
  MC_FORK_COST_BUDGET     = ${MC_FORK_COST_BUDGET}
  MC_SNAPSHOT_CWD         = ${MC_SNAPSHOT_CWD}
  MODEL_API_BASE          = ${MODEL_API_BASE:-<unset — relying on config-provided endpoint>}
EOF

# ---------------------------------------------------------------------------
# 1) Experiment
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] Stage 1/3: run_mc_fork.sh"
bash scripts/run_mc_fork.sh

# ---------------------------------------------------------------------------
# 2) Per-instance y_global CSV
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] Stage 2/3: compute_y_global"
python results/compute_y_global.py "${OUTPUT_DIR}" || \
    echo "[warn] compute_y_global failed; continuing"

# ---------------------------------------------------------------------------
# 3) Merge main + all forks into <inst>.all_steps.jsonl
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] Stage 3/3: merge_all_steps"
python results/merge_all_steps.py "${OUTPUT_DIR}" || \
    echo "[warn] merge_all_steps failed; continuing"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "Artifacts under ${OUTPUT_DIR}:"
find "${OUTPUT_DIR}" -maxdepth 2 -type f \
    \( -name '*.traj.json' -o -name '*.steps.jsonl' \
       -o -name '*.mc.jsonl' -o -name '*.y_global.csv' \
       -o -name '*.all_steps.jsonl' \) | sort

echo "[$(date -Iseconds)] Job ${SLURM_JOB_ID:-local} finished"
