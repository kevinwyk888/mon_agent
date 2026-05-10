#!/bin/bash
# SLURM batch script: run mon-agent tree search + harness eval + CSV export.
#
# Submit (defaults: results/tree_run_<jobid>, slice 0:25, lite/test):
#     sbatch scripts/run_mc_fork.sbatch
#
# Override knobs at submit time:
#     sbatch --export=ALL,SLICE=0:300,SPLIT=test,OUTPUT_DIR=results/full_lite_test \
#            scripts/run_mc_fork.sbatch
#     sbatch --export=ALL,TREE_FORK_EVERY=15,TREE_STEP_BUDGET=90 \
#            scripts/run_mc_fork.sbatch
#
# What this script does
# ---------------------
#   1. activates the project venv,
#   2. runs scripts/run_mc_fork.sh
#      (binary tree search + Singularity-backed SWE-bench harness eval).
#      The runner emits <inst>.steps.csv into CSV_OUT_DIR online, one per
#      instance as it finishes — no separate post-processing step.
#
# Required runtime context (provided by caller, not by this script):
#   - HuggingFace cache populated with the SWE-bench dataset.
#   - Model API key (e.g. DEEPSEEK_API_KEY) in the environment that LiteLLM
#     reads, or already present in ~/.config/mini-swe-agent/.env.
#   - Module ``singularity`` (loaded below) — this also implies first-time
#     instances will pull a ~1GB SWE-bench image into ${SIF_CACHE}.
#
# Resource sizing
# ---------------
# The agent itself does no GPU work (it calls a remote LLM API). GPUs are
# requested only because the cluster's spgpu partition requires them; tune
# down if your account allows. Singularity exec for SWE-bench evaluation is
# CPU-bound (~20-60s/leaf). Increase --time for large slices.

#SBATCH --job-name=mon_tree
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
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
# Some clusters require an explicit version:
# module load singularity/4.3.4

VENV="${VENV:-$HOME/.venvs/mon_agent_env}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Quiet HF tokenizers warning under threaded runs
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---------------------------------------------------------------------------
# Experiment knobs (override via --export=ALL,KEY=VAL,...)
# ---------------------------------------------------------------------------
# Heavy outputs go to scratch (~10TB). Per-instance CSVs land under
# ${PWD}/results/${RUN_NAME}/ (handled by run_mc_fork.sh).
export SCRATCH_BASE="${SCRATCH_BASE:-/scratch/alkontar_root/alkontar0/kevinwyk}"
export RUN_NAME="${RUN_NAME:-tree_run_${SLURM_JOB_ID:-local}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/mon_agent/${RUN_NAME}}"
export CSV_OUT_DIR="${CSV_OUT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/results/${RUN_NAME}}"
export CONFIG="${CONFIG:-configs/swebench_lite.yaml}"
export SUBSET="${SUBSET:-lite}"
export SPLIT="${SPLIT:-test}"
export SLICE="${SLICE:-0:25}"
export WORKERS="${WORKERS:-16}"
export SIF_CACHE="${SIF_CACHE:-${SCRATCH_BASE}/swebench_sif}"

export TREE_FORK_EVERY="${TREE_FORK_EVERY:-10}"
export TREE_STEP_BUDGET="${TREE_STEP_BUDGET:-60}"
export TREE_FORK_COST_BUDGET="${TREE_FORK_COST_BUDGET:-5.0}"
export TREE_SNAPSHOT_CWD="${TREE_SNAPSHOT_CWD:-/testbed}"

export EVAL_HARNESS="${EVAL_HARNESS:-1}"
export EVAL_BACKEND="${EVAL_BACKEND:-singularity}"
export EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-1800}"

cat <<EOF
[config]
  OUTPUT_DIR              = ${OUTPUT_DIR}
  CSV_OUT_DIR             = ${CSV_OUT_DIR}
  CONFIG                  = ${CONFIG}
  SUBSET / SPLIT / SLICE  = ${SUBSET} / ${SPLIT} / ${SLICE}
  WORKERS                 = ${WORKERS}
  TREE_FORK_EVERY         = ${TREE_FORK_EVERY}
  TREE_STEP_BUDGET        = ${TREE_STEP_BUDGET}
  TREE_FORK_COST_BUDGET   = ${TREE_FORK_COST_BUDGET}
  TREE_SNAPSHOT_CWD       = ${TREE_SNAPSHOT_CWD}
  EVAL_HARNESS            = ${EVAL_HARNESS}
  EVAL_BACKEND            = ${EVAL_BACKEND}
  SIF_CACHE               = ${SIF_CACHE}
EOF

# ---------------------------------------------------------------------------
# Run experiment + CSV export (run_mc_fork.sh handles both stages)
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] Stage: run_mc_fork.sh"
bash scripts/run_mc_fork.sh

echo "[$(date -Iseconds)] Job ${SLURM_JOB_ID:-local} finished"
