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
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=144:00:00
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

# Singularity sandbox build unpacks ~5-10 GB per image. /tmp on compute nodes
# is small (~16 GB) and shared across workers — point all temp/cache dirs to
# scratch (~10 TB) to avoid "No space left on device" during sandbox build.
export TMPDIR="${TMPDIR:-${SCRATCH_BASE}/tmp}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${SCRATCH_BASE}/sing_tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${SCRATCH_BASE}/singularity_home/cache}"
mkdir -p "$TMPDIR" "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
export RUN_NAME="${RUN_NAME:-tree_run_${SLURM_JOB_ID:-local}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/mon_agent/${RUN_NAME}}"
export CSV_OUT_DIR="${CSV_OUT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/results/${RUN_NAME}}"
export CONFIG="${CONFIG:-configs/swebench_lite.yaml}"
# SUBSET selects the SWE-bench dataset. Aliases (resolved by runner):
#   lite      = princeton-nlp/SWE-Bench_Lite      (300 short-patch tasks)
#   verified  = princeton-nlp/SWE-Bench_Verified  (500 human-vetted tasks)
#   full      = princeton-nlp/SWE-Bench           (2294 tasks)
export SUBSET="${SUBSET:-verified}"
export SPLIT="${SPLIT:-test}"
export SLICE="${SLICE:-0:500}"
# Override the model defined in CONFIG. Common values:
#   openai/deepseek-v4-pro    (strong, default)
#   openai/deepseek-v4-flash  (lightweight distilled, more diversity)
export MODEL="${MODEL:-openai/deepseek-v4-flash}"
# Default WORKERS=4: each worker may unpack a ~5-10 GB Singularity sandbox
# concurrently; 16 workers easily exhaust /tmp and GPFS metadata. Override
# only if SINGULARITY_TMPDIR is on a fast, large filesystem.
export WORKERS="${WORKERS:-8}"
export SIF_CACHE="${SIF_CACHE:-${SCRATCH_BASE}/swebench_sif}"

export TREE_FORK_EVERY="${TREE_FORK_EVERY:-8}"
export TREE_STEP_BUDGET="${TREE_STEP_BUDGET:-64}"
export TREE_FORK_COST_BUDGET="${TREE_FORK_COST_BUDGET:-5.0}"
export TREE_SNAPSHOT_CWD="${TREE_SNAPSHOT_CWD:-/testbed}"
export TREE_TEMPERATURE_LEFT="${TREE_TEMPERATURE_LEFT:-0.3}"
export TREE_TEMPERATURE_RIGHT="${TREE_TEMPERATURE_RIGHT:-0.3}"

export EVAL_HARNESS="${EVAL_HARNESS:-1}"
export EVAL_BACKEND="${EVAL_BACKEND:-singularity}"
export EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-1800}"

# Docker Hub creds: NEVER hard-code the PAT here (this script is git-tracked).
# Put `export DOCKERHUB_USER=...` and `export DOCKERHUB_TOKEN=...` in
# ~/.dockerhub_creds (chmod 600, gitignored).
if [[ -r "${HOME}/.dockerhub_creds" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.dockerhub_creds"
fi

# ---------------------------------------------------------------------------
# SIF cache (used by BOTH the agent runtime container AND the harness evaluator)
# ---------------------------------------------------------------------------
# Setting MSWEA_SIF_CACHE_DIR makes mon_agent's runner prefetch each instance's
# .sif into ${SIF_CACHE} once and reuse it on subsequent jobs — this avoids
# anonymous Docker Hub rate-limit failures (TOOMANYREQUESTS, 100 pulls/6h/IP)
# that previously killed ~70% of instances mid-batch.
export MSWEA_SIF_CACHE_DIR="${MSWEA_SIF_CACHE_DIR:-${SIF_CACHE}}"

# ---------------------------------------------------------------------------
# Optional: authenticated Docker Hub pulls for first-time SIF builds.
# Anonymous pulls are limited to 100/6h per IP; authenticated free accounts get
# 200/6h. To enable, set DOCKERHUB_USER + DOCKERHUB_TOKEN (a PAT) in your
# environment (NOT in this script) before sbatch, e.g. via ~/.config/mini-swe-agent/.env
# or `sbatch --export=ALL,DOCKERHUB_USER=...,DOCKERHUB_TOKEN=...`.
if [[ -n "${DOCKERHUB_USER:-}" && -n "${DOCKERHUB_TOKEN:-}" ]]; then
    echo "[$(date -Iseconds)] Logging in to docker.io as ${DOCKERHUB_USER}"
    # Singularity 4.x: OCI registries use `registry login` (not `remote login`,
    # which is reserved for Sylabs-style remotes). Try both for back-compat.
    _login_ok=0
    if printf '%s' "${DOCKERHUB_TOKEN}" | singularity registry login \
            --username "${DOCKERHUB_USER}" --password-stdin docker://docker.io 2>&1; then
        _login_ok=1
    elif printf '%s' "${DOCKERHUB_TOKEN}" | singularity remote login \
            --username "${DOCKERHUB_USER}" --password-stdin docker://docker.io 2>&1; then
        _login_ok=1
    fi
    if (( _login_ok )); then
        echo "[$(date -Iseconds)] docker.io login OK"
    else
        echo "[$(date -Iseconds)] WARNING: docker.io login failed; continuing anonymously"
    fi
    unset _login_ok
else
    echo "[$(date -Iseconds)] No DOCKERHUB_USER/DOCKERHUB_TOKEN set — using anonymous Docker Hub pulls (rate-limited)"
fi

cat <<EOF
[config]
  OUTPUT_DIR              = ${OUTPUT_DIR}
  CSV_OUT_DIR             = ${CSV_OUT_DIR}
  CONFIG                  = ${CONFIG}
  MODEL                   = ${MODEL}
  SUBSET / SPLIT / SLICE  = ${SUBSET} / ${SPLIT} / ${SLICE}
  WORKERS                 = ${WORKERS}
  TREE_FORK_EVERY         = ${TREE_FORK_EVERY}
  TREE_STEP_BUDGET        = ${TREE_STEP_BUDGET}
  TREE_FORK_COST_BUDGET   = ${TREE_FORK_COST_BUDGET}
  TREE_SNAPSHOT_CWD       = ${TREE_SNAPSHOT_CWD}
  TREE_TEMPERATURE_LEFT   = ${TREE_TEMPERATURE_LEFT}
  TREE_TEMPERATURE_RIGHT  = ${TREE_TEMPERATURE_RIGHT}
  EVAL_HARNESS            = ${EVAL_HARNESS}
  EVAL_BACKEND            = ${EVAL_BACKEND}
  SIF_CACHE               = ${SIF_CACHE}
  MSWEA_SIF_CACHE_DIR     = ${MSWEA_SIF_CACHE_DIR}
  DOCKERHUB_USER          = ${DOCKERHUB_USER:-<unset, using anonymous pulls>}
EOF

# ---------------------------------------------------------------------------
# Run experiment + CSV export (run_mc_fork.sh handles both stages)
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] Stage: run_mc_fork.sh"
bash scripts/run_mc_fork.sh

echo "[$(date -Iseconds)] Job ${SLURM_JOB_ID:-local} finished"
