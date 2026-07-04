#!/bin/bash
# SLURM batch script for the single-trajectory CUSUM rollback experiment.

#SBATCH --job-name=mon_retry
#SBATCH --account=alkontar0
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

echo "[$(date -Iseconds)] Job ${SLURM_JOB_ID:-local} on $(hostname)"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module load singularity 2>/dev/null || true

VENV="${VENV:-$HOME/.venvs/mon_agent_env}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export SCRATCH_BASE="${SCRATCH_BASE:-/scratch/alkontar_root/alkontar0/kevinwyk}"
export TMPDIR="${TMPDIR:-${SCRATCH_BASE}/tmp}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${SCRATCH_BASE}/sing_tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${SCRATCH_BASE}/singularity_home/cache}"
mkdir -p "$TMPDIR" "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

export SELECTION="${SELECTION:-results/retry_experiment/monitor/selection.json}"
export CONDITION="${CONDITION:-control}"   # control | intervention
export MONITOR_DIR="${MONITOR_DIR:-results/retry_experiment/monitor}"
export RUN_NAME="${RUN_NAME:-retry_${CONDITION}_${SLURM_JOB_ID:-local}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/mon_agent/${RUN_NAME}}"
export LOCAL_SUMMARY_DIR="${LOCAL_SUMMARY_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/results/${RUN_NAME}}"
export CONFIG="${CONFIG:-configs/swebench_lite.yaml}"
export SUBSET="${SUBSET:-verified}"
export SPLIT="${SPLIT:-test}"
export MODEL="${MODEL:-openai/deepseek-v4-flash}"
export WORKERS="${WORKERS:-4}"
export REPEATS="${REPEATS:-20}"
export STEP_LIMIT="${STEP_LIMIT:-64}"
export MAX_TOTAL_STEPS_MULTIPLIER="${MAX_TOTAL_STEPS_MULTIPLIER:-2.5}"
export ROLLBACK_FALLBACK_STEPS="${ROLLBACK_FALLBACK_STEPS:-8}"
export RUN_TEMPERATURE="${RUN_TEMPERATURE:-0.3}"
export RUN_TOP_P="${RUN_TOP_P:-1.0}"
export SEED_BASE="${SEED_BASE:-1234}"
export SIF_CACHE="${SIF_CACHE:-${SCRATCH_BASE}/swebench_sif}"
export EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-1800}"
export ALARM_MODEL_LIB="${ALARM_MODEL_LIB:-${SLURM_SUBMIT_DIR:-$PWD}/alarm_monitor/prefix_alarm_monitor_lib.py}"
export MSWEA_SIF_CACHE_DIR="${MSWEA_SIF_CACHE_DIR:-${SIF_CACHE}}"

if [[ -r "${HOME}/.dockerhub_creds" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.dockerhub_creds"
fi

if [[ "${CONDITION}" != "control" && "${CONDITION}" != "intervention" ]]; then
    echo "CONDITION must be control or intervention, got ${CONDITION}" >&2
    exit 2
fi
if [[ "${CONDITION}" == "intervention" && ! -d "${MONITOR_DIR}" ]]; then
    echo "MONITOR_DIR does not exist: ${MONITOR_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOCAL_SUMMARY_DIR}" "${SIF_CACHE}"

runner_args=(
  --selection "${SELECTION}"
  --condition "${CONDITION}"
  --output "${OUTPUT_DIR}"
  --subset "${SUBSET}"
  --split "${SPLIT}"
  --config "${CONFIG}"
  --model "${MODEL}"
  --workers "${WORKERS}"
  --repeats "${REPEATS}"
  --step-limit "${STEP_LIMIT}"
  --max-total-steps-multiplier "${MAX_TOTAL_STEPS_MULTIPLIER}"
  --rollback-fallback-steps "${ROLLBACK_FALLBACK_STEPS}"
  --temperature "${RUN_TEMPERATURE}"
  --top-p "${RUN_TOP_P}"
  --seed-base "${SEED_BASE}"
  --sif-cache "${SIF_CACHE}"
  --eval-timeout-s "${EVAL_TIMEOUT_S}"
)
if [[ "${CONDITION}" == "intervention" ]]; then
  runner_args+=(--monitor-dir "${MONITOR_DIR}")
fi
if [[ "${REDO_EXISTING:-0}" == "1" ]]; then
  runner_args+=(--redo-existing)
fi
if [[ "${EVALUATE_HARNESS:-1}" != "1" ]]; then
  runner_args+=(--no-evaluate-harness)
fi

echo "[config]"
echo "  CONDITION     = ${CONDITION}"
echo "  SELECTION     = ${SELECTION}"
echo "  MONITOR_DIR   = ${MONITOR_DIR}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "  REPEATS       = ${REPEATS}"
echo "  STEP_LIMIT    = ${STEP_LIMIT}"
echo "  TOTAL_LIMIT   = ${STEP_LIMIT} * ${MAX_TOTAL_STEPS_MULTIPLIER}"

python -m mon_agent.retry_runner "${runner_args[@]}"

cp "${OUTPUT_DIR}/summary.json" "${LOCAL_SUMMARY_DIR}/summary.json"
cp "${OUTPUT_DIR}/runs.jsonl" "${LOCAL_SUMMARY_DIR}/runs.jsonl"
cp "${OUTPUT_DIR}/selection.json" "${LOCAL_SUMMARY_DIR}/selection.json"
echo "[$(date -Iseconds)] Summary copied to ${LOCAL_SUMMARY_DIR}"