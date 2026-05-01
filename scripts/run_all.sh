#!/usr/bin/env bash
# Convenience script: submit vLLM server, wait for it, then submit SWE-bench job.
#
# Usage:
#   bash scripts/run_all.sh
#   MODEL_NAME=google/gemma-4-31B SLICE=0:5 bash scripts/run_all.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

echo "[1/3] Submitting vLLM server job..."
VLLM_JOB_ID=$(sbatch --parsable scripts/01_deploy_model.slurm)
echo "  vLLM Job ID: $VLLM_JOB_ID"

echo "[2/3] Waiting for vLLM server to start..."
HOSTFILE="logs/vllm_host_${VLLM_JOB_ID}.txt"
TIMEOUT=300  # 5 minutes max wait
ELAPSED=0
while [[ ! -f "$HOSTFILE" ]] && (( ELAPSED < TIMEOUT )); do
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  echo "  Waiting... (${ELAPSED}s)"
done

if [[ ! -f "$HOSTFILE" ]]; then
  echo "ERROR: vLLM server did not start within ${TIMEOUT}s"
  echo "Check: squeue -j $VLLM_JOB_ID"
  exit 1
fi

VLLM_URL="$(cat "$HOSTFILE")"
echo "  vLLM URL: $VLLM_URL"

# Wait a bit more for the model to actually load
echo "  Waiting 60s for model to fully load..."
sleep 60

echo "[3/3] Submitting SWE-bench job..."
EXTRA_EXPORTS="VLLM_JOB_ID=${VLLM_JOB_ID}"
[[ -n "${SLICE:-}" ]] && EXTRA_EXPORTS="${EXTRA_EXPORTS},SLICE=${SLICE}"
[[ -n "${WORKERS:-}" ]] && EXTRA_EXPORTS="${EXTRA_EXPORTS},WORKERS=${WORKERS}"

BENCH_JOB_ID=$(sbatch --parsable --export=ALL,${EXTRA_EXPORTS} scripts/02_run_swebench.slurm)
echo "  SWE-bench Job ID: $BENCH_JOB_ID"

echo ""
echo "========================================"
echo " Jobs submitted!"
echo " vLLM server:  $VLLM_JOB_ID"
echo " SWE-bench:    $BENCH_JOB_ID"
echo ""
echo " Monitor:"
echo "   squeue -u \$USER"
echo "   tail -f logs/vllm-${VLLM_JOB_ID}.out"
echo "   tail -f logs/swebench-${BENCH_JOB_ID}.out"
echo ""
echo " Cancel all:"
echo "   scancel $VLLM_JOB_ID $BENCH_JOB_ID"
echo "========================================"
