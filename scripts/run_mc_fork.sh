#!/usr/bin/env bash
# Run mon-agent with Monte-Carlo prefix evaluation (Phase-2 of feasible_method.md).
#
# What this does
# --------------
# At every K main-trajectory steps the runner:
#   1. snapshots /testbed via `git commit-tree` inside the Singularity sandbox,
#   2. spawns M fork rollouts that share the env but use a higher-temperature
#      copy of the model so they actually diverge,
#   3. resets the env after each rollout (and again at the end),
#   4. records Y_k^MC = (#successful forks) / M into:
#        results/<run>/<instance_id>/<instance_id>.mc.jsonl
#        results/<run>/<instance_id>/<instance_id>.steps.jsonl  (y_mc field)
#        results/<run>/<instance_id>/<instance_id>.traj.json    (mc_results)
#
# Cost warning
# ------------
# A single instance with step_limit=60, --mc-fork-every 5, --mc-samples 4,
# --mc-max-fork-steps 20 can multiply LLM calls by ~10-15x. Start small:
# slice 1-2 instances and a low M before scaling up.
#
# Usage
# -----
#   bash scripts/run_mc_fork.sh                    # uses defaults below
#   OUTPUT_DIR=results/mc_run_1 bash scripts/run_mc_fork.sh
#
# Required env (caller decides how to provide):
#   - A reachable LiteLLM-compatible endpoint (e.g. local vLLM on $MODEL_API_BASE)
#   - SWE-bench dataset access (HuggingFace cache)
#   - singularity executable on PATH

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/mc_smoke}"
CONFIG="${CONFIG:-configs/swebench_lite.yaml}"
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-dev}"
SLICE="${SLICE:-1:2}"               # only run the first instance by default
WORKERS="${WORKERS:-1}"            # MC forks are sequential per-instance; cross-instance parallelism only

# MC knobs
MC_FORK_EVERY="${MC_FORK_EVERY:-5}"
MC_SAMPLES="${MC_SAMPLES:-2}"
MC_TEMPERATURE="${MC_TEMPERATURE:-0.3}"
MC_TOP_P="${MC_TOP_P:-0.95}"
MC_MAX_FORK_STEPS="${MC_MAX_FORK_STEPS:-30}"
# Dynamic cap: at snapshot step k, fork runs at most max(MIN, TOTAL - k) steps.
# Set MC_MAX_FORK_STEPS_TOTAL=0 to disable and fall back to MC_MAX_FORK_STEPS.
MC_MAX_FORK_STEPS_MIN="${MC_MAX_FORK_STEPS_MIN:-40}"
MC_MAX_FORK_STEPS_TOTAL="${MC_MAX_FORK_STEPS_TOTAL:-60}"
MC_FORK_COST_BUDGET="${MC_FORK_COST_BUDGET:-1.0}"
MC_SNAPSHOT_CWD="${MC_SNAPSHOT_CWD:-/testbed}"

mkdir -p "${OUTPUT_DIR}"

mon-agent-runner \
  -c "${CONFIG}" \
  --subset "${SUBSET}" \
  --split "${SPLIT}" \
  --slice "${SLICE}" \
  -w "${WORKERS}" \
  -o "${OUTPUT_DIR}" \
  --mc-fork \
  --mc-fork-every "${MC_FORK_EVERY}" \
  --mc-samples "${MC_SAMPLES}" \
  --mc-temperature "${MC_TEMPERATURE}" \
  --mc-top-p "${MC_TOP_P}" \
  --mc-max-fork-steps "${MC_MAX_FORK_STEPS}" \
  --mc-max-fork-steps-min "${MC_MAX_FORK_STEPS_MIN}" \
  --mc-max-fork-steps-total "${MC_MAX_FORK_STEPS_TOTAL}" \
  --mc-fork-cost-budget "${MC_FORK_COST_BUDGET}" \
  --mc-snapshot-cwd "${MC_SNAPSHOT_CWD}"

echo
echo "Done. Inspect results:"
echo "  jq . ${OUTPUT_DIR}/*/$(basename ${OUTPUT_DIR})*/*.mc.jsonl 2>/dev/null || find ${OUTPUT_DIR} -name '*.mc.jsonl'"
