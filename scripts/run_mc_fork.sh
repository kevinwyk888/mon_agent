#!/usr/bin/env bash
# Run mon-agent tree-search on SWE-bench + post-process to CSV.
#
# What this does
# --------------
# 1. Drives a binary-tree search (split every K steps, branching=2, fixed
#    left-temp=0.3 / right-temp=0.3 in the current pilot) on each instance,
#    capped at
#    --tree-step-budget steps along any root->leaf path.
# 2. At each leaf evaluates the produced patch with the *real* SWE-bench
#    harness, using the Singularity backend (no Docker required). The
#    instance image is pulled once into ``--tree-eval-sif-cache`` and reused
#    across runs.
# 3. The runner emits <inst>.steps.csv into CSV_OUT_DIR online — one CSV per
#    instance, as soon as that instance's tree finishes. No batch
#    post-processing step is needed.
#
# Usage
# -----
#   bash scripts/run_mc_fork.sh                          # uses defaults below
#   OUTPUT_DIR=results/run1 SLICE=1:5 bash scripts/run_mc_fork.sh
#
# Required runtime context
# ------------------------
#   - ``module load singularity`` (real binary in /opt/singularity/<ver>/bin)
#   - ``source ~/.venvs/mon_agent_env/bin/activate``
#   - LiteLLM-compatible endpoint configured via the YAML (e.g. DeepSeek)
#   - HuggingFace cache populated for the SWE-bench dataset
#
# First call for a new instance does ``singularity pull`` (~1GB, 5-10 min);
# subsequent calls reuse the cached .sif.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Knobs (override via env)                                                    #
# --------------------------------------------------------------------------- #
# Heavy outputs (trajectories, tree.json, preds, .sif images) live on scratch
# (~10TB on Great Lakes). Lightweight per-instance CSVs are written into the
# project's local ./results/<run_name>/ directory at the end so they are easy
# to inspect / commit / sync.
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/alkontar_root/alkontar0/kevinwyk}"
RUN_NAME="${RUN_NAME:-tree_run}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/mon_agent/${RUN_NAME}}"
CSV_OUT_DIR="${CSV_OUT_DIR:-${PWD}/results/${RUN_NAME}}"
CONFIG="${CONFIG:-configs/swebench_lite.yaml}"
# SUBSET selects the SWE-bench dataset: lite | verified | full
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-0:25}"              # first 25 test-split instances by default
WORKERS="${WORKERS:-4}"             # cross-instance concurrency
SIF_CACHE="${SIF_CACHE:-${SCRATCH_BASE}/swebench_sif}"
# Override CONFIG's model.model_name. Common values:
#   openai/deepseek-v4-pro    (strong, default)
#   openai/deepseek-v4-flash  (lightweight distilled, more diversity)
MODEL="${MODEL:-openai/deepseek-v4-pro}"

# Tree-search knobs
TREE_FORK_EVERY="${TREE_FORK_EVERY:-5}"
TREE_STEP_BUDGET="${TREE_STEP_BUDGET:-60}"
TREE_FORK_COST_BUDGET="${TREE_FORK_COST_BUDGET:-5.0}"
TREE_SNAPSHOT_CWD="${TREE_SNAPSHOT_CWD:-/testbed}"
TREE_TEMPERATURE_LEFT="${TREE_TEMPERATURE_LEFT:-0.2}"
TREE_TEMPERATURE_RIGHT="${TREE_TEMPERATURE_RIGHT:-0.6}"

# Harness eval knobs
EVAL_HARNESS="${EVAL_HARNESS:-1}"   # 1 = run real SWE-bench harness on each leaf
EVAL_BACKEND="${EVAL_BACKEND:-singularity}"   # 'docker' or 'singularity'
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-1800}"

# --------------------------------------------------------------------------- #
# Sanity checks                                                               #
# --------------------------------------------------------------------------- #
if ! command -v singularity >/dev/null 2>&1 \
   || ! file "$(command -v singularity)" 2>/dev/null | grep -q ELF; then
  cat >&2 <<'MSG'
[run_mc_fork] singularity not on PATH (or PATH points at a text stub).
              Run:  module load singularity
              and verify:  which singularity  ->  /opt/singularity/.../bin/singularity
MSG
  exit 1
fi

if ! command -v mon-agent-runner >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[run_mc_fork] mon-agent-runner not on PATH.
              Run:  source ~/.venvs/mon_agent_env/bin/activate
MSG
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${SIF_CACHE}" "${CSV_OUT_DIR}"

cat <<EOF
[run_mc_fork] config
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
EOF

# --------------------------------------------------------------------------- #
# 1) Run tree search                                                          #
# --------------------------------------------------------------------------- #
runner_args=(
  -c "${CONFIG}"
  --model "${MODEL}"
  --subset "${SUBSET}"
  --split "${SPLIT}"
  --slice "${SLICE}"
  -w "${WORKERS}"
  -o "${OUTPUT_DIR}"
  --csv-out-dir "${CSV_OUT_DIR}"
  --tree-search
  --tree-fork-every "${TREE_FORK_EVERY}"
  --tree-step-budget "${TREE_STEP_BUDGET}"
  --tree-fork-cost-budget "${TREE_FORK_COST_BUDGET}"
  --tree-snapshot-cwd "${TREE_SNAPSHOT_CWD}"
  --tree-temperature-left "${TREE_TEMPERATURE_LEFT}"
  --tree-temperature-right "${TREE_TEMPERATURE_RIGHT}"
)
if [[ "${EVAL_HARNESS}" == "1" ]]; then
  runner_args+=(
    --tree-eval-harness
    --tree-eval-backend "${EVAL_BACKEND}"
    --tree-eval-sif-cache "${SIF_CACHE}"
    --mc-eval-timeout-s "${EVAL_TIMEOUT_S}"
  )
fi

echo "[$(date -Iseconds)] mon-agent-runner ${runner_args[*]}"
mon-agent-runner "${runner_args[@]}"

# --------------------------------------------------------------------------- #
# 2) Summary                                                                  #
# --------------------------------------------------------------------------- #
# CSVs are already on disk: the runner writes <inst>.steps.csv into          #
# CSV_OUT_DIR as soon as each instance finishes (no batch post-processing). #
echo
echo "Heavy artifacts under ${OUTPUT_DIR}:"
find "${OUTPUT_DIR}" -maxdepth 2 -type f \
    \( -name '*.traj.json' -o -name '*.tree.json' \
       -o -name '*.tree.jsonl' \
       -o -name 'preds.json' -o -name 'minisweagent.log' \) | sort

echo
echo "CSV outputs under ${CSV_OUT_DIR}:"
find "${CSV_OUT_DIR}" -maxdepth 1 -type f -name '*.steps.csv' | sort

echo
echo "Per-instance y_root summary:"
python3 - <<PY
import json, glob, os
for p in sorted(glob.glob("${OUTPUT_DIR}/*/*.tree.json")):
    try:
        d = json.load(open(p))
        s = d.get("stats", {})
        inst = os.path.basename(os.path.dirname(p))
        print(f"  {inst:<45s} y_root={s.get('y_root'):.3f}  "
              f"n_leaves={s.get('n_leaves')}  n_success={s.get('n_success')}  "
              f"wall_s={s.get('wall_s')}")
    except Exception as e:
        print(f"  {p}: ERR {e}")
PY
