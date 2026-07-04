# mon-agent

`mon-agent` is a monitoring + tree-search wrapper around
[`mini-SWE-agent`](https://github.com/SWE-agent/mini-SWE-agent) for SWE-bench
runs. It adds:

- **Per-step logging** (token counts, command type, observation tag,
  repetition / failure-streak signals, plus a self-reported `confidence`
  in `[0, 1]` from a small one-shot LLM probe asked right after each step).
- **Binary tree search**: every K steps the trajectory snapshots the sandbox
  and forks into 2 children with fixed equal sampling temperatures
  (left and right both T=0.3 in the current pilot; configurable via
  `TREE_TEMPERATURE_LEFT` / `TREE_TEMPERATURE_RIGHT`). Search continues
  until any path hits the step budget (default 60) or submits a patch.
- **Real SWE-bench evaluation** at every leaf via the official harness — with
  a **Singularity backend** that works on HPC clusters where Docker is not
  available (e.g. UMich Great Lakes).
- **Online CSV export**: as soon as an instance's tree finishes, the runner
  flattens it into `<inst>.steps.csv` (one row per (node, step), with a `y`
  column = per-node subtree success rate). You can watch CSVs appear under
  `CSV_OUT_DIR` while the batch is still running — no post-processing step.

## Current Prefix Monitor

This checkout also contains the current offline prefix-alarm training
pipeline:

| artifact | purpose |
| --- | --- |
| [`prefix_alarm_monitor.ipynb`](prefix_alarm_monitor.ipynb) | notebook that trains the prefix score model, calibrates `p_success`, tunes alarm rules, runs the no-confidence ablation, and computes the CUSUM backtracking diagnostic |
| [`prefix_alarm_monitor_lib.py`](prefix_alarm_monitor_lib.py) | importable copy of the notebook implementation, including `run_experiment(args)` |
| [`MONITOR_TRAINING.md`](MONITOR_TRAINING.md) | detailed explanation of the model, rules, metrics, latest numbers, and how to rerun |
| [`monitor_results/leaf_low_fp_monitor/`](monitor_results/leaf_low_fp_monitor/) | latest full-feature monitor artifacts (`summary.json`, predictions, feature weights, calibration files) |
| [`monitor_results/no_confidence_leaf_low_fp_monitor/`](monitor_results/no_confidence_leaf_low_fp_monitor/) | ablation that drops all confidence-derived features |

The current default rule is `calibrated_leaf_low_fp`: alarm when calibrated
`p_success` falls below a validation-tuned threshold chosen to minimize
successful-path false alarms subject to a target failing-path recall. The
notebook also reports `cusum_leaf_low_fp`, which lowers false alarms by
accumulating risk over time, and `gated_leaf_low_fp`, which confirms the
calibrated alarm with uncertainty/drop signals.

Latest full-feature test metrics from `summary.json`:

| rule | failing-path recall | successful-path false alarm | median true alarm step | mean early-warning |
| --- | ---: | ---: | ---: | ---: |
| `calibrated_leaf_low_fp` | 84.3% | 18.0% | 40 | 38.5% |
| `cusum_leaf_low_fp` | 82.1% | 10.4% | 49 | 22.3% |
| `gated_leaf_low_fp` | 84.1% | 17.3% | 40 | 38.0% |

`CUSUM Alarm Backtracking` in the notebook traces each CUSUM alarm backward
to the shortest recent contribution window that explains the threshold
crossing. In the latest run, the all-path median rollback is about 32 real
steps, or 8 sampled prefixes, on both validation and test.

## Layout

```
src/mon_agent/
    agent.py                 MonitoringAgent (per-step features)
    tree_search.py           Binary tree-search driver
    tree_to_csv.py           Library: flatten tree.json -> steps.csv
                             (called by runner online; also exposed as CLI)
    mc_fork.py               (legacy) flat MC fork
    runner.py                CLI entrypoint (mon-agent-runner)
    evaluate.py              Docker harness evaluator
    evaluate_singularity.py  Singularity harness evaluator (default on HPC)
configs/
    swebench_lite.yaml       Dataset / model / sandbox config
scripts/
    setup_env.sh             Create venv & install
    run_mc_fork.sh           Run experiment (CSVs emitted incrementally)
    run_mc_fork.sbatch       SLURM wrapper around the above
results/
    tree_to_csv.py           Standalone CLI re-runner over an existing tree dir
```

## Quick start (Great Lakes, from scratch)

```bash
# 1. Get a node (spgpu, 2h is plenty for a few instances).
salloc --account=alkontar0 --partition=spgpu \
       --gpus=2 --cpus-per-task=16 --mem=64G --time=2:00:00

# 2. One-time environment setup (creates ~/.venvs/mon_agent_env).
git clone <your-mon-agent-repo-url> ~/mon_agent
cd ~/mon_agent
bash scripts/setup_env.sh

# 3. Every shell that runs the agent needs these two:
module load singularity
source ~/.venvs/mon_agent_env/bin/activate

# 4. Provide the LLM API key. The default config uses DeepSeek.
#    Either export it now or put it into ~/.config/mini-swe-agent/.env
export DEEPSEEK_API_KEY=sk-...

# 5. Run a 1-instance smoke test (with real harness eval).
bash scripts/run_mc_fork.sh
```

This will:

1. drive the binary tree search on the first dev-split SWE-bench Lite instance,
2. evaluate every leaf's patch with the official harness via Singularity
   (first run pulls a ~1 GB `.sif` into `${SIF_CACHE}`; ~5–10 min one-time),
3. write the trajectory and tree to `${OUTPUT_DIR}/<instance_id>/` on scratch,
4. **as each instance finishes**, write `<instance_id>.steps.csv` into
   `${CSV_OUT_DIR}` (defaults to `./results/${RUN_NAME}/`) — you can `ls` /
   `tail` these while the rest of the batch is still running,
5. print a per-instance `y_root` summary.

## Tweaking the run

`scripts/run_mc_fork.sh` reads the following env vars (all optional):

| var                       | default                                              | meaning                                                                   |
| ------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------- |
| `SCRATCH_BASE`            | `/scratch/alkontar_root/alkontar0/kevinwyk`          | Root for heavy artifacts (trees, sifs). Override on other clusters.       |
| `RUN_NAME`                | `tree_run` (sbatch: `tree_run_<jobid>`)              | Subdir name under both `OUTPUT_DIR` and `CSV_OUT_DIR`.                    |
| `OUTPUT_DIR`              | `${SCRATCH_BASE}/mon_agent/${RUN_NAME}`              | Heavy artifacts (trajectories, `tree.json`, preds). Lives on scratch.     |
| `CSV_OUT_DIR`             | `${PWD}/results/${RUN_NAME}`                         | Per-instance `<inst>.steps.csv` lands here (online, in repo for easy diff). |
| `CONFIG`                  | `configs/swebench_lite.yaml`                         | mini-swe-agent config (model + sandbox).                                  |
| `SUBSET` / `SPLIT`        | `lite` / `dev`                                       | SWE-bench subset and split.                                               |
| `SLICE`                   | `1:2` (sbatch: `1:5`)                                | Python-style slice over the dataset (`0:23` = full lite-dev).             |
| `WORKERS`                 | `4`                                                  | Cross-instance concurrency.                                               |
| `TREE_FORK_EVERY`         | `5` (sbatch: `10`)                                   | Split every K steps.                                                      |
| `TREE_STEP_BUDGET`        | `60`                                                 | Hard cap along any root→leaf path.                                        |
| `TREE_FORK_COST_BUDGET`   | `5.0`                                                | Max additional USD per child segment.                                     |
| `EVAL_HARNESS`            | `1`                                                  | Set `0` to skip real evaluation (much faster; `y` becomes "patch non-empty"). |
| `EVAL_BACKEND`            | `singularity`                                        | Or `docker` if your machine has a Docker daemon.                          |
| `SIF_CACHE`               | `${SCRATCH_BASE}/swebench_sif`                       | Where instance `.sif` images are cached.                                  |
| `EVAL_TIMEOUT_S`          | `1800`                                               | Per-leaf evaluation wall-clock cap.                                       |

Examples:

```bash
# Full lite-dev (23 instances), 4 cross-instance workers, custom run name.
RUN_NAME=full_lite_dev SLICE=0:23 WORKERS=4 \
  bash scripts/run_mc_fork.sh

# Skip harness evaluation (debug / fast iteration).
EVAL_HARNESS=0 bash scripts/run_mc_fork.sh

# Watch CSVs appear in real time while the batch runs.
watch -n 5 'ls -1 results/${RUN_NAME:-tree_run}/*.steps.csv 2>/dev/null | wc -l'
```

## Submitting via SLURM

```bash
# defaults: OUTPUT_DIR=${SCRATCH_BASE}/mon_agent/tree_run_<jobid>,
#           CSV_OUT_DIR=./results/tree_run_<jobid>, slice 1:5,
#           2 GPUs / 16 CPUs / 64G / 8h
sbatch scripts/run_mc_fork.sbatch

# override knobs at submit time (full lite-dev, custom run name)
sbatch --export=ALL,SLICE=0:23,RUN_NAME=full_lite_dev \
       scripts/run_mc_fork.sbatch

# inspect logs (and watch CSVs land in ./results/<RUN_NAME>/)
tail -f logs/slurm-<jobid>.out
ls -1 results/<RUN_NAME>/*.steps.csv
```

The sbatch file already requests the same allocation as the interactive
`salloc` line above.

## Outputs

Heavy artifacts under `${OUTPUT_DIR}/<instance_id>/` (on scratch):

| file                          | contents                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| `<inst>.traj.json`            | mini-SWE-agent trajectory (messages + final info).                                    |
| `<inst>.tree.json`            | Nested tree with per-node `step_logs`, `submission`, `success`, `y`, plus `stats`.    |
| `<inst>.tree.jsonl`           | One flat record per node (depth, step range, features, harness fields).               |
| `${OUTPUT_DIR}/preds.json`    | SWE-bench predictions file (the deepest successful leaf's patch per instance).        |

Lightweight per-instance CSVs under `${CSV_OUT_DIR}/` (in repo by default):

| file                  | contents                                                                              |
| --------------------- | ------------------------------------------------------------------------------------- |
| `<inst>.steps.csv`    | One row per (node, step) with feature columns and `y`. Drops `cost`, `command_raw`.   |

CSVs are emitted **incrementally** by `mon-agent-runner` itself (via
`mon_agent.tree_to_csv.convert_one`) — each instance's CSV appears as soon as
that instance's `tree.json` is on disk. There is no separate batch
post-processing step.

CSV columns: `instance_id, node_id, depth, temperature, step_idx,
prompt_tokens, completion_tokens, prompt_tokens_cum, completion_tokens_cum,
step_wall_s, context_len_cum, command_type, target_file, returncode,
exception_flag, output_len, output_elided_chars, obs_tag,
repeat_cmd_score_recent, repeat_file_score_recent, failure_streak,
confidence, y`. The `confidence` column holds the LLM probe's
self-reported score in `[0, 1]`, or `NaN` if the probe failed both of its
two attempts to emit a parseable number.

See [`feasible_method.md`](feasible_method.md) for the full column dictionary
and a worked example. A short excerpt below shows the rows we care about
most — a fork where the two siblings, despite identical parent prefix and
identical sampling temperature (0.3), diverge sharply in their subtree
success rate `y`. These mixed-`y` forks (not the `y \in \{0, 1\}` extremes)
are the primary objects of study.

Real excerpt from
`results/tree_run_50411867/django__django-11815.steps.csv` (root segment,
first fork, then the next level down — long rows wrapped for display).
This run predates the `confidence` column; newer runs include it as the
second-to-last column, just before `y`.

```csv
instance_id,node_id,depth,temperature,step_idx,prompt_tokens,completion_tokens,prompt_tokens_cum,completion_tokens_cum,step_wall_s,context_len_cum,command_type,target_file,returncode,exception_flag,output_len,output_elided_chars,obs_tag,repeat_cmd_score_recent,repeat_file_score_recent,failure_streak,y
django__django-11815,0,0,0.3,1,1590,211,1590,211,2.47,6874,read,,0,0,1021,0,success,0.0,0.0,0,0.59375
django__django-11815,0,0,0.3,2,2180,80,3770,291,1.25,7812,search,,0,0,938,0,success,0.0,0.0,0,0.59375
django__django-11815,0,0,0.3,10,7271,80,45612,1722,1.65,23103,search,,0,0,652,0,success,0.333,0.0,0,0.59375
django__django-11815,0.0,1,0.3,11,7534,86,7534,86,1.56,24704,search,/testbed/tests/migrations/test_writer.py,0,0,1601,0,success,0.0,0.0,0,0.375
django__django-11815,0.1,1,0.3,11,7534,80,7534,80,1.38,24704,search,/testbed/tests/migrations/test_writer.py,0,0,1601,0,success,0.0,0.0,0,0.8125
django__django-11815,0.0.0,2,0.3,21,11970,498,11970,498,3.6,29408,run,django.conf,1,0,2024,0,exception,0.0,0.0,1,0.25
django__django-11815,0.0.1,2,0.3,21,11970,491,11970,491,3.64,27883,run,django.conf,0,0,499,0,success,0.0,0.0,0,0.5
django__django-11815,0.1.0,2,0.3,21,11748,200,11748,200,2.49,28896,run,django.db.migrations.serializer,1,0,380,0,exception,0.0,0.0,1,0.75
django__django-11815,0.1.1,2,0.3,21,11748,153,11748,153,2.62,28805,run,django.db.migrations.serializer,1,0,289,0,exception,0.0,0.0,1,0.875
```

Reading the excerpt:

- Root `node_id=0` has `y = 0.59375` — roughly 60% of leaves in its subtree
  ended up resolving the SWE-bench task.
- At step 11 the root forks: left child `0.0` collapses to `y = 0.375`,
  while right child `0.1` is much healthier at `y = 0.8125`. Same prefix,
  same temperature, ~2× gap in downstream success rate.
- One level down, the four grandchildren (`0.0.0`, `0.0.1`, `0.1.0`,
  `0.1.1`) span `y \in \{0.25, 0.5, 0.75, 0.875\}`. Note in particular
  that `0.1.0` opens with an exception (`returncode=1`,
  `failure_streak=1`) yet still leads to a `y = 0.75` subtree — local
  step-level failures do not directly imply doomed prefixes, which is the
  whole reason we need a prefix-level `Y_k` rather than per-step error
  counting.

To re-export CSVs from existing trees (e.g. after editing `tree_to_csv.py`):

```bash
python3 results/tree_to_csv.py ${OUTPUT_DIR} -d ${CSV_OUT_DIR}    # whole directory
python3 results/tree_to_csv.py ${OUTPUT_DIR}/<id>/<id>.tree.json  # one file
```

## Sandbox / harness notes

- `configs/swebench_lite.yaml` uses `environment_class: singularity` for the
  agent's own `/testbed` sandbox. The evaluator is independent and only needs
  `singularity` on PATH.
- On Great Lakes, `/usr/local/bin/singularity` is an Ansible **text stub**.
  Always `module load singularity` first; the real binary lives at
  `/opt/singularity/<ver>/bin/singularity`. The Singularity evaluator has
  ELF-magic detection to fall back to that path automatically.
- First evaluation of a new instance does
  `singularity pull docker://swebench/sweb.eval.x86_64.<sanitized>:latest`.
  The `<sanitized>` form replaces `__` with `_1776_`. Cached `.sif` files are
  named `<instance_id>.sif` under `${SIF_CACHE}` and reused.
- A successful leaf evaluation takes ~20–60 s; a single tree (≤4 leaves) plus
  pull adds ~3–15 min on first run, ~2–4 min on subsequent runs of the same
  instance.

## Verifying the install

```bash
source ~/.venvs/mon_agent_env/bin/activate
python - <<'PY'
import mon_agent.agent, mon_agent.runner, mon_agent.tree_search
import mon_agent.evaluate_singularity, mon_agent.tree_to_csv
print("imports_ok")
PY
mon-agent-runner --help | head -40
```

## Reproducibility note

The `salloc` / `sbatch` lines above use UMich Great Lakes specifics
(`--account=alkontar0`, `--partition=spgpu`). On other clusters, replace those
with your own account/partition. The agent itself does no GPU work; GPUs are
only requested because `spgpu` requires them.
