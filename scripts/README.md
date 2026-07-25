# Scripts

## Public Workflow

1. Create and activate an environment:

```bash
bash scripts/setup_env.sh
source ~/.venvs/mon_agent_env/bin/activate
```

2. Inspect the runner interface:

```bash
mon-agent-runner --help
```

3. Run with your own model endpoint and output directory:

```bash
mon-agent-runner \
  --subset lite \
  --split test \
  --output /path/to/results \
  --workers 1 \
  -c configs/swebench_lite.yaml \
  -c model.model_kwargs.api_base=http://127.0.0.1:8000/v1
```

4. Optionally add a slice for smoke tests:

```bash
mon-agent-runner \
  --subset lite \
  --split test \
  --slice 0:5 \
  --output /path/to/results \
  --workers 1 \
  -c configs/swebench_lite.yaml \
  -c model.model_kwargs.api_base=http://127.0.0.1:8000/v1
```

## CUSUM Rollback Experiment

The current intervention experiment is single-trajectory A/B testing, not tree
splitting:

1. Select 10-20 mixed-root tasks and train a CUSUM monitor on the remaining CSVs:

```bash
python scripts/prepare_retry_experiment.py \
  --data-dir alarm_monitor/data \
  --output-dir results/retry_experiment/monitor \
  --num-tasks 15 --seed 7
```

2. Run the control group: each selected task runs 20 times with no splitting and
   a 64-step index limit.

```bash
SELECTION=results/retry_experiment/monitor/selection.json \
CONDITION=control \
sbatch scripts/run_retry_experiment.sbatch
```

3. Run the intervention group: each selected task runs 20 times with no
   splitting; the trained CUSUM monitor triggers rollback to a previous
   checkpoint. The index limit remains 64 and total attempted steps are capped
   at `64 * 2.5 = 160`.

```bash
SELECTION=results/retry_experiment/monitor/selection.json \
CONDITION=intervention \
MONITOR_DIR=results/retry_experiment/monitor \
sbatch scripts/run_retry_experiment.sbatch
```

4. Compare average success rates:

```bash
python scripts/compare_retry_experiment.py \
  --control results/retry_control_<jobid> \
  --intervention results/retry_intervention_<jobid>
```

### XGBoost tuned CUSUM

Reuse the original held-out tasks when comparing XGBoost with an existing
linear CUSUM experiment. This retrains XGBoost only on the remaining CSVs and
tunes CUSUM on pure validation trajectories:

```bash
SELECTED_IDS=$(
  python -c 'import json; print(",".join(json.load(open("results/mon_lin/monitor/selection.json"))["selected_instances"]))'
)

python scripts/prepare_retry_experiment.py \
  --data-dir alarm_monitor/data \
  --output-dir results/mon_xgb/monitor \
  --selected-ids "$SELECTED_IDS" \
  --scorer xgboost \
  --window-size 64 \
  --stride 8 \
  --min-step 9 \
  --seed 7
```

Run one repeat without harness evaluation before submitting the full job:

```bash
SELECTION=results/mon_xgb/monitor/selection.json \
CONDITION=intervention \
MONITOR_DIR=results/mon_xgb/monitor \
RUN_NAME=smoke_xgboost_cusum \
REPEATS=1 WORKERS=1 EVALUATE_HARNESS=0 \
sbatch scripts/run_retry_experiment.sbatch
```

Then submit the 64-step-path, 160-total-step intervention:

```bash
SELECTION=results/mon_xgb/monitor/selection.json \
CONDITION=intervention \
MONITOR_DIR=results/mon_xgb/monitor \
RUN_NAME=int_xgb \
STEP_LIMIT=64 MAX_TOTAL_STEPS_MULTIPLIER=2.5 REPEATS=20 \
sbatch scripts/run_retry_experiment.sbatch
```

`ALARM_MIN_STEP` is optional. When it is unset, the runner uses the value saved
in the monitor artifacts, which is recommended for train/runtime consistency.

## Recreating your own batch script

If you need a cluster launcher, build a local wrapper around:

- `scripts/setup_env.sh`
- `mon-agent-runner`
- `configs/swebench_lite.yaml`

Your local wrapper should set:

- Python environment activation
- model endpoint URL
- output directory
- any site-specific cache, container, or scratch paths
