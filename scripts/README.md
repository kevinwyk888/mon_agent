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
