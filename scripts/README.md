# Scripts

This directory intentionally contains only the public setup helper:

- `setup_env.sh`

The original project also used several SLURM launch and validation scripts for a
specific HPC environment. Those scripts are not published here because they
embedded site-specific details such as:

- SLURM account names
- scratch filesystem paths
- partition and GPU choices
- container cache locations

## Public workflow

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
