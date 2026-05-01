# mon-agent

`mon-agent` is a lightweight monitoring layer on top of `mini-SWE-agent` for
SWE-bench style runs. It adds:

- step-level logging
- compact prefix construction
- simple behavioral signals such as repetition and failure streak
- batch runner glue for SWE-bench experiments

## What is in this repo

- `src/mon_agent/`: main implementation
- `configs/`: experiment config and LiteLLM model registry
- `scripts/`: SLURM and setup scripts used for deployment and evaluation
- `feasible_method.md`: research notes for the monitoring approach
- `results/visualize.py`: helper script for analyzing saved trajectories

## Dependency model

`mon-agent` imports `minisweagent` directly. It therefore depends on
`mini-SWE-agent`.

- Default install path:
  `pyproject.toml` pins `mini-SWE-agent` to commit
  `ed58678c7e4670e1ffcbac35e90ca6ac8a5ddf7c`, so `pip install -e .` works even
  without a local sibling checkout.
- Local development path:
  if you already have a sibling `mini-SWE-agent/` checkout, the setup script
  will install that editable copy instead.

The local development layout is:

```text
Agent/
  mini-SWE-agent/
  mon-agent/
```

## Quick setup

### Option 1: simplest path

Clone only `mon-agent` and let pip install the pinned upstream dependency:

```bash
git clone <your-mon-agent-repo-url>
cd mon-agent
python3 -m venv ~/.venvs/mon_agent_env
source ~/.venvs/mon_agent_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[vllm]"
mon-agent-runner --help
```

### Option 2: develop against a local mini-SWE-agent checkout

Use this if you also want to inspect or modify the upstream agent code:

```bash
git clone <your-mon-agent-repo-url>
git clone https://github.com/SWE-agent/mini-SWE-agent
cd mini-SWE-agent
git checkout ed58678c7e4670e1ffcbac35e90ca6ac8a5ddf7c

cd ../mon-agent
bash scripts/setup_env.sh
source ~/.venvs/mon_agent_env/bin/activate
mon-agent-runner --help
```

### Import check

```bash
python - <<'PY'
import mon_agent.agent
import mon_agent.runner
import mon_agent.prefix
print("imports_ok")
PY
```

## Running the code

The main CLI entrypoint is:

```bash
mon-agent-runner --help
```

The existing SLURM wrappers remain available under `scripts/`, for example:

```bash
sbatch scripts/run_all_single_node.slurm
```

## Reproducibility note

Several SLURM scripts contain cluster-specific values such as:

- account names
- scratch paths
- GPU partition names

These should be treated as local infrastructure settings and may need edits on a
different cluster.
