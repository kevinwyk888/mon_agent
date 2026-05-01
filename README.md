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

## Important dependency

This project is not fully standalone. It depends on a sibling checkout of
`mini-SWE-agent` and imports `minisweagent` directly.

The current setup script expects this directory layout:

```text
Agent/
  mini-SWE-agent/
  mon-agent/
```

## Quick setup

```bash
cd /path/to/Agent/mon-agent
bash scripts/setup_env.sh
source ~/.venvs/mon_agent_env/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## Reproducibility note

Several SLURM scripts contain cluster-specific values such as:

- account names
- scratch paths
- GPU partition names

These should be treated as local infrastructure settings and may need edits on a
different cluster.
