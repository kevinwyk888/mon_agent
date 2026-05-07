"""SWE-bench runner that uses MonitoringAgent instead of the default agent.

This mirrors minisweagent.run.benchmarks.swebench but swaps in MonitoringAgent
for step-level logging and prefix tracking.  Everything else (dataset loading,
environment setup, config merging, progress display) is reused from mini-SWE-agent.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
import traceback
from pathlib import Path

import typer
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    filter_instances,
    get_sb_environment,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

from mon_agent.agent import MonitoringAgent

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# ------------------------------------------------------------------
# Per-instance processing
# ------------------------------------------------------------------


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWE-bench instance with MonitoringAgent."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id

    # Clean stale state (trajectory + monitoring sidecars; the sidecars are
    # written in append mode so leaving them around would mix runs).
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    for stale in (
        f"{instance_id}.traj.json",
        f"{instance_id}.steps.jsonl",
        f"{instance_id}.mc.jsonl",
        f"{instance_id}.traj.steps.jsonl",
        f"{instance_id}.traj.mc.jsonl",
    ):
        (instance_dir / stale).unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    result = None
    extra_info: dict = {}

    try:
        env = get_sb_environment(config, instance)

        # Set output_path so DefaultAgent.run() saves after EVERY step.
        # This way we don't lose data if SLURM kills the job.
        agent_config = config.get("agent", {}).copy()
        traj_path = instance_dir / f"{instance_id}.traj.json"
        agent_config["output_path"] = str(traj_path)

        agent = MonitoringAgent(
            model,
            env,
            task=task,
            mc_config=config.get("mc_fork"),
            **agent_config,
        )

        # Provide progress updates via a simple callback
        _orig_step = agent.step

        def _step_with_progress():
            progress_manager.update_instance_status(
                instance_id,
                f"Step {agent.n_calls + 1:3d} (${agent.cost:.2f})",
            )
            return _orig_step()

        agent.step = _step_with_progress  # type: ignore[assignment]

        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error("Error processing instance %s: %s", instance_id, e, exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            # Save trajectory (includes step_logs and prefix via serialize())
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            # Save step logs as separate JSONL for easy downstream processing
            agent.save_step_logs(instance_dir / f"{instance_id}.steps.jsonl")
            logger.info("Saved trajectory + step logs for '%s'", instance_id)

        update_preds_file(
            output_dir / "preds.json", instance_id, model.config.model_name, result
        )
        progress_manager.on_instance_end(instance_id, exit_status)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


@app.command()
def main(
    subset: str = typer.Option("lite", "--subset"),
    split: str = typer.Option("dev", "--split"),
    slice_spec: str = typer.Option("", "--slice"),
    filter_spec: str = typer.Option("", "--filter"),
    shuffle: bool = typer.Option(False, "--shuffle"),
    output: str = typer.Option("", "-o", "--output"),
    workers: int = typer.Option(1, "-w", "--workers"),
    model: str | None = typer.Option(None, "-m", "--model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config"
    ),
    environment_class: str | None = typer.Option(None, "--environment-class"),
    # ------------------------------------------------------------------
    # Monte-Carlo prefix evaluation flags (Phase-2 of feasible_method.md)
    # ------------------------------------------------------------------
    mc_fork: bool = typer.Option(
        False, "--mc-fork/--no-mc-fork",
        help="Enable Monte-Carlo fork rollouts to estimate Y_k = P(success | prefix_k).",
    ),
    mc_fork_every: int = typer.Option(
        5, "--mc-fork-every", help="Run MC every N main-trajectory steps."
    ),
    mc_samples: int = typer.Option(
        4, "--mc-samples", help="Number of fork rollouts per snapshot (M)."
    ),
    mc_temperature: float = typer.Option(
        0.7, "--mc-temperature", help="Sampling temperature inside forks."
    ),
    mc_top_p: float = typer.Option(
        0.95, "--mc-top-p", help="top_p inside forks."
    ),
    mc_max_fork_steps: int = typer.Option(
        20, "--mc-max-fork-steps", help="Static cap on fork steps (used when --mc-max-fork-steps-total <= 0)."
    ),
    mc_max_fork_steps_min: int = typer.Option(
        0, "--mc-max-fork-steps-min",
        help="Floor for the dynamic cap. Effective with --mc-max-fork-steps-total > 0.",
    ),
    mc_max_fork_steps_total: int = typer.Option(
        0, "--mc-max-fork-steps-total",
        help="If > 0, cap = max(--mc-max-fork-steps-min, total - current_step). Overrides --mc-max-fork-steps.",
    ),
    mc_fork_cost_budget: float = typer.Option(
        1.0, "--mc-fork-cost-budget", help="Max additional $ cost per fork."
    ),
    mc_snapshot_cwd: str = typer.Option(
        "/testbed", "--mc-snapshot-cwd", help="Container working dir to snapshot via git."
    ),
) -> None:
    """Run mini-SWE-agent on SWE-bench with MonitoringAgent."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info("Loading dataset %s, split %s...", dataset_path, split)
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(
        instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle
    )
    if not redo_existing and (output_path / "preds.json").exists():
        existing = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info("Skipping %d existing instances", len(existing))
        instances = [i for i in instances if i["instance_id"] not in existing]
    logger.info("Running on %d instances...", len(instances))

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "environment": {"environment_class": environment_class or UNSET},
            "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
            "mc_fork": {
                "enabled": mc_fork,
                "fork_every": mc_fork_every,
                "samples": mc_samples,
                "temperature": mc_temperature,
                "top_p": mc_top_p,
                "max_fork_steps": mc_max_fork_steps,
                "max_fork_steps_min": mc_max_fork_steps_min,
                "max_fork_steps_total": mc_max_fork_steps_total,
                "fork_cost_budget": mc_fork_cost_budget,
                "snapshot_cwd": mc_snapshot_cwd,
            },
        }
    )
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                iid = futures[future]
                logger.error("Error in future for %s: %s", iid, e, exc_info=True)
                progress_manager.on_uncaught_exception(iid, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance, inst, output_path, config, progress_manager
                ): inst["instance_id"]
                for inst in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling pending jobs...")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
