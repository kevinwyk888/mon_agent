"""Single-trajectory baseline and CUSUM rollback experiment runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

from mon_agent._singularity_patch import apply_patches as _apply_sing_patches
from mon_agent._singularity_patch import build_env_for_instance
from mon_agent.agent import MonitoringAgent
from mon_agent.alarm_monitor import PrefixAlarmMonitor
from mon_agent.evaluate_singularity import evaluate_submission
from mon_agent.mc_fork import restore_workdir, snapshot_workdir
from mon_agent.tree_search import _seed_root_messages

_apply_sing_patches()

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"
DEFAULT_SELECTION_SOURCES = [
    ("lite", "test"),
    ("verified", "test"),
]


@dataclass
class AgentCheckpoint:
    step_idx: int
    snapshot_sha: str
    messages: list[dict[str, Any]]
    step_logs: list[dict[str, Any]]
    n_calls: int
    cost: float
    failure_streak: int
    last_test_output: str


def _stable_seed(seed_base: int, instance_id: str, repeat_idx: int, retry_round: int = 0) -> int:
    text = f"{seed_base}:{instance_id}:{repeat_idx}:{retry_round}".encode("utf-8")
    return seed_base + int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


def _copy_checkpoint(agent: MonitoringAgent, step_idx: int, snapshot_sha: str) -> AgentCheckpoint:
    return AgentCheckpoint(
        step_idx=step_idx,
        snapshot_sha=snapshot_sha,
        messages=copy.deepcopy(agent.messages),
        step_logs=copy.deepcopy(agent.step_logs),
        n_calls=agent.n_calls,
        cost=agent.cost,
        failure_streak=int(getattr(agent, "_failure_streak", 0)),
        last_test_output=str(getattr(agent, "_last_test_output", "")),
    )


def _restore_checkpoint(agent: MonitoringAgent, checkpoint: AgentCheckpoint) -> None:
    agent.messages = copy.deepcopy(checkpoint.messages)
    agent.step_logs = copy.deepcopy(checkpoint.step_logs)
    agent.n_calls = checkpoint.n_calls
    agent.cost = checkpoint.cost
    agent._failure_streak = checkpoint.failure_streak
    agent._last_test_output = checkpoint.last_test_output


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def _make_model_config(base_config: dict[str, Any], args: argparse.Namespace, seed: int) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    model_cfg = dict(config.get("model", {}))
    kwargs = dict(model_cfg.get("model_kwargs") or {})
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    kwargs["seed"] = seed
    model_cfg["model_kwargs"] = kwargs
    config["model"] = model_cfg
    return config


def _reseed_agent_model(agent: MonitoringAgent, args: argparse.Namespace, repeat_idx: int, retry_round: int) -> None:
    try:
        model_cfg = agent.model.config.model_dump()
    except Exception as exc:
        logger.warning("Could not reseed model after rollback: %s", exc)
        return
    kwargs = dict(model_cfg.get("model_kwargs") or {})
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    kwargs["seed"] = _stable_seed(args.seed_base, agent.instance_id, repeat_idx, retry_round)
    model_cfg["model_kwargs"] = kwargs
    agent.model = get_model(config=model_cfg)


def _last_exit(agent: MonitoringAgent) -> tuple[str, str]:
    if not agent.messages:
        return "", ""
    extra = agent.messages[-1].get("extra", {}) or {}
    return extra.get("exit_status", "") or "", extra.get("submission", "") or ""


def _score_cusum_details(
    monitor: PrefixAlarmMonitor,
    step_logs: list[dict[str, Any]],
    *,
    min_step: int | None = None,
) -> dict[str, Any]:
    if not step_logs:
        return {"scored": False, "alarmed": False}
    effective_min = monitor.min_step if min_step is None else int(min_step)
    rows = monitor._row_records(step_logs, "0", 0, 0.0)
    last = len(rows) - 1
    raw_score, p_success = monitor._p_for_prefix(rows, last)
    current_step = int(rows[-1].step_idx)
    if current_step < effective_min:
        return {
            "scored": False,
            "alarmed": False,
            "step_idx": current_step,
            "p_success": p_success,
            "raw_score": raw_score,
        }

    stat = 0.0
    crossed = False
    first_alarm_step = None
    segment_start_idx: int | None = None
    crossed_start_idx: int | None = None
    samples: list[dict[str, Any]] = []
    drift = monitor.cusum_drift or 0.0
    threshold = monitor.cusum_threshold or float("inf")

    for idx, row in enumerate(rows):
        step_idx = int(row.step_idx)
        sampled = step_idx >= effective_min and (
            (step_idx - 1) % monitor.stride == 0
        )
        if not sampled:
            continue
        _, p_i = monitor._p_for_prefix(rows, idx)
        fail_score = 1.0 - p_i
        increment = fail_score - drift
        next_stat = stat + increment
        if next_stat <= 0.0:
            stat = 0.0
            segment_start_idx = None
        else:
            if stat <= 0.0 or segment_start_idx is None:
                segment_start_idx = idx
            stat = next_stat
        sample = {
            "step_idx": step_idx,
            "p_success": p_i,
            "fail_score": fail_score,
            "cusum_stat": stat,
            "segment_start_step": (
                int(rows[segment_start_idx].step_idx)
                if segment_start_idx is not None else None
            ),
        }
        samples.append(sample)
        if not crossed and stat >= threshold:
            crossed = True
            first_alarm_step = step_idx
            crossed_start_idx = segment_start_idx

    rollback_step = None
    if crossed_start_idx is not None:
        rollback_step = int(rows[crossed_start_idx - 1].step_idx) if crossed_start_idx > 0 else 0

    return {
        "scored": True,
        "alarmed": crossed,
        "step_idx": current_step,
        "p_success": p_success,
        "raw_score": raw_score,
        "cusum_stat": stat,
        "cusum_threshold": threshold,
        "cusum_drift": drift,
        "first_alarm_step": first_alarm_step,
        "rollback_step": rollback_step,
        "samples": samples,
    }


def _choose_rollback_step(
    details: dict[str, Any],
    checkpoints: dict[int, AgentCheckpoint],
    current_step: int,
    fallback_steps: int,
) -> int:
    target = details.get("rollback_step")
    if not isinstance(target, int) or target >= current_step:
        target = max(0, current_step - fallback_steps)
    candidates = [step for step in checkpoints if step <= target and step < current_step]
    if not candidates:
        candidates = [step for step in checkpoints if step < current_step]
    return max(candidates) if candidates else 0


def _evaluate(
    *,
    args: argparse.Namespace,
    instance_id: str,
    submission: str,
    dataset_name: str,
    run_id: str,
    model_name: str,
) -> dict[str, Any]:
    if not args.evaluate_harness:
        return {"resolved": bool(submission.strip()), "eval_error": "not_evaluated"}
    result = evaluate_submission(
        instance_id=instance_id,
        patch=submission,
        dataset_name=dataset_name,
        run_id=run_id,
        model_name=model_name,
        timeout_s=args.eval_timeout_s,
        sif_cache_dir=args.sif_cache,
        keep_work_dir=args.keep_eval_logs,
        work_dir=(str(Path(args.eval_work_dir) / run_id) if args.eval_work_dir else None),
    )
    return {
        "resolved": bool(result.resolved),
        "eval_error": result.error,
        "eval_wall_s": result.wall_s,
        "eval_report_path": result.report_path,
    }


def _run_control(
    *,
    agent: MonitoringAgent,
    task: str,
) -> tuple[str, str, dict[str, Any]]:
    try:
        info = agent.run(task)
        return info.get("exit_status", ""), info.get("submission", ""), {}
    except Exception as exc:
        logger.error("Control run crashed: %s", exc, exc_info=True)
        return type(exc).__name__, "", {
            "traceback": traceback.format_exc(),
            "exception_str": str(exc),
        }


def _run_intervention(
    *,
    agent: MonitoringAgent,
    task: str,
    monitor: PrefixAlarmMonitor,
    args: argparse.Namespace,
    repeat_idx: int,
) -> tuple[str, str, dict[str, Any]]:
    _seed_root_messages(agent, task)
    root_sha = snapshot_workdir(agent.env, args.snapshot_cwd)
    checkpoints: dict[int, AgentCheckpoint] = {
        0: _copy_checkpoint(agent, 0, root_sha)
    }
    retry_events: list[dict[str, Any]] = []
    all_step_logs: list[dict[str, Any]] = []
    total_steps = 0
    retry_round = 0
    exit_status = ""
    submission = ""
    error_info: dict[str, Any] = {}

    while total_steps < args.max_total_steps:
        prev_len = len(agent.step_logs)
        try:
            agent.step()
        except InterruptAgentFlow as exc:
            agent.add_messages(*exc.messages)
        except Exception as exc:
            agent.handle_uncaught_exception(exc)
            error_info = {
                "traceback": traceback.format_exc(),
                "exception_str": str(exc),
            }
            break

        new_logs = agent.step_logs[prev_len:]
        for record in new_logs:
            total_steps += 1
            enriched = dict(record)
            enriched["total_step_idx"] = total_steps
            enriched["retry_round"] = retry_round
            all_step_logs.append(enriched)

        if agent.messages and agent.messages[-1].get("role") == "exit":
            exit_status, submission = _last_exit(agent)
            break
        if not new_logs:
            continue

        current_step = agent.n_calls
        try:
            sha = snapshot_workdir(agent.env, args.snapshot_cwd)
            checkpoints[current_step] = _copy_checkpoint(agent, current_step, sha)
        except Exception as exc:
            logger.warning("Checkpoint failed at step %d: %s", current_step, exc)

        details = _score_cusum_details(monitor, agent.step_logs, min_step=args.alarm_min_step)
        if not details.get("alarmed"):
            continue

        rollback_step = _choose_rollback_step(
            details,
            checkpoints,
            current_step,
            args.rollback_fallback_steps,
        )
        checkpoint = checkpoints.get(rollback_step)
        if checkpoint is None or rollback_step >= current_step:
            continue

        event = {
            "retry_round": retry_round + 1,
            "alarm_step": current_step,
            "rollback_step": rollback_step,
            "total_steps_at_alarm": total_steps,
            "p_success": details.get("p_success"),
            "cusum_stat": details.get("cusum_stat"),
            "cusum_threshold": details.get("cusum_threshold"),
            "first_alarm_step": details.get("first_alarm_step"),
            "rollback_reason": "cusum_contributing_window_start",
        }
        retry_events.append(event)
        restore_workdir(agent.env, args.snapshot_cwd, checkpoint.snapshot_sha)
        _restore_checkpoint(agent, checkpoint)
        retry_round += 1
        _reseed_agent_model(agent, args, repeat_idx, retry_round)

    if not exit_status:
        exit_status, submission = _last_exit(agent)
    budget_exhausted = total_steps >= args.max_total_steps and exit_status != "Submitted"
    if budget_exhausted:
        exit_status = "RetryTotalStepsExceeded"
        submission = ""

    return exit_status, submission, {
        **error_info,
        "total_steps": total_steps,
        "path_steps": agent.n_calls,
        "n_retries": len(retry_events),
        "retry_events": retry_events,
        "all_step_logs": all_step_logs,
        "budget_exhausted": budget_exhausted,
    }


def run_one(
    instance: dict[str, Any],
    repeat_idx: int,
    base_config: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    monitor: PrefixAlarmMonitor | None,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    dataset_name = instance.get("_dataset_name") or DATASET_MAPPING.get(args.subset, args.subset)
    dataset_split = instance.get("_dataset_split") or args.split
    run_name = f"{instance_id}__r{repeat_idx:02d}__{args.condition}"
    run_dir = output_dir / instance_id / f"r{repeat_idx:02d}_{args.condition}"
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.redo_existing:
        return json.loads(result_path.read_text())

    run_dir.mkdir(parents=True, exist_ok=True)
    seed = _stable_seed(args.seed_base, instance_id, repeat_idx)
    config = _make_model_config(base_config, args, seed)
    model = get_model(config=config.get("model", {}))
    model_name = getattr(getattr(model, "config", None), "model_name", None) or "retry_runner"
    env = None
    agent = None
    t0 = time.monotonic()
    exit_status = ""
    submission = ""
    extra: dict[str, Any] = {}

    try:
        env = build_env_for_instance(config, instance)
        agent_config = copy.deepcopy(config.get("agent", {}))
        agent_config["step_limit"] = args.step_limit
        agent_config["output_path"] = str(run_dir / f"{run_name}.traj.json")
        agent = MonitoringAgent(
            model,
            env,
            task=instance["problem_statement"],
            instance_id=instance_id,
            mc_config={"enabled": False},
            **agent_config,
        )
        if args.condition == "control":
            exit_status, submission, extra = _run_control(agent=agent, task=instance["problem_statement"])
        else:
            if monitor is None:
                raise RuntimeError("intervention condition requires a monitor")
            exit_status, submission, extra = _run_intervention(
                agent=agent,
                task=instance["problem_statement"],
                monitor=monitor,
                args=args,
                repeat_idx=repeat_idx,
            )
    except Exception as exc:
        logger.error("Run crashed for %s repeat %d: %s", instance_id, repeat_idx, exc, exc_info=True)
        exit_status = type(exc).__name__
        submission = ""
        extra = {"traceback": traceback.format_exc(), "exception_str": str(exc)}
    finally:
        if agent is not None:
            agent.save(
                run_dir / f"{run_name}.traj.json",
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": submission,
                        **{k: v for k, v in extra.items() if k not in ("all_step_logs", "retry_events")},
                    },
                    "instance_id": instance_id,
                    "repeat_idx": repeat_idx,
                    "condition": args.condition,
                },
            )
            agent.save_step_logs(run_dir / f"{run_name}.steps.jsonl")

    if extra.get("all_step_logs"):
        _write_jsonl(run_dir / f"{run_name}.all_steps.jsonl", extra["all_step_logs"])
    if extra.get("retry_events") is not None:
        _dump_json(run_dir / f"{run_name}.retry_events.json", extra.get("retry_events", []))

    eval_info = _evaluate(
        args=args,
        instance_id=instance_id,
        submission=submission,
        dataset_name=dataset_name,
        run_id=run_name,
        model_name=model_name,
    )
    row = {
        "instance_id": instance_id,
        "repeat_idx": repeat_idx,
        "condition": args.condition,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "seed": seed,
        "exit_status": exit_status,
        "submitted": bool(submission.strip()) and exit_status == "Submitted",
        "resolved": bool(eval_info.get("resolved")),
        "total_steps": int(extra.get("total_steps", len(getattr(agent, "step_logs", []) if agent else []))),
        "path_steps": int(extra.get("path_steps", getattr(agent, "n_calls", 0) if agent else 0)),
        "n_retries": int(extra.get("n_retries", 0)),
        "budget_exhausted": bool(extra.get("budget_exhausted", False)),
        "wall_s": round(time.monotonic() - t0, 2),
        "output_dir": str(run_dir),
        **eval_info,
    }
    _dump_json(result_path, row)
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance_id"], []).append(row)

    instance_rows = []
    for instance_id, inst_rows in sorted(by_instance.items()):
        n = len(inst_rows)
        resolved = sum(1 for row in inst_rows if row.get("resolved"))
        instance_rows.append(
            {
                "instance_id": instance_id,
                "n_runs": n,
                "success_rate": resolved / max(n, 1),
                "mean_total_steps": sum(float(row.get("total_steps", 0)) for row in inst_rows) / max(n, 1),
                "mean_retries": sum(float(row.get("n_retries", 0)) for row in inst_rows) / max(n, 1),
            }
        )

    n_total = len(rows)
    n_success = sum(1 for row in rows if row.get("resolved"))
    return {
        "condition": rows[0]["condition"] if rows else "",
        "n_runs": n_total,
        "n_success": n_success,
        "success_rate": n_success / max(n_total, 1),
        "mean_total_steps": sum(float(row.get("total_steps", 0)) for row in rows) / max(n_total, 1),
        "mean_retries": sum(float(row.get("n_retries", 0)) for row in rows) / max(n_total, 1),
        "instances": instance_rows,
    }


def _selection_sources(selection: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, str]]:
    raw_sources = selection.get("dataset_sources") or selection.get("source_datasets")
    sources: list[tuple[str, str]] = []
    if raw_sources:
        for item in raw_sources:
            if isinstance(item, str):
                parts = item.split(":", 1)
                sources.append((parts[0], parts[1] if len(parts) > 1 else args.split))
            elif isinstance(item, dict):
                subset = item.get("subset") or item.get("name") or item.get("dataset")
                split = item.get("split") or args.split
                if subset:
                    sources.append((str(subset), str(split)))
    if not sources:
        # A single --subset still works for old one-dataset selections. The
        # default sbatch value is verified, so include Lite as well to support
        # the 707-task Lite+Verified pool used by the current experiment.
        requested = (args.subset, args.split)
        sources = [requested]
        for source in DEFAULT_SELECTION_SOURCES:
            if source not in sources:
                sources.append(source)
    return sources


def _load_selected_instances(selection: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    from datasets import load_dataset

    selected_ids = selection.get("selected_instances") or selection.get("held_out_instances") or []
    if not selected_ids:
        raise SystemExit(f"No selected_instances found in {args.selection}")

    dataset_by_instance = selection.get("dataset_by_instance") or {}
    loaded_by_source: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    resolved: dict[str, dict[str, Any]] = {}
    membership: list[dict[str, str]] = []

    def load_source(subset: str, split: str) -> dict[str, dict[str, Any]]:
        key = (subset, split)
        if key in loaded_by_source:
            return loaded_by_source[key]
        dataset_name = DATASET_MAPPING.get(subset, subset)
        dataset = list(load_dataset(dataset_name, split=split))
        by_id: dict[str, dict[str, Any]] = {}
        for row in dataset:
            item = dict(row)
            item["_dataset_subset"] = subset
            item["_dataset_name"] = dataset_name
            item["_dataset_split"] = split
            by_id[item["instance_id"]] = item
        loaded_by_source[key] = by_id
        return by_id

    sources = _selection_sources(selection, args)
    for instance_id in selected_ids:
        preferred = dataset_by_instance.get(instance_id)
        candidates = list(sources)
        if isinstance(preferred, str):
            parts = preferred.split(":", 1)
            preferred_source = (parts[0], parts[1] if len(parts) > 1 else args.split)
            candidates = [preferred_source] + [source for source in candidates if source != preferred_source]
        elif isinstance(preferred, dict):
            subset = preferred.get("subset") or preferred.get("name") or preferred.get("dataset")
            split = preferred.get("split") or args.split
            if subset:
                preferred_source = (str(subset), str(split))
                candidates = [preferred_source] + [source for source in candidates if source != preferred_source]

        for subset, split in candidates:
            by_id = load_source(subset, split)
            if instance_id not in by_id:
                continue
            resolved[instance_id] = by_id[instance_id]
            membership.append({"instance_id": instance_id, "subset": subset, "split": split})
            break

    missing = [instance_id for instance_id in selected_ids if instance_id not in resolved]
    if missing:
        source_text = ", ".join(f"{subset}/{split}" for subset, split in sources)
        raise SystemExit(f"Selected ids not found in selection sources ({source_text}): {missing}")
    return [resolved[instance_id] for instance_id in selected_ids], membership


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="selection.json from prepare_retry_experiment.py")
    parser.add_argument("--condition", choices=("control", "intervention"), required=True)
    parser.add_argument("--monitor-dir", default="", help="Trained monitor artifact dir; required for intervention.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset", default="lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", dest="config_spec", action="append", default=[str(DEFAULT_CONFIG_FILE)])
    parser.add_argument("--environment-class", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-class", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--step-limit", type=int, default=64)
    parser.add_argument("--max-total-steps", type=int, default=0)
    parser.add_argument("--max-total-steps-multiplier", type=float, default=2.5)
    parser.add_argument("--snapshot-cwd", default="/testbed")
    parser.add_argument("--rollback-fallback-steps", type=int, default=8)
    parser.add_argument(
        "--alarm-min-step",
        type=int,
        default=None,
        help="Override the monitor artifact's minimum scoring step.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--evaluate-harness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-timeout-s", type=int, default=1800)
    parser.add_argument("--sif-cache", default=str(Path.home() / ".cache" / "swebench_sif"))
    parser.add_argument("--eval-work-dir", default="")
    parser.add_argument("--keep-eval-logs", action="store_true")
    parser.add_argument("--redo-existing", action="store_true")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    args.max_total_steps = args.max_total_steps or int(args.step_limit * args.max_total_steps_multiplier)
    if args.condition == "intervention" and not args.monitor_dir:
        raise SystemExit("--condition intervention requires --monitor-dir")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "retry_runner.log")

    selection = json.loads(Path(args.selection).read_text())
    selected_ids = selection.get("selected_instances") or selection.get("held_out_instances") or []
    if not selected_ids:
        raise SystemExit(f"No selected_instances found in {args.selection}")
    _dump_json(output_dir / "selection.json", selection)

    instances, dataset_membership = _load_selected_instances(selection, args)

    configs = [get_config_from_spec(spec) for spec in args.config_spec]
    configs.append(
        {
            "environment": {"environment_class": args.environment_class or UNSET},
            "model": {"model_name": args.model or UNSET, "model_class": args.model_class or UNSET},
            "mc_fork": {"enabled": False},
            "tree_search": {"enabled": False},
        }
    )
    base_config = recursive_merge(*configs)

    monitor = None
    if args.condition == "intervention":
        monitor = PrefixAlarmMonitor.from_artifacts_dir(args.monitor_dir, rule="cusum_leaf_low_fp")
        logger.info(
            "Loaded CUSUM monitor from %s (drift=%s threshold=%s)",
            args.monitor_dir,
            monitor.cusum_drift,
            monitor.cusum_threshold,
        )

    jobs = [(instance, repeat_idx) for instance in instances for repeat_idx in range(1, args.repeats + 1)]
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                instance,
                repeat_idx,
                base_config,
                output_dir,
                args,
                monitor,
            ): (instance["instance_id"], repeat_idx)
            for instance, repeat_idx in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            instance_id, repeat_idx = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                logger.error("Future failed for %s repeat %d: %s", instance_id, repeat_idx, exc, exc_info=True)
                row = {
                    "instance_id": instance_id,
                    "repeat_idx": repeat_idx,
                    "condition": args.condition,
                    "exit_status": type(exc).__name__,
                    "resolved": False,
                    "traceback": traceback.format_exc(),
                }
            rows.append(row)
            print(
                f"[{len(rows)}/{len(jobs)}] {instance_id} r{repeat_idx:02d} "
                f"resolved={row.get('resolved')} retries={row.get('n_retries', 0)} "
                f"steps={row.get('total_steps', 0)}"
            )

    rows.sort(key=lambda row: (row["instance_id"], int(row["repeat_idx"])))
    _write_jsonl(output_dir / "runs.jsonl", rows)
    summary = _summarize(rows)
    summary.update(
        {
            "selection": str(Path(args.selection).resolve()),
            "monitor_dir": str(Path(args.monitor_dir).resolve()) if args.monitor_dir else "",
            "step_limit": args.step_limit,
            "max_total_steps": args.max_total_steps,
            "repeats": args.repeats,
            "dataset_membership": dataset_membership,
            "dataset_sources": _selection_sources(selection, args),
        }
    )
    _dump_json(output_dir / "summary.json", summary)
    print(
        f"\n{args.condition}: n={summary['n_runs']} "
        f"success_rate={summary['success_rate']:.3f} "
        f"mean_total_steps={summary['mean_total_steps']:.1f} "
        f"mean_retries={summary['mean_retries']:.2f}"
    )


if __name__ == "__main__":
    main()