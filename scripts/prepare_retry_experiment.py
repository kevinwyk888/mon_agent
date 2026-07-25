#!/usr/bin/env python3
"""Prepare the CUSUM rollback intervention experiment.

This selects 10-20 mixed-root SWE-bench tasks from existing tree-search
``*.steps.csv`` files, trains a prefix alarm monitor on the remaining files
with the same pipeline as ``alarm_monitor/prefix_alarm_monitor_lib.py``, and
writes a ``selection.json`` consumed by ``mon_agent.retry_runner``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path


DEFAULT_DATASET_SOURCES = [
    {"subset": "verified", "split": "test"},
    {"subset": "lite", "split": "test"},
]


def _load_lib(repo_root: Path):
    path = os.environ.get("ALARM_MODEL_LIB") or str(
        repo_root / "alarm_monitor" / "prefix_alarm_monitor_lib.py"
    )
    spec = importlib.util.spec_from_file_location("prefix_alarm_monitor_lib", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot import alarm lib from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("prefix_alarm_monitor_lib", module)
    spec.loader.exec_module(module)
    return module


def _load_xgboost_training(repo_root: Path):
    alarm_monitor_dir = repo_root / "alarm_monitor"
    if str(alarm_monitor_dir) not in sys.path:
        sys.path.insert(0, str(alarm_monitor_dir))
    from prefix_alarm_monitor.comparison import (
        BenchmarkConfig,
        retune_linear_monitor,
        train_xgboost_monitor,
    )

    return BenchmarkConfig, retune_linear_monitor, train_xgboost_monitor


def _root_y(csv_path: Path) -> float | None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                return float(row["y"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _instance_id(csv_path: Path) -> str:
    return csv_path.name[: -len(".steps.csv")] if csv_path.name.endswith(
        ".steps.csv"
    ) else csv_path.stem


def _parse_ids(text: str) -> list[str]:
    return [item for item in re.split(r"[,\s]+", text.strip()) if item]


def _resolve_dataset_membership(instance_ids: list[str]) -> dict[str, dict[str, str]]:
    from datasets import load_dataset

    dataset_names = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
    }
    membership: dict[str, dict[str, str]] = {}
    for source in DEFAULT_DATASET_SOURCES:
        subset = source["subset"]
        split = source["split"]
        ids = {row["instance_id"] for row in load_dataset(dataset_names[subset], split=split)}
        for instance_id in instance_ids:
            if instance_id in ids and instance_id not in membership:
                membership[instance_id] = {"subset": subset, "split": split}
    missing = [instance_id for instance_id in instance_ids if instance_id not in membership]
    if missing:
        raise SystemExit(
            "Selected ids are not in the Lite/Verified test pool used by the "
            f"707-task CSV set: {missing}"
        )
    return membership


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory of *.steps.csv files.")
    parser.add_argument(
        "--output-dir",
        default="results/retry_experiment/monitor",
        help="Where to write trained monitor artifacts and selection.json.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=15,
        help="Number of mixed tasks to hold out for the A/B experiment (10-20).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scorer",
        choices=("linear", "xgboost"),
        default="linear",
        help="Prefix scorer used by the tuned CUSUM monitor.",
    )
    parser.add_argument(
        "--selected-ids",
        default="",
        help="Comma/space separated mixed instance ids to force as the selected set.",
    )
    parser.add_argument(
        "--train-dir",
        default="",
        help="Scratch dir for symlinked training CSVs. Defaults to <output-dir>/_train_csvs.",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-step", type=int, default=9)
    parser.add_argument("--pure-val-ratio", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--row-weight", type=float, default=0.2)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--min-pair-gap", type=float, default=0.05)
    parser.add_argument("--calibration-learning-rate", type=float, default=0.05)
    parser.add_argument("--calibration-epochs", type=int, default=400)
    parser.add_argument("--calibration-l2", type=float, default=1e-3)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--target-leaf-recall", type=float, default=0.85)
    parser.add_argument("--rule-threshold-max-candidates", type=int, default=48)
    parser.add_argument("--drop-confidence-features", action="store_true")
    args = parser.parse_args()

    if args.selected_ids:
        requested = _parse_ids(args.selected_ids)
        if not 10 <= len(requested) <= 20:
            raise SystemExit("--selected-ids must contain 10 to 20 mixed tasks.")
    elif not 10 <= args.num_tasks <= 20:
        raise SystemExit("--num-tasks must be between 10 and 20.")

    lib = _load_lib(repo_root)
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(args.train_dir).resolve() if args.train_dir else output_dir / "_train_csvs"

    csv_paths = sorted(data_dir.glob("*.steps.csv")) or sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {data_dir}")

    id_to_path: dict[str, Path] = {}
    id_to_root_y: dict[str, float] = {}
    mixed: list[str] = []
    counts = {"pure0": 0, "pure1": 0, "mixed": 0, "unknown": 0}
    for path in csv_paths:
        inst = _instance_id(path)
        id_to_path[inst] = path
        root_y = _root_y(path)
        if root_y is None:
            counts["unknown"] += 1
            continue
        id_to_root_y[inst] = root_y
        kind = lib.root_kind_from_y(root_y)
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "mixed":
            mixed.append(inst)

    if len(mixed) < 2:
        raise SystemExit("Need at least two mixed instances so one can remain for training.")

    if args.selected_ids:
        selected = sorted(requested)
        bad = [inst for inst in selected if inst not in set(mixed)]
        if bad:
            raise SystemExit(f"Requested ids are not mixed / not found: {bad}")
    else:
        if len(mixed) <= args.num_tasks:
            raise SystemExit(
                f"Only {len(mixed)} mixed instances found; cannot hold out "
                f"{args.num_tasks} and leave mixed tasks for training."
            )
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(sorted(mixed), args.num_tasks))

    selected_set = set(selected)
    train_ids = [inst for inst in id_to_path if inst not in selected_set]

    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    for inst in train_ids:
        src = id_to_path[inst]
        dst = train_dir / src.name
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    print(
        f"Instances: {counts} | mixed={len(mixed)} | selected={len(selected)} "
        f"| training_files={len(train_ids)}"
    )
    print("Selected mixed instances for 20x control/intervention runs:")
    for inst in selected:
        print(f"  - {inst}  (root_y={id_to_root_y[inst]:.3f})")

    exp_args = argparse.Namespace(
        data_dir=str(train_dir),
        output_dir=str(output_dir),
        window_size=args.window_size,
        stride=args.stride,
        min_step=args.min_step,
        pure_val_ratio=args.pure_val_ratio,
        seed=args.seed,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        row_weight=args.row_weight,
        pairwise_weight=args.pairwise_weight,
        l2=args.l2,
        min_pair_gap=args.min_pair_gap,
        calibration_learning_rate=args.calibration_learning_rate,
        calibration_epochs=args.calibration_epochs,
        calibration_l2=args.calibration_l2,
        confidence_z=args.confidence_z,
        target_leaf_recall=args.target_leaf_recall,
        rule_threshold_max_candidates=args.rule_threshold_max_candidates,
        drop_confidence_features=args.drop_confidence_features,
    )
    print(f"\nTraining {args.scorer} tuned-CUSUM monitor ...")
    BenchmarkConfig, retune_linear_monitor, train_xgboost_monitor = (
        _load_xgboost_training(repo_root)
    )
    benchmark_config = BenchmarkConfig(
        data_dir=train_dir,
        output_root=output_dir,
        window_size=args.window_size,
        stride=args.stride,
        min_step=args.min_step,
        pure_val_ratio=args.pure_val_ratio,
        seed=args.seed,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        row_weight=args.row_weight,
        pairwise_weight=args.pairwise_weight,
        l2=args.l2,
        min_pair_gap=args.min_pair_gap,
        calibration_learning_rate=args.calibration_learning_rate,
        calibration_epochs=args.calibration_epochs,
        calibration_l2=args.calibration_l2,
        confidence_z=args.confidence_z,
        target_leaf_recall=args.target_leaf_recall,
        rule_threshold_max_candidates=args.rule_threshold_max_candidates,
    )
    if args.scorer == "xgboost":
        summary = train_xgboost_monitor(
            benchmark_config,
            output_dir,
            drop_confidence_features=args.drop_confidence_features,
        )
    else:
        summary = lib.run_experiment(exp_args)
        summary = retune_linear_monitor(benchmark_config, output_dir)
    cusum = summary["thresholds"].get("cusum_leaf_low_fp", {})
    threshold_parts = [
        "Trained.",
        f"cusum_drift={cusum.get('cusum_drift', float('nan')):.4g}",
        f"cusum_threshold={cusum.get('cusum_threshold', float('nan')):.4g}",
    ]
    success_threshold = summary["thresholds"].get("success_probability_threshold")
    if success_threshold is not None:
        threshold_parts.insert(1, f"success_threshold={success_threshold:.3f}")
    print(" ".join(threshold_parts))

    filter_regex = "(?:" + "|".join(re.escape(inst) for inst in selected) + ")$"
    dataset_by_instance = _resolve_dataset_membership(selected)
    selection = {
        "data_dir": str(data_dir),
        "monitor_dir": str(output_dir),
        "scorer": args.scorer,
        "drop_confidence_features": args.drop_confidence_features,
        "selected_instances": selected,
        "selected_root_y": {inst: id_to_root_y[inst] for inst in selected},
        "seed": args.seed,
        "num_tasks": len(selected),
        "n_mixed_total": len(mixed),
        "n_training_files": len(train_ids),
        "dataset_sources": DEFAULT_DATASET_SOURCES,
        "dataset_by_instance": dataset_by_instance,
        "filter_regex": filter_regex,
        "step_limit": 64,
        "max_total_steps_multiplier": 2.5,
        "default_repeats": 20,
    }
    sel_path = output_dir / "selection.json"
    sel_path.write_text(json.dumps(selection, indent=2))
    print(f"\nWrote {sel_path}")
    print("\nNext steps:")
    print(
        f"  SELECTION='{sel_path}' CONDITION=control "
        "sbatch scripts/run_retry_experiment.sbatch"
    )
    print(
        f"  SELECTION='{sel_path}' CONDITION=intervention MONITOR_DIR='{output_dir}' "
        "sbatch scripts/run_retry_experiment.sbatch"
    )


if __name__ == "__main__":
    main()