"""End-to-end benchmark for baseline and alternative prefix alarm monitors."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

try:
    from .. import prefix_alarm_monitor_lib as baseline
except ImportError:
    import prefix_alarm_monitor_lib as baseline

from .lightgbm_model import LightGBMPrefixScorer
from .mlp_model import LightweightMLPScorer
from .rules import (
    TunedRule,
    tune_consecutive_rule,
    tune_cusum_rule,
    tune_ewma_rule,
    tune_leaky_cusum_rule,
    tune_sliding_window_risk_rule,
)
from .xgboost_model import XGBoostPrefixScorer


TABLE_COLUMNS = [
    "scorer",
    "setup",
    "rule",
    "auroc",
    "pure0_leaf_alarm_recall",
    "pure1_leaf_false_alarm_rate",
    "pure0_leaf_alarm_median_step",
    "pure0_successful_warning_mean_early_pct",
]


@dataclass
class BenchmarkConfig:
    data_dir: Path
    output_root: Path
    window_size: int = 64
    stride: int = 8
    min_step: int = 9
    pure_val_ratio: float = 0.5
    seed: int = 42
    learning_rate: float = 0.05
    epochs: int = 250
    row_weight: float = 0.2
    pairwise_weight: float = 1.0
    l2: float = 1e-3
    min_pair_gap: float = 0.05
    calibration_learning_rate: float = 0.05
    calibration_epochs: int = 400
    calibration_l2: float = 1e-3
    confidence_z: float = 1.96
    target_leaf_recall: float = 0.85
    rule_threshold_max_candidates: int = 48
    force_baseline_retrain: bool = False
    force_scorer_retrain: bool = False
    force_rule_retune: bool = False


def default_config(root: Path | None = None) -> BenchmarkConfig:
    root = Path.cwd() if root is None else Path(root)
    return BenchmarkConfig(
        data_dir=root / "data",
        output_root=root / "monitor_results",
        window_size=baseline.WINDOW_SIZE,
        stride=baseline.STRIDE,
        min_step=baseline.MIN_STEP,
        pure_val_ratio=baseline.PURE_VAL_RATIO,
        seed=baseline.SEED,
        learning_rate=baseline.LEARNING_RATE,
        epochs=baseline.EPOCHS,
        row_weight=baseline.ROW_WEIGHT,
        pairwise_weight=baseline.PAIRWISE_WEIGHT,
        l2=baseline.L2,
        min_pair_gap=baseline.MIN_PAIR_GAP,
        calibration_learning_rate=baseline.CALIBRATION_LEARNING_RATE,
        calibration_epochs=baseline.CALIBRATION_EPOCHS,
        calibration_l2=baseline.CALIBRATION_L2,
        confidence_z=baseline.CONFIDENCE_Z,
        target_leaf_recall=baseline.TARGET_LEAF_RECALL,
        rule_threshold_max_candidates=baseline.RULE_THRESHOLD_MAX_CANDIDATES,
    )


def _baseline_args(config: BenchmarkConfig, output_dir: Path, drop_confidence: bool) -> argparse.Namespace:
    values = asdict(config)
    values.pop("output_root")
    values.pop("force_baseline_retrain")
    values.pop("force_scorer_retrain")
    values.pop("force_rule_retune")
    values["data_dir"] = str(config.data_dir)
    values["output_dir"] = str(output_dir)
    values["drop_confidence_features"] = drop_confidence
    return argparse.Namespace(**values)


def _load_or_run_baseline(
    config: BenchmarkConfig,
    output_dir: Path,
    drop_confidence: bool,
) -> Dict[str, object]:
    required_artifacts = [
        output_dir / "summary.json",
        output_dir / "val_predictions.csv",
        output_dir / "test_predictions.csv",
    ]
    if not config.force_baseline_retrain and all(path.is_file() for path in required_artifacts):
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    return baseline.run_experiment(_baseline_args(config, output_dir, drop_confidence))


def _prepare_data(config: BenchmarkConfig):
    instances = baseline.load_all_instances(config.data_dir)
    splits = baseline.stratified_pure_split(instances, config.pure_val_ratio, config.seed)
    bundle = baseline.build_samples(
        instances=instances,
        featurizer=baseline.PrefixFeaturizer(),
        split_spec=splits,
        window_size=config.window_size,
        stride=config.stride,
        min_step=config.min_step,
        min_pair_gap=config.min_pair_gap,
    )
    x_train_raw, y_train = baseline.samples_to_arrays(bundle.train_samples)
    x_val_raw, _ = baseline.samples_to_arrays(bundle.val_samples)
    x_test_raw, _ = baseline.samples_to_arrays(bundle.test_samples)
    scaler = baseline.Standardizer().fit(x_train_raw)
    (
        no_confidence_feature_names,
        x_train_no_confidence_raw,
        x_val_no_confidence_raw,
        x_test_no_confidence_raw,
        _,
        dropped_confidence_feature_names,
    ) = baseline.filter_feature_matrices(
        bundle.feature_names,
        x_train_raw,
        x_val_raw,
        x_test_raw,
        bundle.pair_diffs,
        drop_confidence_features=True,
    )
    no_confidence_scaler = baseline.Standardizer().fit(x_train_no_confidence_raw)
    return {
        "instances": instances,
        "bundle": bundle,
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "x_train": scaler.transform(x_train_raw),
        "x_val": scaler.transform(x_val_raw),
        "x_test": scaler.transform(x_test_raw),
        "x_train_no_confidence_raw": x_train_no_confidence_raw,
        "x_val_no_confidence_raw": x_val_no_confidence_raw,
        "x_test_no_confidence_raw": x_test_no_confidence_raw,
        "x_train_no_confidence": no_confidence_scaler.transform(x_train_no_confidence_raw),
        "x_val_no_confidence": no_confidence_scaler.transform(x_val_no_confidence_raw),
        "x_test_no_confidence": no_confidence_scaler.transform(x_test_no_confidence_raw),
        "feature_names": list(bundle.feature_names),
        "no_confidence_feature_names": no_confidence_feature_names,
        "dropped_confidence_feature_names": dropped_confidence_feature_names,
        "y_train": y_train,
        "y_val_success": np.asarray(
            [1.0 if sample.root_kind == "pure1" else 0.0 for sample in bundle.val_samples]
        ),
        "val_index": baseline.build_trajectory_index(bundle.val_samples, instances),
        "test_index": baseline.build_trajectory_index(bundle.test_samples, instances),
    }


def _load_baseline_probabilities(path: Path, samples) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(samples):
        raise RuntimeError(f"Prediction row count in {path} does not match prepared samples")
    for row, sample in zip(frame.itertuples(index=False), samples):
        if (
            row.instance_id != sample.instance_id
            or str(row.node_id) != sample.node_id
            or int(row.step_idx) != sample.step_idx
        ):
            raise RuntimeError(f"Prediction ordering in {path} does not match prepared samples")
    return frame["pred_success"].to_numpy(dtype=np.float64)


def _rule_payload(rule: TunedRule) -> Dict[str, object]:
    return {
        "parameters": rule.parameters,
        "metrics": {"val": rule.val_metrics, "test": rule.test_metrics},
    }


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _without_rollback(value):
    if isinstance(value, dict):
        return {
            key: _without_rollback(item)
            for key, item in value.items()
            if "rollback" not in key.lower()
        }
    if isinstance(value, list):
        return [_without_rollback(item) for item in value]
    return value


def _without_coarse_cusum(metrics_by_split):
    return {
        split: {
            rule_name: metrics
            for rule_name, metrics in metrics_by_rule.items()
            if rule_name != "cusum_leaf_low_fp"
        }
        for split, metrics_by_rule in metrics_by_split.items()
    }


def _runtime_trajectory_index(
    trajectory_index,
    stride: int,
):
    """Keep the samples that the single-trajectory runtime can reproduce."""
    result = []
    for item in trajectory_index:
        path_steps = np.asarray(item["path_steps"], dtype=np.int64)
        sample_indices = np.asarray(item["sample_indices"], dtype=np.int64)
        keep = (path_steps - 1) % stride == 0
        result.append(
            {
                **item,
                "path_steps": path_steps[keep],
                "sample_indices": sample_indices[keep],
            }
        )
    return result


def retune_linear_monitor(config: BenchmarkConfig, output_dir: Path) -> Dict[str, object]:
    """Replace the Linear monitor's tree-path CUSUM with runtime parameters."""
    output_dir = Path(output_dir).resolve()
    data = _prepare_data(config)
    bundle = data["bundle"]
    val_success = _load_baseline_probabilities(
        output_dir / "val_predictions.csv", bundle.val_samples
    )
    test_success = _load_baseline_probabilities(
        output_dir / "test_predictions.csv", bundle.test_samples
    )
    tuned_cusum = tune_cusum_rule(
        1.0 - val_success,
        1.0 - test_success,
        val_trajectory_index=_runtime_trajectory_index(
            data["val_index"], config.stride
        ),
        test_trajectory_index=_runtime_trajectory_index(
            data["test_index"], config.stride
        ),
        metric_fn=baseline.trajectory_alarm_metrics_from_index,
        target_leaf_recall=config.target_leaf_recall,
        max_candidates=64,
    )
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config"]["cusum_sampling_protocol"] = "runtime_fixed_stride"
    summary["thresholds"]["cusum_leaf_low_fp"] = {
        **tuned_cusum.val_metrics,
        **tuned_cusum.parameters,
        "selection_mode": "min_leaf_false_alarm_subject_to_leaf_recall",
    }
    summary["metrics"]["leaf_rule_comparison"]["val"][
        "cusum_leaf_low_fp"
    ] = tuned_cusum.val_metrics
    summary["metrics"]["leaf_rule_comparison"]["test"][
        "cusum_leaf_low_fp"
    ] = tuned_cusum.test_metrics
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def train_xgboost_monitor(
    config: BenchmarkConfig,
    output_dir: Path,
    *,
    drop_confidence_features: bool = False,
) -> Dict[str, object]:
    """Train XGBoost and tune CUSUM on held-out pure validation paths."""
    config.data_dir = Path(config.data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _prepare_data(config)
    matrix_suffix = "_no_confidence" if drop_confidence_features else ""
    feature_names = data[
        "no_confidence_feature_names" if drop_confidence_features else "feature_names"
    ]
    dropped_features = (
        data["dropped_confidence_feature_names"] if drop_confidence_features else []
    )

    scorer = XGBoostPrefixScorer(seed=config.seed).fit(
        data[f"x_train{matrix_suffix}_raw"],
        data["y_train"],
        data[f"x_val{matrix_suffix}_raw"],
        data["y_val_success"],
    )
    val_success = scorer.predict_success_prob(data[f"x_val{matrix_suffix}_raw"])
    test_success = scorer.predict_success_prob(data[f"x_test{matrix_suffix}_raw"])
    val_online_index = _runtime_trajectory_index(data["val_index"], config.stride)
    test_online_index = _runtime_trajectory_index(data["test_index"], config.stride)
    tuned_cusum = tune_cusum_rule(
        1.0 - val_success,
        1.0 - test_success,
        val_trajectory_index=val_online_index,
        test_trajectory_index=test_online_index,
        metric_fn=baseline.trajectory_alarm_metrics_from_index,
        target_leaf_recall=config.target_leaf_recall,
        max_candidates=64,
    )

    model_path = output_dir / "xgboost_model.json"
    scorer.save(model_path)
    bundle = data["bundle"]
    summary = {
        "config": {
            "data_dir": str(config.data_dir),
            "window_size": config.window_size,
            "stride": config.stride,
            "min_step": config.min_step,
            "pure_val_ratio": config.pure_val_ratio,
            "seed": config.seed,
            "target_leaf_recall": config.target_leaf_recall,
            "rule_threshold_max_candidates": config.rule_threshold_max_candidates,
            "drop_confidence_features": drop_confidence_features,
            "cusum_sampling_protocol": "runtime_fixed_stride",
        },
        "counts": {
            "num_instances": len(data["instances"]),
            "mixed_train_files": len(bundle.splits.train_mixed),
            "val_pure_files": len(bundle.splits.val_pure),
            "test_pure_files": len(bundle.splits.test_pure),
            "train_samples": len(bundle.train_samples),
            "val_samples": len(bundle.val_samples),
            "test_samples": len(bundle.test_samples),
            "num_features": len(feature_names),
        },
        "feature_filter": {"dropped_features": dropped_features},
        "thresholds": {"cusum_leaf_low_fp": tuned_cusum.parameters},
        "model": {
            "type": "xgboost",
            "path": model_path.name,
            "feature_names": feature_names,
        },
        "metrics": {
            "leaf_rule_comparison": {
                "val": {"cusum_leaf_low_fp": tuned_cusum.val_metrics},
                "test": {"cusum_leaf_low_fp": tuned_cusum.test_metrics},
            }
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "train_mixed": bundle.splits.train_mixed,
                "val_pure": bundle.splits.val_pure,
                "test_pure": bundle.splits.test_pure,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _append_rows(
    rows: list[Dict[str, object]],
    scorer: str,
    setup: str,
    metrics_by_rule: Mapping[str, Mapping[str, float]],
    auroc: float,
) -> None:
    for rule_name, metrics in metrics_by_rule.items():
        rows.append(
            {
                "scorer": scorer,
                "setup": setup,
                "rule": rule_name,
                "auroc": float(auroc),
                **metrics,
            }
        )


def run_benchmark(config: BenchmarkConfig | None = None) -> pd.DataFrame:
    """Run the original rules plus alternative scorer/rule combinations."""
    config = default_config() if config is None else config
    config.data_dir = Path(config.data_dir).resolve()
    config.output_root = Path(config.output_root).resolve()
    config.output_root.mkdir(parents=True, exist_ok=True)
    comparison_dir = config.output_root / "algorithm_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_path = comparison_dir / "summary.json"
    cached_additional: Dict[str, object] = {}
    if summary_path.is_file() and not config.force_rule_retune:
        cached_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cached_config = cached_payload.get("config", {})
        cache_keys = (
            "data_dir",
            "window_size",
            "stride",
            "min_step",
            "pure_val_ratio",
            "seed",
            "target_leaf_recall",
            "rule_threshold_max_candidates",
        )
        current_config = asdict(config)
        if all(str(cached_config.get(key)) == str(current_config.get(key)) for key in cache_keys):
            cached_additional = cached_payload.get("additional", {})

    full_dir = config.output_root / "leaf_low_fp_monitor"
    no_confidence_dir = config.output_root / "no_confidence_leaf_low_fp_monitor"
    full_summary = _load_or_run_baseline(config, full_dir, False)
    no_confidence_summary = _load_or_run_baseline(config, no_confidence_dir, True)

    data = _prepare_data(config)
    bundle = data["bundle"]
    val_index = data["val_index"]
    test_index = data["test_index"]
    metric_fn = baseline.trajectory_alarm_metrics_from_index
    linear_val_success = _load_baseline_probabilities(
        full_dir / "val_predictions.csv", bundle.val_samples
    )
    linear_test_success = _load_baseline_probabilities(
        full_dir / "test_predictions.csv", bundle.test_samples
    )
    no_confidence_val_success = _load_baseline_probabilities(
        no_confidence_dir / "val_predictions.csv", bundle.val_samples
    )
    no_confidence_test_success = _load_baseline_probabilities(
        no_confidence_dir / "test_predictions.csv", bundle.test_samples
    )
    y_test_fail = np.asarray(
        [1 if sample.root_kind == "pure0" else 0 for sample in bundle.test_samples],
        dtype=np.int64,
    )

    scorer_probabilities = {
        ("linear", "full_features"): (linear_val_success, linear_test_success),
        ("linear", "no_confidence"): (
            no_confidence_val_success,
            no_confidence_test_success,
        ),
    }

    for setup, matrix_suffix, artifact_suffix in (
        ("full_features", "", ""),
        ("no_confidence", "_no_confidence", "_no_confidence"),
    ):
        lightgbm_path = comparison_dir / f"lightgbm{artifact_suffix}_model.txt"
        if lightgbm_path.is_file() and not config.force_scorer_retrain:
            lightgbm = LightGBMPrefixScorer.load(lightgbm_path, seed=config.seed)
        else:
            lightgbm = LightGBMPrefixScorer(seed=config.seed).fit(
                data[f"x_train{matrix_suffix}_raw"],
                data["y_train"],
                data[f"x_val{matrix_suffix}_raw"],
                data["y_val_success"],
            )
            lightgbm.save(lightgbm_path)
        scorer_probabilities[("lightgbm", setup)] = (
            lightgbm.predict_success_prob(data[f"x_val{matrix_suffix}_raw"]),
            lightgbm.predict_success_prob(data[f"x_test{matrix_suffix}_raw"]),
        )

        mlp_path = comparison_dir / f"mlp{artifact_suffix}_model.npz"
        if mlp_path.is_file() and not config.force_scorer_retrain:
            mlp = LightweightMLPScorer.load(mlp_path, seed=config.seed)
        else:
            mlp = LightweightMLPScorer(seed=config.seed).fit(
                data[f"x_train{matrix_suffix}"],
                data["y_train"],
                data[f"x_val{matrix_suffix}"],
                data["y_val_success"],
            )
            mlp.save(mlp_path)
            (comparison_dir / f"mlp{artifact_suffix}_history.json").write_text(
                json.dumps(mlp.history, indent=2), encoding="utf-8"
            )
        scorer_probabilities[("mlp", setup)] = (
            mlp.predict_success_prob(data[f"x_val{matrix_suffix}"]),
            mlp.predict_success_prob(data[f"x_test{matrix_suffix}"]),
        )

        xgboost_path = comparison_dir / f"xgboost{artifact_suffix}_model.json"
        if xgboost_path.is_file() and not config.force_scorer_retrain:
            xgboost = XGBoostPrefixScorer.load(xgboost_path, seed=config.seed)
        else:
            xgboost = XGBoostPrefixScorer(seed=config.seed).fit(
                data[f"x_train{matrix_suffix}_raw"],
                data["y_train"],
                data[f"x_val{matrix_suffix}_raw"],
                data["y_val_success"],
            )
            xgboost.save(xgboost_path)
        scorer_probabilities[("xgboost", setup)] = (
            xgboost.predict_success_prob(data[f"x_val{matrix_suffix}_raw"]),
            xgboost.predict_success_prob(data[f"x_test{matrix_suffix}_raw"]),
        )

    scorer_aurocs = {
        f"{scorer}_{setup}": baseline.roc_auc_score_binary(
            y_test_fail, 1.0 - test_success
        )
        for (scorer, setup), (_, test_success) in scorer_probabilities.items()
    }

    tune_kwargs = {
        "val_trajectory_index": val_index,
        "test_trajectory_index": test_index,
        "metric_fn": metric_fn,
        "target_leaf_recall": config.target_leaf_recall,
        "max_candidates": min(config.rule_threshold_max_candidates, 32),
    }
    additional_payload: Dict[str, object] = {}

    def add_rule(name, tune):
        if name in cached_additional:
            additional_payload[name] = {
                key: value
                for key, value in cached_additional[name].items()
                if key != "rollback"
            }
        else:
            additional_payload[name] = _rule_payload(tune())

    for (scorer_name, setup), (val_success, test_success) in scorer_probabilities.items():
        cache_prefix = scorer_name if setup == "full_features" else f"{scorer_name}_no_confidence"
        add_rule(
            f"{cache_prefix}_consecutive_k_low_fp",
            lambda val_success=val_success, test_success=test_success: tune_consecutive_rule(
                1.0 - val_success,
                1.0 - test_success,
                **tune_kwargs,
            ),
        )
        add_rule(
            f"{cache_prefix}_ewma_low_fp",
            lambda val_success=val_success, test_success=test_success: tune_ewma_rule(
                1.0 - val_success,
                1.0 - test_success,
                **tune_kwargs,
            ),
        )
        add_rule(
            f"{cache_prefix}_cusum_tuned_low_fp",
            lambda val_success=val_success, test_success=test_success: tune_cusum_rule(
                1.0 - val_success,
                1.0 - test_success,
                **{**tune_kwargs, "max_candidates": 64},
            ),
        )
        add_rule(
            f"{cache_prefix}_leaky_cusum_low_fp",
            lambda val_success=val_success, test_success=test_success: tune_leaky_cusum_rule(
                1.0 - val_success,
                1.0 - test_success,
                **{**tune_kwargs, "max_candidates": 48},
            ),
        )
        add_rule(
            f"{cache_prefix}_sliding_window_risk_low_fp",
            lambda val_success=val_success, test_success=test_success: tune_sliding_window_risk_rule(
                1.0 - val_success,
                1.0 - test_success,
                **tune_kwargs,
            ),
        )

    calibrated_payloads: Dict[tuple[str, str], Dict[str, object]] = {}
    for (scorer_name, setup), (val_success, test_success) in scorer_probabilities.items():
        if scorer_name == "linear":
            baseline_summary = full_summary if setup == "full_features" else no_confidence_summary
            calibrated_payloads[(scorer_name, setup)] = {
                "metrics": {
                    split: baseline_summary["metrics"]["leaf_rule_comparison"][split][
                        "calibrated_leaf_low_fp"
                    ]
                    for split in ("val", "test")
                }
            }
            continue

        cache_prefix = scorer_name if setup == "full_features" else f"{scorer_name}_no_confidence"
        cache_name = f"{cache_prefix}_calibrated_leaf_low_fp"
        if cache_name in cached_additional:
            calibrated_payload = cached_additional[cache_name]
        else:
            threshold_info = baseline.choose_leaf_aware_success_threshold(
                y_true_fail=1.0 - data["y_val_success"],
                p_success=val_success,
                trajectory_index=val_index,
                target_leaf_recall=config.target_leaf_recall,
                max_candidates=config.rule_threshold_max_candidates,
            )
            success_threshold = threshold_info["success_threshold"]
            calibrated_payload = {
                "parameters": threshold_info,
                "metrics": {
                    "val": baseline.trajectory_alarm_metrics_from_index(
                        np.asarray(val_success < success_threshold, dtype=np.int64),
                        val_index,
                    ),
                    "test": baseline.trajectory_alarm_metrics_from_index(
                        np.asarray(test_success < success_threshold, dtype=np.int64),
                        test_index,
                    ),
                },
            }
        additional_payload[cache_name] = calibrated_payload
        calibrated_payloads[(scorer_name, setup)] = calibrated_payload

    rows: list[Dict[str, object]] = []
    for scorer_name in ("linear", "lightgbm", "mlp", "xgboost"):
        for setup in ("full_features", "no_confidence"):
            auroc = float(scorer_aurocs[f"{scorer_name}_{setup}"])
            rows.append(
                {
                    "scorer": scorer_name,
                    "setup": setup,
                    "rule": "calibrated_leaf_low_fp",
                    "auroc": auroc,
                    **calibrated_payloads[(scorer_name, setup)]["metrics"]["test"],
                }
            )
            if scorer_name == "linear":
                baseline_summary = full_summary if setup == "full_features" else no_confidence_summary
                rows.append(
                    {
                        "scorer": scorer_name,
                        "setup": setup,
                        "rule": "gated_leaf_low_fp",
                        "auroc": auroc,
                        **baseline_summary["metrics"]["leaf_rule_comparison"]["test"][
                            "gated_leaf_low_fp"
                        ],
                    }
                )

            cache_prefix = scorer_name if setup == "full_features" else f"{scorer_name}_no_confidence"
            for rule_suffix in (
                "cusum_tuned_low_fp",
                "consecutive_k_low_fp",
                "ewma_low_fp",
                "leaky_cusum_low_fp",
                "sliding_window_risk_low_fp",
            ):
                result = additional_payload[f"{cache_prefix}_{rule_suffix}"]
                rows.append(
                    {
                        "scorer": scorer_name,
                        "setup": setup,
                        "rule": rule_suffix,
                        "auroc": auroc,
                        **result["metrics"]["test"],
                    }
                )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(comparison_dir / "algorithm_comparison.csv", index=False)
    payload = {
        "config": asdict(config),
        "scorer_test_prefix_auroc": scorer_aurocs,
        "baseline": {
            "full_features": _without_coarse_cusum(
                full_summary["metrics"]["leaf_rule_comparison"]
            ),
            "no_confidence": _without_coarse_cusum(
                no_confidence_summary["metrics"]["leaf_rule_comparison"]
            ),
        },
        "additional": {
            **additional_payload,
        },
    }
    summary_path.write_text(
        json.dumps(_without_rollback(payload), indent=2, default=_json_default),
        encoding="utf-8",
    )
    return comparison[TABLE_COLUMNS]
