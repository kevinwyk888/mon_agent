"""Validation-tuned sequential alarm rules for prefix trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence

import numpy as np


Trajectory = Mapping[str, object]
MetricFunction = Callable[[np.ndarray, Sequence[Trajectory]], Dict[str, float]]


@dataclass
class TunedRule:
    """Selected rule parameters and metrics on validation and test paths."""

    parameters: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    val_alarm_flags: np.ndarray
    test_alarm_flags: np.ndarray
    val_rollback: Dict[str, float]
    test_rollback: Dict[str, float]


def _candidate_thresholds(values: np.ndarray, max_candidates: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.5], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, min(max_candidates, values.size))
    return np.unique(np.quantile(values, quantiles))


def _selection_key(metrics: Mapping[str, float]) -> tuple[float, float, float, float]:
    early = float(metrics.get("pure0_successful_warning_mean_early_pct", float("nan")))
    if not np.isfinite(early):
        early = -1.0
    median_step = float(metrics.get("pure0_leaf_alarm_median_step", float("nan")))
    if not np.isfinite(median_step):
        median_step = float("inf")
    return (
        -float(metrics["pure1_leaf_false_alarm_rate"]),
        float(metrics["pure0_leaf_alarm_recall"]),
        early,
        -median_step,
    )


def _fallback_key(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["pure0_leaf_alarm_recall"])
        - float(metrics["pure1_leaf_false_alarm_rate"]),
        float(metrics["pure0_leaf_alarm_recall"]),
        -float(metrics["pure1_leaf_false_alarm_rate"]),
    )


def consecutive_alarm_flags(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    threshold: float,
    consecutive_k: int,
) -> np.ndarray:
    """Alarm after ``consecutive_k`` adjacent sampled prefixes exceed a threshold."""
    fail_scores = np.asarray(fail_scores, dtype=np.float64)
    alarm_flags = np.zeros(fail_scores.shape[0], dtype=np.int64)
    for item in trajectory_index:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        run_length = 0
        for sample_index in indices:
            if fail_scores[sample_index] >= threshold:
                run_length += 1
            else:
                run_length = 0
            if run_length >= consecutive_k:
                alarm_flags[sample_index] = 1
    return alarm_flags


def ewma_values(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    alpha: float,
) -> np.ndarray:
    """Compute path-local EWMA values and map them back to prefix rows."""
    fail_scores = np.asarray(fail_scores, dtype=np.float64)
    values = np.full(fail_scores.shape[0], np.nan, dtype=np.float64)
    for item in trajectory_index:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        running = 0.0
        for position, sample_index in enumerate(indices):
            score = float(fail_scores[sample_index])
            running = score if position == 0 else alpha * score + (1.0 - alpha) * running
            if np.isfinite(values[sample_index]) and not np.isclose(values[sample_index], running):
                raise RuntimeError("Shared prefix received inconsistent EWMA histories")
            values[sample_index] = running
    return np.nan_to_num(values, nan=0.0)


def cusum_values(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    drift: float,
) -> np.ndarray:
    """Compute one-sided path-local CUSUM values for every prefix row."""
    fail_scores = np.asarray(fail_scores, dtype=np.float64)
    values = np.full(fail_scores.shape[0], np.nan, dtype=np.float64)
    for item in trajectory_index:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        if indices.size == 0:
            continue
        cumulative = np.cumsum(fail_scores[indices] - drift)
        prefix_minimum = np.minimum.accumulate(
            np.concatenate(([0.0], cumulative))
        )[1:]
        running = cumulative - prefix_minimum
        existing = values[indices]
        if np.any(np.isfinite(existing) & ~np.isclose(existing, running)):
            raise RuntimeError("Shared prefix received inconsistent CUSUM histories")
        values[indices] = running
    return np.nan_to_num(values, nan=0.0)


def leaky_cusum_values(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    drift: float,
    decay: float,
) -> np.ndarray:
    """Compute path-local CUSUM values with exponentially decaying evidence."""
    fail_scores = np.asarray(fail_scores, dtype=np.float64)
    values = np.full(fail_scores.shape[0], np.nan, dtype=np.float64)
    for item in trajectory_index:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        running = np.zeros(indices.size, dtype=np.float64)
        state = 0.0
        for position, sample_index in enumerate(indices):
            state = max(0.0, decay * state + float(fail_scores[sample_index]) - drift)
            running[position] = state
        existing = values[indices]
        if np.any(np.isfinite(existing) & ~np.isclose(existing, running)):
            raise RuntimeError("Shared prefix received inconsistent leaky CUSUM histories")
        values[indices] = running
    return np.nan_to_num(values, nan=0.0)


def sliding_window_risk_values(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    window_size: int,
) -> np.ndarray:
    """Compute the mean failure risk over each path's most recent prefixes."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    fail_scores = np.asarray(fail_scores, dtype=np.float64)
    values = np.full(fail_scores.shape[0], np.nan, dtype=np.float64)
    for item in trajectory_index:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        path_scores = fail_scores[indices]
        cumulative = np.concatenate(([0.0], np.cumsum(path_scores)))
        positions = np.arange(indices.size)
        starts = np.maximum(0, positions - window_size + 1)
        running = (cumulative[positions + 1] - cumulative[starts]) / (
            positions - starts + 1
        )
        existing = values[indices]
        if np.any(np.isfinite(existing) & ~np.isclose(existing, running)):
            raise RuntimeError("Shared prefix received inconsistent sliding-window histories")
        values[indices] = running
    return np.nan_to_num(values, nan=0.0)


def _padded_path_scores(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Trajectory],
) -> tuple[np.ndarray, np.ndarray]:
    path_lengths = np.asarray(
        [len(item["sample_indices"]) for item in trajectory_index], dtype=np.int64
    )
    max_length = int(path_lengths.max()) if path_lengths.size else 0
    scores = np.zeros((len(trajectory_index), max_length), dtype=np.float64)
    valid = np.arange(max_length)[None, :] < path_lengths[:, None]
    for row, item in enumerate(trajectory_index):
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        scores[row, : indices.size] = fail_scores[indices]
    return scores, valid


def _cusum_path_maxima(path_scores: np.ndarray, valid: np.ndarray, drift: float) -> np.ndarray:
    if path_scores.shape[1] == 0:
        return np.zeros(path_scores.shape[0], dtype=np.float64)
    increments = np.where(valid, path_scores - drift, 0.0)
    cumulative = np.cumsum(increments, axis=1)
    prefix_minimum = np.minimum.accumulate(np.minimum(cumulative, 0.0), axis=1)
    running = cumulative - prefix_minimum
    return np.max(np.where(valid, running, 0.0), axis=1)


def _leaky_cusum_path_maxima(
    path_scores: np.ndarray,
    valid: np.ndarray,
    drift: float,
    decay: float,
) -> np.ndarray:
    states = np.zeros(path_scores.shape[0], dtype=np.float64)
    maxima = np.zeros(path_scores.shape[0], dtype=np.float64)
    for position in range(path_scores.shape[1]):
        active = valid[:, position]
        states[active] = np.maximum(
            0.0,
            decay * states[active] + path_scores[active, position] - drift,
        )
        maxima[active] = np.maximum(maxima[active], states[active])
    return maxima


def _rollback_summary(
    alarm_flags: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    risk_start_positions: Callable[[Trajectory, int], int],
) -> Dict[str, float]:
    rollback_steps = []
    rollback_prefixes = []
    for item in trajectory_index:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        steps = np.asarray(item["path_steps"], dtype=np.int64)
        hits = np.flatnonzero(alarm_flags[indices] == 1)
        if hits.size == 0:
            continue
        alarm_position = int(hits[0])
        start_position = max(0, min(alarm_position, risk_start_positions(item, alarm_position)))
        rollback_steps.append(int(steps[alarm_position] - steps[start_position]))
        rollback_prefixes.append(alarm_position - start_position)
    return {
        "alarmed_leaf_paths": float(len(rollback_steps)),
        "mean_rollback_steps": float(np.mean(rollback_steps)) if rollback_steps else float("nan"),
        "median_rollback_steps": float(np.median(rollback_steps)) if rollback_steps else float("nan"),
        "mean_rollback_sampled_prefixes": (
            float(np.mean(rollback_prefixes)) if rollback_prefixes else float("nan")
        ),
    }


def _consecutive_rollback(
    alarm_flags: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    consecutive_k: int,
) -> Dict[str, float]:
    return _rollback_summary(
        alarm_flags,
        trajectory_index,
        lambda _item, alarm_position: alarm_position - consecutive_k + 1,
    )


def _ewma_rollback(
    alarm_flags: np.ndarray,
    ewma: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    threshold: float,
) -> Dict[str, float]:
    def risk_start(item: Trajectory, alarm_position: int) -> int:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        path_ewma = ewma[indices]
        lower_threshold = 0.5 * threshold
        prior_safe = np.flatnonzero(path_ewma[: alarm_position + 1] < lower_threshold)
        return int(prior_safe[-1] + 1) if prior_safe.size else 0

    return _rollback_summary(alarm_flags, trajectory_index, risk_start)


def _cusum_rollback(
    alarm_flags: np.ndarray,
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Trajectory],
    drift: float,
    threshold: float,
) -> Dict[str, float]:
    def risk_start(item: Trajectory, alarm_position: int) -> int:
        indices = np.asarray(item["sample_indices"], dtype=np.int64)
        increments = fail_scores[indices[: alarm_position + 1]] - drift
        contribution = 0.0
        start_position = alarm_position
        for position in range(alarm_position, -1, -1):
            contribution += float(increments[position])
            if contribution >= threshold - 1e-12:
                start_position = position
                break
        return start_position

    return _rollback_summary(alarm_flags, trajectory_index, risk_start)


def tune_consecutive_rule(
    val_fail_scores: np.ndarray,
    test_fail_scores: np.ndarray,
    val_trajectory_index: Sequence[Trajectory],
    test_trajectory_index: Sequence[Trajectory],
    metric_fn: MetricFunction,
    target_leaf_recall: float,
    max_candidates: int = 32,
    consecutive_candidates: Sequence[int] = (2, 3, 4),
) -> TunedRule:
    """Tune threshold and run length using validation leaf-level metrics."""
    thresholds = _candidate_thresholds(val_fail_scores, max_candidates)
    best = None
    fallback = None
    for consecutive_k in consecutive_candidates:
        for threshold in thresholds:
            flags = consecutive_alarm_flags(
                val_fail_scores, val_trajectory_index, float(threshold), int(consecutive_k)
            )
            metrics = metric_fn(flags, val_trajectory_index)
            candidate = (float(threshold), int(consecutive_k), flags, metrics)
            if fallback is None or _fallback_key(metrics) > _fallback_key(fallback[3]):
                fallback = candidate
            if metrics["pure0_leaf_alarm_recall"] >= target_leaf_recall:
                if best is None or _selection_key(metrics) > _selection_key(best[3]):
                    best = candidate

    selected = best or fallback
    if selected is None:
        raise RuntimeError("Unable to tune consecutive alarm rule")
    threshold, consecutive_k, val_flags, val_metrics = selected
    test_flags = consecutive_alarm_flags(
        test_fail_scores, test_trajectory_index, threshold, consecutive_k
    )
    return TunedRule(
        parameters={
            "fail_threshold": threshold,
            "consecutive_k": float(consecutive_k),
            "selection_target_leaf_recall": float(target_leaf_recall),
        },
        val_metrics=val_metrics,
        test_metrics=metric_fn(test_flags, test_trajectory_index),
        val_alarm_flags=val_flags,
        test_alarm_flags=test_flags,
        val_rollback=_consecutive_rollback(val_flags, val_trajectory_index, consecutive_k),
        test_rollback=_consecutive_rollback(test_flags, test_trajectory_index, consecutive_k),
    )


def tune_ewma_rule(
    val_fail_scores: np.ndarray,
    test_fail_scores: np.ndarray,
    val_trajectory_index: Sequence[Trajectory],
    test_trajectory_index: Sequence[Trajectory],
    metric_fn: MetricFunction,
    target_leaf_recall: float,
    max_candidates: int = 32,
    alpha_candidates: Sequence[float] = (0.2, 0.35, 0.5, 0.7),
) -> TunedRule:
    """Tune EWMA smoothing and threshold using validation leaf-level metrics."""
    best = None
    fallback = None
    for alpha in alpha_candidates:
        val_ewma = ewma_values(val_fail_scores, val_trajectory_index, float(alpha))
        for threshold in _candidate_thresholds(val_ewma, max_candidates):
            flags = np.asarray(val_ewma >= threshold, dtype=np.int64)
            metrics = metric_fn(flags, val_trajectory_index)
            candidate = (float(alpha), float(threshold), val_ewma, flags, metrics)
            if fallback is None or _fallback_key(metrics) > _fallback_key(fallback[4]):
                fallback = candidate
            if metrics["pure0_leaf_alarm_recall"] >= target_leaf_recall:
                if best is None or _selection_key(metrics) > _selection_key(best[4]):
                    best = candidate

    selected = best or fallback
    if selected is None:
        raise RuntimeError("Unable to tune EWMA alarm rule")
    alpha, threshold, val_ewma, val_flags, val_metrics = selected
    test_ewma = ewma_values(test_fail_scores, test_trajectory_index, alpha)
    test_flags = np.asarray(test_ewma >= threshold, dtype=np.int64)
    return TunedRule(
        parameters={
            "alpha": alpha,
            "fail_threshold": threshold,
            "rollback_lower_threshold": 0.5 * threshold,
            "selection_target_leaf_recall": float(target_leaf_recall),
        },
        val_metrics=val_metrics,
        test_metrics=metric_fn(test_flags, test_trajectory_index),
        val_alarm_flags=val_flags,
        test_alarm_flags=test_flags,
        val_rollback=_ewma_rollback(val_flags, val_ewma, val_trajectory_index, threshold),
        test_rollback=_ewma_rollback(test_flags, test_ewma, test_trajectory_index, threshold),
    )


def tune_cusum_rule(
    val_fail_scores: np.ndarray,
    test_fail_scores: np.ndarray,
    val_trajectory_index: Sequence[Trajectory],
    test_trajectory_index: Sequence[Trajectory],
    metric_fn: MetricFunction,
    target_leaf_recall: float,
    max_candidates: int = 64,
    drift_candidates: Sequence[float] | None = None,
) -> TunedRule:
    """Tune one-sided CUSUM drift and threshold on validation leaf paths."""
    val_fail_scores = np.asarray(val_fail_scores, dtype=np.float64)
    test_fail_scores = np.asarray(test_fail_scores, dtype=np.float64)
    if drift_candidates is None:
        quantile_drifts = np.quantile(
            val_fail_scores[np.isfinite(val_fail_scores)],
            np.linspace(0.0, 0.95, 25),
        )
        drift_candidates = np.unique(
            np.concatenate(
                ([0.0], quantile_drifts, np.linspace(0.05, 0.9, 18))
            )
        )

    root_kinds = np.asarray([item["root_kind"] for item in val_trajectory_index])
    pure0_mask = root_kinds == "pure0"
    pure1_mask = root_kinds == "pure1"
    path_scores, valid = _padded_path_scores(val_fail_scores, val_trajectory_index)
    best = None
    fallback = None
    for drift in drift_candidates:
        drift = float(drift)
        if not 0.0 <= drift < 1.0:
            continue
        path_maxima = _cusum_path_maxima(path_scores, valid, drift)
        thresholds = _candidate_thresholds(path_maxima[path_maxima > 0.0], max_candidates)
        for threshold in thresholds:
            threshold = float(threshold)
            path_alarms = path_maxima >= threshold
            recall = float(np.mean(path_alarms[pure0_mask])) if pure0_mask.any() else 0.0
            false_alarm = float(np.mean(path_alarms[pure1_mask])) if pure1_mask.any() else 0.0
            candidate = (drift, threshold, recall, false_alarm)
            fallback_key = (recall - false_alarm, recall, -false_alarm, -threshold)
            if fallback is None or fallback_key > (
                fallback[2] - fallback[3], fallback[2], -fallback[3], -fallback[1]
            ):
                fallback = candidate
            if recall >= target_leaf_recall:
                selection_key = (-false_alarm, recall, -threshold)
                if best is None or selection_key > (-best[3], best[2], -best[1]):
                    best = candidate

    selected = best or fallback
    if selected is None:
        raise RuntimeError("Unable to tune CUSUM alarm rule")
    drift, threshold, _, _ = selected
    val_cusum = cusum_values(val_fail_scores, val_trajectory_index, drift)
    val_flags = np.asarray(val_cusum >= threshold, dtype=np.int64)
    val_metrics = metric_fn(val_flags, val_trajectory_index)
    test_cusum = cusum_values(test_fail_scores, test_trajectory_index, drift)
    test_flags = np.asarray(test_cusum >= threshold, dtype=np.int64)
    return TunedRule(
        parameters={
            "cusum_drift": drift,
            "cusum_threshold": threshold,
            "selection_target_leaf_recall": float(target_leaf_recall),
        },
        val_metrics=val_metrics,
        test_metrics=metric_fn(test_flags, test_trajectory_index),
        val_alarm_flags=val_flags,
        test_alarm_flags=test_flags,
        val_rollback=_cusum_rollback(
            val_flags, val_fail_scores, val_trajectory_index, drift, threshold
        ),
        test_rollback=_cusum_rollback(
            test_flags, test_fail_scores, test_trajectory_index, drift, threshold
        ),
    )


def tune_leaky_cusum_rule(
    val_fail_scores: np.ndarray,
    test_fail_scores: np.ndarray,
    val_trajectory_index: Sequence[Trajectory],
    test_trajectory_index: Sequence[Trajectory],
    metric_fn: MetricFunction,
    target_leaf_recall: float,
    max_candidates: int = 48,
    decay_candidates: Sequence[float] = (0.5, 0.7, 0.85, 0.95, 0.99),
    drift_candidates: Sequence[float] | None = None,
) -> TunedRule:
    """Tune leaky CUSUM decay, drift, and threshold on validation paths."""
    val_fail_scores = np.asarray(val_fail_scores, dtype=np.float64)
    test_fail_scores = np.asarray(test_fail_scores, dtype=np.float64)
    if drift_candidates is None:
        drift_candidates = np.unique(
            np.concatenate(
                (
                    [0.0],
                    np.quantile(
                        val_fail_scores[np.isfinite(val_fail_scores)],
                        np.linspace(0.0, 0.9, 12),
                    ),
                    np.linspace(0.1, 0.8, 8),
                )
            )
        )

    root_kinds = np.asarray([item["root_kind"] for item in val_trajectory_index])
    pure0_mask = root_kinds == "pure0"
    pure1_mask = root_kinds == "pure1"
    path_scores, valid = _padded_path_scores(val_fail_scores, val_trajectory_index)
    best = None
    fallback = None
    for decay in decay_candidates:
        decay = float(decay)
        if not 0.0 <= decay <= 1.0:
            continue
        for drift in drift_candidates:
            drift = float(drift)
            if not 0.0 <= drift < 1.0:
                continue
            path_maxima = _leaky_cusum_path_maxima(
                path_scores, valid, drift, decay
            )
            thresholds = _candidate_thresholds(
                path_maxima[path_maxima > 0.0], max_candidates
            )
            for threshold in thresholds:
                threshold = float(threshold)
                path_alarms = path_maxima >= threshold
                recall = float(np.mean(path_alarms[pure0_mask])) if pure0_mask.any() else 0.0
                false_alarm = (
                    float(np.mean(path_alarms[pure1_mask])) if pure1_mask.any() else 0.0
                )
                candidate = (decay, drift, threshold, recall, false_alarm)
                fallback_key = (recall - false_alarm, recall, -false_alarm, -threshold)
                if fallback is None or fallback_key > (
                    fallback[3] - fallback[4],
                    fallback[3],
                    -fallback[4],
                    -fallback[2],
                ):
                    fallback = candidate
                if recall >= target_leaf_recall:
                    selection_key = (-false_alarm, recall, -threshold)
                    if best is None or selection_key > (
                        -best[4], best[3], -best[2]
                    ):
                        best = candidate

    selected = best or fallback
    if selected is None:
        raise RuntimeError("Unable to tune leaky CUSUM alarm rule")
    decay, drift, threshold, _, _ = selected
    val_values = leaky_cusum_values(
        val_fail_scores, val_trajectory_index, drift, decay
    )
    test_values = leaky_cusum_values(
        test_fail_scores, test_trajectory_index, drift, decay
    )
    val_flags = np.asarray(val_values >= threshold, dtype=np.int64)
    test_flags = np.asarray(test_values >= threshold, dtype=np.int64)
    return TunedRule(
        parameters={
            "decay": decay,
            "cusum_drift": drift,
            "cusum_threshold": threshold,
            "selection_target_leaf_recall": float(target_leaf_recall),
        },
        val_metrics=metric_fn(val_flags, val_trajectory_index),
        test_metrics=metric_fn(test_flags, test_trajectory_index),
        val_alarm_flags=val_flags,
        test_alarm_flags=test_flags,
        val_rollback={},
        test_rollback={},
    )


def tune_sliding_window_risk_rule(
    val_fail_scores: np.ndarray,
    test_fail_scores: np.ndarray,
    val_trajectory_index: Sequence[Trajectory],
    test_trajectory_index: Sequence[Trajectory],
    metric_fn: MetricFunction,
    target_leaf_recall: float,
    max_candidates: int = 32,
    window_candidates: Sequence[int] = (2, 3, 4, 6, 8),
) -> TunedRule:
    """Tune rolling-mean window and risk threshold on validation paths."""
    best = None
    fallback = None
    for window_size in window_candidates:
        window_size = int(window_size)
        val_values = sliding_window_risk_values(
            val_fail_scores, val_trajectory_index, window_size
        )
        for threshold in _candidate_thresholds(val_values, max_candidates):
            threshold = float(threshold)
            flags = np.asarray(val_values >= threshold, dtype=np.int64)
            metrics = metric_fn(flags, val_trajectory_index)
            candidate = (window_size, threshold, flags, metrics)
            if fallback is None or _fallback_key(metrics) > _fallback_key(fallback[3]):
                fallback = candidate
            if metrics["pure0_leaf_alarm_recall"] >= target_leaf_recall:
                if best is None or _selection_key(metrics) > _selection_key(best[3]):
                    best = candidate

    selected = best or fallback
    if selected is None:
        raise RuntimeError("Unable to tune sliding-window risk rule")
    window_size, threshold, val_flags, val_metrics = selected
    test_values = sliding_window_risk_values(
        test_fail_scores, test_trajectory_index, window_size
    )
    test_flags = np.asarray(test_values >= threshold, dtype=np.int64)
    return TunedRule(
        parameters={
            "window_size": float(window_size),
            "fail_threshold": threshold,
            "selection_target_leaf_recall": float(target_leaf_recall),
        },
        val_metrics=val_metrics,
        test_metrics=metric_fn(test_flags, test_trajectory_index),
        val_alarm_flags=val_flags,
        test_alarm_flags=test_flags,
        val_rollback={},
        test_rollback={},
    )