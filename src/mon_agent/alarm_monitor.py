"""Online prefix alarm monitor + intervention config.

This module re-uses the *exact* feature extractor and linear/calibrated head
trained by ``alarm_monitor/prefix_alarm_monitor_lib.py`` (via the notebook or via
``scripts/prepare_retry_experiment.py``) and applies it live while an agent is
running so downstream experiment code can raise an alarm mid-trajectory.

Given a trained-artifacts directory (produced by ``run_experiment``: it writes
``scaler.json``, ``calibrator.json``, ``feature_weights.csv`` and
``summary.json``), :class:`PrefixAlarmMonitor` reconstructs

    p_success = sigmoid( calib_scale * ( standardize(x) @ w + b ) + calib_bias )

for the current prefix window. Two alarm rules are supported (both trained /
tuned by ``run_experiment`` and stored in ``summary.json['thresholds']``):

* ``calibrated_leaf_low_fp`` -- fire when ``p_success < success_threshold``
  (reacts to a single low-probability prefix).
* ``cusum_leaf_low_fp`` (default) -- run a sequential CUSUM of
  ``fail_score = 1 - p_success`` sampled at the training cadence
  (``should_sample_step`` with ``stride``) and fire when the running statistic
  ``S_t = max(0, S_{t-1} + fail_score - cusum_drift)`` reaches
  ``cusum_threshold``. This accumulates moderate risk over time and empirically
  gives the best leaf-level low-false-positive behaviour.

The monitor is intentionally read-only and side-effect-free. Runtime
intervention policies, such as CUSUM-triggered rollback, live outside this
module and consume :class:`PrefixAlarmMonitor` scores.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class _XGBoostPredictor:
    def __init__(self, path: Path) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost monitor artifacts require the alarm-monitor extra"
            ) from exc
        self.model = xgb.Booster()
        self.model.load_model(path)

    def predict_success_prob(self, features: np.ndarray) -> float:
        best_iteration = getattr(self.model, "best_iteration", None)
        iteration_range = (
            (0, best_iteration + 1) if best_iteration is not None else (0, 0)
        )
        prediction = self.model.inplace_predict(
            np.asarray(features, dtype=np.float32)[None, :],
            iteration_range=iteration_range,
        )
        return float(np.clip(prediction[0], 1e-6, 1.0 - 1e-6))


# ---------------------------------------------------------------------------
# Locating and importing the shared alarm-model library
# ---------------------------------------------------------------------------


def _load_alarm_lib():
    """Import ``prefix_alarm_monitor_lib`` from the alarm_monitor directory.

    The library is not an installed package (it lives next to the notebook), so
    we locate it relative to the repo root or via the ``ALARM_MODEL_LIB``
    environment variable and import it by file path.
    """
    # 1) explicit override
    env_path = os.environ.get("ALARM_MODEL_LIB")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    # 2) repo-root relative (src/mon_agent/alarm_monitor.py -> repo root is
    #    three parents up: .../repo/src/mon_agent/alarm_monitor.py)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "alarm_monitor" / "prefix_alarm_monitor_lib.py")
    candidates.append(repo_root / "alarm_model" / "prefix_alarm_monitor_lib.py")

    # 3) current working directory fallbacks
    candidates.append(Path.cwd() / "alarm_monitor" / "prefix_alarm_monitor_lib.py")
    candidates.append(Path.cwd() / "alarm_model" / "prefix_alarm_monitor_lib.py")

    for path in candidates:
        if path and path.is_file():
            spec = importlib.util.spec_from_file_location(
                "prefix_alarm_monitor_lib", str(path)
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("prefix_alarm_monitor_lib", module)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not locate prefix_alarm_monitor_lib.py. Set the ALARM_MODEL_LIB "
        "environment variable to its full path, or ensure it exists under "
        "<repo>/alarm_monitor/."
    )


# ---------------------------------------------------------------------------
# The monitor
# ---------------------------------------------------------------------------


@dataclass
class AlarmScore:
    step_idx: int
    p_success: float
    raw_score: float
    alarmed: bool
    scored: bool
    """False when the prefix was too short (step_idx < min_step) to score."""
    cusum_stat: float = 0.0
    """Running CUSUM statistic at the current step (0.0 for the calibrated rule)."""


class PrefixAlarmMonitor:
    """Live re-implementation of the trained prefix alarm head."""

    VALID_RULES = ("cusum_leaf_low_fp", "calibrated_leaf_low_fp")

    def __init__(
        self,
        *,
        weights: np.ndarray,
        bias: float,
        kept_feature_names: Sequence[str],
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        calib_scale: float,
        calib_bias: float,
        success_threshold: float,
        window_size: int = 64,
        min_step: int = 9,
        stride: int = 8,
        rule: str = "cusum_leaf_low_fp",
        cusum_drift: float | None = None,
        cusum_threshold: float | None = None,
        predictor: _XGBoostPredictor | None = None,
    ) -> None:
        self._lib = _load_alarm_lib()
        self._featurizer = self._lib.PrefixFeaturizer()

        full_names = list(self._featurizer.feature_names)
        name_to_idx = {name: i for i, name in enumerate(full_names)}
        try:
            self._keep_idx = np.asarray(
                [name_to_idx[name] for name in kept_feature_names], dtype=np.int64
            )
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Trained feature {exc} not produced by the current "
                "PrefixFeaturizer; the alarm_model library and the trained "
                "artifacts are out of sync."
            ) from exc

        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.calib_scale = float(calib_scale)
        self.calib_bias = float(calib_bias)
        self.success_threshold = float(success_threshold)
        self.window_size = int(window_size)
        self.min_step = int(min_step)
        self.stride = max(1, int(stride))

        rule = (rule or "cusum_leaf_low_fp").strip()
        if rule not in self.VALID_RULES:
            raise ValueError(
                f"Unknown alarm rule {rule!r}; expected one of {self.VALID_RULES}."
            )
        # Fall back to the calibrated rule if CUSUM thresholds are unavailable.
        if rule == "cusum_leaf_low_fp" and (
            cusum_drift is None or cusum_threshold is None
        ):
            rule = "calibrated_leaf_low_fp"
        self.rule = rule
        self.cusum_drift = float(cusum_drift) if cusum_drift is not None else None
        self.cusum_threshold = (
            float(cusum_threshold) if cusum_threshold is not None else None
        )
        self._predictor = predictor

        n = len(self._keep_idx)
        for name, arr in (
            ("weights", self.weights),
            ("scaler_mean", self.scaler_mean),
            ("scaler_scale", self.scaler_scale),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} has length {arr.shape[0]} but there are {n} kept "
                    "features; artifacts are inconsistent."
                )

    # ------------------------------------------------------------------
    # Construction from disk
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts_dir(
        cls,
        artifacts_dir: str | Path,
        *,
        rule: str = "cusum_leaf_low_fp",
    ) -> "PrefixAlarmMonitor":
        artifacts_dir = Path(artifacts_dir)
        summary = json.loads((artifacts_dir / "summary.json").read_text())
        model_info = summary.get("model", {})
        if model_info.get("type") == "xgboost":
            dropped = set(summary.get("feature_filter", {}).get("dropped_features", []))
            full_names = list(_load_alarm_lib().PrefixFeaturizer().feature_names)
            kept_names = [name for name in full_names if name not in dropped]
            configured_names = model_info.get("feature_names")
            if configured_names and list(configured_names) != kept_names:
                raise ValueError(
                    "XGBoost artifact feature order does not match the current featurizer"
                )
            model_path = artifacts_dir / model_info.get("path", "xgboost_model.json")
            predictor = _XGBoostPredictor(model_path)
            if predictor.model.num_features() != len(kept_names):
                raise ValueError(
                    f"XGBoost model expects {predictor.model.num_features()} features "
                    f"but the current artifact configuration provides {len(kept_names)}"
                )

            cfg = summary.get("config", {})
            thresholds = summary.get("thresholds", {})
            cusum = thresholds.get("cusum_leaf_low_fp", {}) or {}
            n_features = len(kept_names)
            return cls(
                weights=np.zeros(n_features, dtype=np.float64),
                bias=0.0,
                kept_feature_names=kept_names,
                scaler_mean=np.zeros(n_features, dtype=np.float64),
                scaler_scale=np.ones(n_features, dtype=np.float64),
                calib_scale=1.0,
                calib_bias=0.0,
                success_threshold=float(
                    thresholds.get("success_probability_threshold", 0.5)
                ),
                window_size=int(cfg.get("window_size", 64)),
                min_step=int(cfg.get("min_step", 9)),
                stride=int(cfg.get("stride", 8)),
                rule=rule,
                cusum_drift=(
                    float(cusum["cusum_drift"])
                    if "cusum_drift" in cusum
                    else None
                ),
                cusum_threshold=(
                    float(cusum["cusum_threshold"])
                    if "cusum_threshold" in cusum
                    else None
                ),
                predictor=predictor,
            )

        scaler = json.loads((artifacts_dir / "scaler.json").read_text())
        calibrator = json.loads((artifacts_dir / "calibrator.json").read_text())

        # feature_weights.csv is written *sorted by |weight|* (see
        # save_feature_weights), so it is only a name->weight *map*, not a
        # positional order. The scaler (and the trained weight vector) live in
        # the featurizer's own order with any dropped features removed, so we
        # rebuild that canonical order here and align the weights to it.
        weight_by_name: dict[str, float] = {}
        with (artifacts_dir / "feature_weights.csv").open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            name_key = "feature" if "feature" in fields else fields[0]
            weight_key = "weight" if "weight" in fields else fields[-1]
            for row in reader:
                weight_by_name[row[name_key]] = float(row[weight_key])

        dropped = set(summary.get("feature_filter", {}).get("dropped_features", []))
        full_names = list(_load_alarm_lib().PrefixFeaturizer().feature_names)
        kept_names = [name for name in full_names if name not in dropped]

        missing = [n for n in kept_names if n not in weight_by_name]
        if missing:  # pragma: no cover - defensive
            raise ValueError(
                f"feature_weights.csv is missing weights for {missing[:5]}... "
                "(artifacts inconsistent with the current featurizer)."
            )
        weights = np.asarray([weight_by_name[n] for n in kept_names], dtype=np.float64)

        cfg = summary.get("config", {})
        thresholds = summary.get("thresholds", {})
        cusum = thresholds.get("cusum_leaf_low_fp", {}) or {}

        return cls(
            weights=weights,
            bias=float(model_info.get("bias", 0.0)),
            kept_feature_names=kept_names,
            scaler_mean=np.asarray(scaler.get("mean", []), dtype=np.float64),
            scaler_scale=np.asarray(scaler.get("scale", []), dtype=np.float64),
            calib_scale=float(calibrator.get("scale", 1.0)),
            calib_bias=float(calibrator.get("bias", 0.0)),
            success_threshold=float(
                thresholds.get("success_probability_threshold", 0.5)
            ),
            window_size=int(cfg.get("window_size", 64)),
            min_step=int(cfg.get("min_step", 9)),
            stride=int(cfg.get("stride", 8)),
            rule=rule,
            cusum_drift=(
                float(cusum["cusum_drift"]) if "cusum_drift" in cusum else None
            ),
            cusum_threshold=(
                float(cusum["cusum_threshold"])
                if "cusum_threshold" in cusum
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _row_records(
        self,
        step_logs: Sequence[dict[str, Any]],
        node_id: str,
        depth: int,
        temperature: float,
    ) -> list[Any]:
        RowRecord = self._lib.RowRecord
        safe_float = self._lib.safe_float
        rows: list[Any] = []
        for log in step_logs:
            confidence = log.get("confidence", float("nan"))
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = float("nan")
            rows.append(
                RowRecord(
                    instance_id=str(log.get("instance_id", "")),
                    node_id=node_id,
                    depth=int(depth),
                    temperature=float(temperature),
                    step_idx=int(log.get("step_idx", 0)),
                    prompt_tokens=safe_float(str(log.get("prompt_tokens", 0))),
                    completion_tokens=safe_float(str(log.get("completion_tokens", 0))),
                    prompt_tokens_cum=safe_float(str(log.get("prompt_tokens_cum", 0))),
                    completion_tokens_cum=safe_float(
                        str(log.get("completion_tokens_cum", 0))
                    ),
                    step_wall_s=safe_float(str(log.get("step_wall_s", 0))),
                    context_len_cum=safe_float(str(log.get("context_len_cum", 0))),
                    command_type=(log.get("command_type") or "unknown"),
                    target_file=(log.get("target_file") or ""),
                    returncode=safe_float(str(log.get("returncode", 0))),
                    exception_flag=safe_float(str(log.get("exception_flag", 0))),
                    output_len=safe_float(str(log.get("output_len", 0))),
                    output_elided_chars=safe_float(
                        str(log.get("output_elided_chars", 0))
                    ),
                    obs_tag=(log.get("obs_tag") or "unknown"),
                    repeat_cmd_score_recent=safe_float(
                        str(log.get("repeat_cmd_score_recent", 0))
                    ),
                    repeat_file_score_recent=safe_float(
                        str(log.get("repeat_file_score_recent", 0))
                    ),
                    failure_streak=safe_float(str(log.get("failure_streak", 0))),
                    confidence=confidence,
                    y=0.0,
                )
            )
        return rows

    def _p_for_prefix(self, rows: list[Any], end_index: int) -> tuple[float, float]:
        """Return ``(raw_score, p_success)`` for the prefix ``rows[:end_index+1]``.

        Mirrors training: only the trailing ``window_size`` rows feed the
        featurizer.
        """
        prefix_end = end_index + 1
        window_rows = rows[max(0, prefix_end - self.window_size) : prefix_end]
        full_vec = self._featurizer.transform_window(window_rows, self.window_size)
        x = full_vec[self._keep_idx]
        if self._predictor is not None:
            p_success = self._predictor.predict_success_prob(x)
            return p_success, p_success
        x_std = (x - self.scaler_mean) / self.scaler_scale
        raw_score = float(np.dot(x_std, self.weights) + self.bias)
        z = self.calib_scale * raw_score + self.calib_bias
        p_success = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))
        return raw_score, p_success

    def score(
        self,
        step_logs: Sequence[dict[str, Any]],
        *,
        node_id: str = "0",
        depth: int = 0,
        temperature: float = 0.0,
        min_step: int | None = None,
    ) -> AlarmScore:
        """Score the current prefix (chronological ``step_logs``).

        ``step_logs`` must be the full root->current path logs; only the
        trailing ``window_size`` entries are used, mirroring training.
        """
        if not step_logs:
            return AlarmScore(0, float("nan"), float("nan"), False, False)

        current_step = int(step_logs[-1].get("step_idx", len(step_logs)))
        effective_min = self.min_step if min_step is None else int(min_step)
        if current_step < effective_min:
            return AlarmScore(current_step, float("nan"), float("nan"), False, False)

        rows = self._row_records(step_logs, node_id, depth, temperature)
        last = len(rows) - 1
        raw_score, p_success = self._p_for_prefix(rows, last)

        if self.rule == "cusum_leaf_low_fp":
            # Sequential CUSUM of fail_score = 1 - p_success, evaluated at the
            # same sampled prefixes used in training (should_sample_step). The
            # running statistic is monotone-ish and, once it reaches the tuned
            # threshold, the trajectory is considered alarmed.
            node_step_indices = [int(r.step_idx) for r in rows]
            stat = 0.0
            crossed = False
            for i, r in enumerate(rows):
                s = int(r.step_idx)
                sampled = s >= effective_min and (s - 1) % self.stride == 0
                if not sampled:
                    continue
                _, p_i = self._p_for_prefix(rows, i)
                fail = 1.0 - p_i
                stat = max(0.0, stat + fail - (self.cusum_drift or 0.0))
                if stat >= (self.cusum_threshold or float("inf")):
                    crossed = True
            return AlarmScore(
                current_step, p_success, raw_score, bool(crossed), True, float(stat)
            )

        # calibrated_leaf_low_fp: single-prefix threshold on p_success.
        alarmed = p_success < self.success_threshold
        return AlarmScore(current_step, p_success, raw_score, bool(alarmed), True)

