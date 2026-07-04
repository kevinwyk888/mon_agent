"""Shared implementation for the prefix alarm monitor notebook."""

from types import SimpleNamespace

import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print

# Shared implementation lives here so the notebook can stay concise.
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


COMMAND_TYPES = [
    "read",
    "search",
    "edit",
    "run",
    "test",
    "other",
    "diff",
    "explore",
    "install",
]

OBS_TAGS = [
    "success",
    "exception",
    "other_error",
    "cmd_error",
    "test_fail",
    "syntax_error",
]

CURRENT_NUMERIC_KEYS = [
    "step_idx",
    "depth",
    "temperature",
    "prompt_tokens",
    "completion_tokens",
    "prompt_tokens_cum",
    "completion_tokens_cum",
    "step_wall_s",
    "context_len_cum",
    "returncode",
    "exception_flag",
    "output_len",
    "output_elided_chars",
    "repeat_cmd_score_recent",
    "repeat_file_score_recent",
    "failure_streak",
    "confidence",
]

WINDOW_STATS_KEYS = [
    "prompt_tokens",
    "completion_tokens",
    "step_wall_s",
    "context_len_cum",
    "returncode",
    "exception_flag",
    "output_len",
    "output_elided_chars",
    "repeat_cmd_score_recent",
    "repeat_file_score_recent",
    "failure_streak",
    "confidence",
]

TARGET_FILE_FEATURE_NAMES = [
    "target_has_value",
    "target_len",
    "target_path_depth",
    "target_dot_count",
    "target_has_py",
    "target_has_test",
    "target_has_init",
    "target_is_abs_like",
    "target_symbol_like",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))




def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))

def safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def safe_int(value: str) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def is_pure_zero(y: float, tol: float = 1e-9) -> bool:
    return abs(y - 0.0) <= tol


def is_pure_one(y: float, tol: float = 1e-9) -> bool:
    return abs(y - 1.0) <= tol


def root_kind_from_y(y: float) -> str:
    if is_pure_zero(y):
        return "pure0"
    if is_pure_one(y):
        return "pure1"
    return "mixed"


def node_parent(node_id: str) -> str | None:
    if node_id == "0":
        return None
    return node_id.rsplit(".", 1)[0]


def node_depth(node_id: str) -> int:
    return node_id.count(".")


def sorted_node_ids(node_ids: Iterable[str]) -> List[str]:
    def key(node_id: str) -> Tuple[int, List[int]]:
        if node_id == "0":
            return (0, [0])
        return (node_depth(node_id), [int(part) for part in node_id.split(".")])

    return sorted(node_ids, key=key)


@dataclass
class RowRecord:
    instance_id: str
    node_id: str
    depth: int
    temperature: float
    step_idx: int
    prompt_tokens: float
    completion_tokens: float
    prompt_tokens_cum: float
    completion_tokens_cum: float
    step_wall_s: float
    context_len_cum: float
    command_type: str
    target_file: str
    returncode: float
    exception_flag: float
    output_len: float
    output_elided_chars: float
    obs_tag: str
    repeat_cmd_score_recent: float
    repeat_file_score_recent: float
    failure_streak: float
    confidence: float
    y: float

    def get(self, key: str) -> float:
        return float(getattr(self, key))


@dataclass
class InstanceData:
    instance_id: str
    root_y: float
    root_kind: str
    rows_by_node: Dict[str, List[RowRecord]]
    path_rows_by_node: Dict[str, List[RowRecord]]
    children_by_parent: Dict[str, List[str]]
    node_y: Dict[str, float]


@dataclass
class PrefixSample:
    instance_id: str
    root_kind: str
    node_id: str
    step_idx: int
    target_y: float
    is_final_step: bool
    is_root_prefix: bool
    feature_vector: np.ndarray


@dataclass
class SplitSpec:
    train_mixed: List[str]
    val_pure: List[str]
    test_pure: List[str]


@dataclass
class DatasetBundle:
    train_samples: List[PrefixSample]
    val_samples: List[PrefixSample]
    test_samples: List[PrefixSample]
    pair_diffs: np.ndarray
    pair_labels: np.ndarray
    pair_weights: np.ndarray
    feature_names: List[str]
    splits: SplitSpec


class PrefixFeaturizer:
    """Feature extractor over truncated prefix windows."""

    def __init__(self) -> None:
        self.command_types = list(COMMAND_TYPES)
        self.obs_tags = list(OBS_TAGS)
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        names.append("window_len")
        names.append("window_fill_ratio")

        for key in CURRENT_NUMERIC_KEYS:
            names.append(f"cur__{key}")
        names.append("cur__confidence_missing")

        for name in TARGET_FILE_FEATURE_NAMES:
            names.append(f"cur__{name}")

        for key in WINDOW_STATS_KEYS:
            for suffix in ("last", "mean", "min", "max", "std", "delta"):
                names.append(f"win__{key}__{suffix}")
        for key in ("confidence_missing", "returncode_nonzero", "target_missing"):
            for suffix in ("mean", "max", "delta"):
                names.append(f"win__{key}__{suffix}")

        for command_type in self.command_types:
            names.append(f"cur__cmd__{command_type}")
        names.append("cur__cmd__unknown")

        for obs_tag in self.obs_tags:
            names.append(f"cur__obs__{obs_tag}")
        names.append("cur__obs__unknown")

        for command_type in self.command_types:
            names.append(f"win__cmd_frac__{command_type}")
        names.append("win__cmd_frac__unknown")

        for obs_tag in self.obs_tags:
            names.append(f"win__obs_frac__{obs_tag}")
        names.append("win__obs_frac__unknown")

        names.extend(
            [
                "win__unique_target_ratio",
                "win__consecutive_same_target_ratio",
                "win__command_switch_ratio",
                "win__obs_switch_ratio",
            ]
        )
        return names

    @staticmethod
    def _clean_numeric(values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return np.zeros(1, dtype=np.float64)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _binary_from_nan(values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return np.zeros(1, dtype=np.float64)
        return np.asarray(np.isnan(arr), dtype=np.float64)

    def _stats(self, values: Sequence[float]) -> List[float]:
        arr = self._clean_numeric(values)
        last = float(arr[-1])
        mean = float(arr.mean())
        min_ = float(arr.min())
        max_ = float(arr.max())
        std = float(arr.std())
        delta = float(arr[-1] - arr[0])
        return [last, mean, min_, max_, std, delta]

    def _stats_short(self, values: Sequence[float]) -> List[float]:
        arr = self._clean_numeric(values)
        mean = float(arr.mean())
        max_ = float(arr.max())
        delta = float(arr[-1] - arr[0])
        return [mean, max_, delta]

    def _target_features(self, target_file: str) -> List[float]:
        text = (target_file or "").strip()
        lower = text.lower()
        path_depth = text.count("/") + text.count("\\")
        dot_count = text.count(".")
        has_py = int(".py" in lower or lower.endswith(".py"))
        has_test = int("test" in lower)
        has_init = int("__init__" in lower)
        is_abs_like = int(lower.startswith("/") or ":\\" in lower)
        symbol_like = int("/" not in text and "\\" not in text and "." in text)
        return [
            float(bool(text)),
            float(len(text)),
            float(path_depth),
            float(dot_count),
            float(has_py),
            float(has_test),
            float(has_init),
            float(is_abs_like),
            float(symbol_like),
        ]

    def _one_hot(self, value: str, vocab: Sequence[str]) -> List[float]:
        result = [0.0] * (len(vocab) + 1)
        if value in vocab:
            result[vocab.index(value)] = 1.0
        else:
            result[-1] = 1.0
        return result

    def _fraction_bag(self, values: Sequence[str], vocab: Sequence[str]) -> List[float]:
        counts = Counter(values)
        total = max(len(values), 1)
        result = []
        for item in vocab:
            result.append(counts.get(item, 0) / total)
        known = sum(counts.get(item, 0) for item in vocab)
        result.append(max(total - known, 0) / total)
        return result

    @staticmethod
    def _switch_ratio(values: Sequence[str]) -> float:
        if len(values) <= 1:
            return 0.0
        switches = 0
        for prev, cur in zip(values[:-1], values[1:]):
            if prev != cur:
                switches += 1
        return switches / (len(values) - 1)

    def transform_window(self, window_rows: Sequence[RowRecord], window_size: int) -> np.ndarray:
        if not window_rows:
            raise ValueError("window_rows must not be empty")

        current = window_rows[-1]
        values: List[float] = []
        values.append(float(len(window_rows)))
        values.append(float(len(window_rows) / window_size))

        for key in CURRENT_NUMERIC_KEYS:
            raw = current.get(key)
            values.append(0.0 if math.isnan(raw) else raw)
        values.append(float(math.isnan(current.confidence)))

        values.extend(self._target_features(current.target_file))

        for key in WINDOW_STATS_KEYS:
            values.extend(self._stats([row.get(key) for row in window_rows]))

        values.extend(
            self._stats_short([1.0 if math.isnan(row.confidence) else 0.0 for row in window_rows])
        )
        values.extend(
            self._stats_short([1.0 if row.returncode != 0 else 0.0 for row in window_rows])
        )
        values.extend(
            self._stats_short([0.0 if row.target_file else 1.0 for row in window_rows])
        )

        values.extend(self._one_hot(current.command_type, self.command_types))
        values.extend(self._one_hot(current.obs_tag, self.obs_tags))

        values.extend(self._fraction_bag([row.command_type for row in window_rows], self.command_types))
        values.extend(self._fraction_bag([row.obs_tag for row in window_rows], self.obs_tags))

        targets = [row.target_file for row in window_rows]
        unique_target_ratio = len(set(targets)) / max(len(targets), 1)
        same_target = 0
        if len(targets) > 1:
            same_target = sum(1 for a, b in zip(targets[:-1], targets[1:]) if a == b)
            same_target /= len(targets) - 1
        values.extend(
            [
                float(unique_target_ratio),
                float(same_target),
                float(self._switch_ratio([row.command_type for row in window_rows])),
                float(self._switch_ratio([row.obs_tag for row in window_rows])),
            ]
        )

        arr = np.asarray(values, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr


def load_instance(csv_path: Path) -> InstanceData:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                RowRecord(
                    instance_id=row["instance_id"],
                    node_id=row["node_id"],
                    depth=safe_int(row["depth"]),
                    temperature=safe_float(row["temperature"]),
                    step_idx=safe_int(row["step_idx"]),
                    prompt_tokens=safe_float(row["prompt_tokens"]),
                    completion_tokens=safe_float(row["completion_tokens"]),
                    prompt_tokens_cum=safe_float(row["prompt_tokens_cum"]),
                    completion_tokens_cum=safe_float(row["completion_tokens_cum"]),
                    step_wall_s=safe_float(row["step_wall_s"]),
                    context_len_cum=safe_float(row["context_len_cum"]),
                    command_type=(row["command_type"] or "unknown").strip(),
                    target_file=(row["target_file"] or "").strip(),
                    returncode=safe_float(row["returncode"]),
                    exception_flag=safe_float(row["exception_flag"]),
                    output_len=safe_float(row["output_len"]),
                    output_elided_chars=safe_float(row["output_elided_chars"]),
                    obs_tag=(row["obs_tag"] or "unknown").strip(),
                    repeat_cmd_score_recent=safe_float(row["repeat_cmd_score_recent"]),
                    repeat_file_score_recent=safe_float(row["repeat_file_score_recent"]),
                    failure_streak=safe_float(row["failure_streak"]),
                    confidence=safe_float(row["confidence"]),
                    y=safe_float(row["y"]),
                )
            )

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    instance_id = rows[0].instance_id
    root_y = rows[0].y
    root_kind = root_kind_from_y(root_y)
    rows_by_node: Dict[str, List[RowRecord]] = defaultdict(list)
    node_y: Dict[str, float] = {}
    for row in rows:
        rows_by_node[row.node_id].append(row)
        node_y[row.node_id] = row.y

    for node_rows in rows_by_node.values():
        node_rows.sort(key=lambda item: item.step_idx)

    path_rows_by_node: Dict[str, List[RowRecord]] = {}
    for node_id in sorted_node_ids(rows_by_node):
        parts = node_id.split(".")
        path_node_ids = [".".join(parts[:idx]) for idx in range(1, len(parts) + 1)]
        path_rows: List[RowRecord] = []
        for path_node_id in path_node_ids:
            path_rows.extend(rows_by_node[path_node_id])
        path_rows_by_node[node_id] = path_rows

    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    for node_id in rows_by_node:
        parent = node_parent(node_id)
        if parent is not None:
            children_by_parent[parent].append(node_id)

    for parent, children in children_by_parent.items():
        children_by_parent[parent] = sorted_node_ids(children)

    return InstanceData(
        instance_id=instance_id,
        root_y=root_y,
        root_kind=root_kind,
        rows_by_node=dict(rows_by_node),
        path_rows_by_node=path_rows_by_node,
        children_by_parent=dict(children_by_parent),
        node_y=node_y,
    )


def load_all_instances(data_dir: Path) -> Dict[str, InstanceData]:
    instances: Dict[str, InstanceData] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        instance = load_instance(csv_path)
        instances[instance.instance_id] = instance
    return instances


def stratified_pure_split(
    instances: Dict[str, InstanceData],
    pure_val_ratio: float,
    seed: int,
) -> SplitSpec:
    rng = random.Random(seed)
    pure0 = [inst_id for inst_id, inst in instances.items() if inst.root_kind == "pure0"]
    pure1 = [inst_id for inst_id, inst in instances.items() if inst.root_kind == "pure1"]
    mixed = [inst_id for inst_id, inst in instances.items() if inst.root_kind == "mixed"]

    rng.shuffle(pure0)
    rng.shuffle(pure1)

    def split_half(ids: List[str]) -> Tuple[List[str], List[str]]:
        cut = int(round(len(ids) * pure_val_ratio))
        cut = min(max(cut, 1), max(len(ids) - 1, 1)) if len(ids) > 1 else len(ids)
        return ids[:cut], ids[cut:]

    val0, test0 = split_half(pure0)
    val1, test1 = split_half(pure1)

    return SplitSpec(
        train_mixed=sorted(mixed),
        val_pure=sorted(val0 + val1),
        test_pure=sorted(test0 + test1),
    )


def should_sample_step(
    step_idx: int,
    node_step_indices: Sequence[int],
    stride: int,
    min_step: int,
) -> bool:
    if step_idx < min_step:
        return False
    return (step_idx - 1) % stride == 0 or step_idx == node_step_indices[-1]


def build_samples(
    instances: Dict[str, InstanceData],
    featurizer: PrefixFeaturizer,
    split_spec: SplitSpec,
    window_size: int,
    stride: int,
    min_step: int,
    min_pair_gap: float,
) -> DatasetBundle:
    feature_names = list(featurizer.feature_names)
    train_samples: List[PrefixSample] = []
    val_samples: List[PrefixSample] = []
    test_samples: List[PrefixSample] = []
    feature_by_key: Dict[Tuple[str, str, int], np.ndarray] = {}
    label_by_key: Dict[Tuple[str, str, int], float] = {}

    train_set = set(split_spec.train_mixed)
    val_set = set(split_spec.val_pure)
    test_set = set(split_spec.test_pure)

    for instance_id, instance in instances.items():
        for node_id in sorted_node_ids(instance.rows_by_node):
            node_rows = instance.rows_by_node[node_id]
            node_step_indices = [row.step_idx for row in node_rows]
            path_rows = instance.path_rows_by_node[node_id]
            path_by_step = {row.step_idx: idx for idx, row in enumerate(path_rows)}

            for row in node_rows:
                if not should_sample_step(row.step_idx, node_step_indices, stride, min_step):
                    continue
                prefix_end = path_by_step[row.step_idx] + 1
                window_rows = path_rows[max(0, prefix_end - window_size) : prefix_end]
                feature_vector = featurizer.transform_window(window_rows, window_size)
                sample = PrefixSample(
                    instance_id=instance_id,
                    root_kind=instance.root_kind,
                    node_id=node_id,
                    step_idx=row.step_idx,
                    target_y=row.y,
                    is_final_step=(row.step_idx == node_step_indices[-1]),
                    is_root_prefix=(node_id == "0"),
                    feature_vector=feature_vector,
                )
                key = (instance_id, node_id, row.step_idx)
                feature_by_key[key] = feature_vector
                label_by_key[key] = row.y
                if instance_id in train_set:
                    train_samples.append(sample)
                elif instance_id in val_set:
                    val_samples.append(sample)
                elif instance_id in test_set:
                    test_samples.append(sample)

    pair_diffs: List[np.ndarray] = []
    pair_labels: List[float] = []
    pair_weights: List[float] = []

    for instance_id in split_spec.train_mixed:
        instance = instances[instance_id]
        for parent, children in instance.children_by_parent.items():
            if len(children) != 2:
                continue
            left, right = children
            left_y = instance.node_y[left]
            right_y = instance.node_y[right]
            gap = abs(left_y - right_y)
            if gap < min_pair_gap:
                continue

            left_steps = {
                row.step_idx
                for row in instance.rows_by_node[left]
                if should_sample_step(
                    row.step_idx,
                    [item.step_idx for item in instance.rows_by_node[left]],
                    stride,
                    min_step,
                )
            }
            right_steps = {
                row.step_idx
                for row in instance.rows_by_node[right]
                if should_sample_step(
                    row.step_idx,
                    [item.step_idx for item in instance.rows_by_node[right]],
                    stride,
                    min_step,
                )
            }
            common_steps = sorted(left_steps & right_steps)
            if not common_steps:
                continue

            label = 1.0 if left_y > right_y else 0.0
            for step_idx in common_steps:
                left_key = (instance_id, left, step_idx)
                right_key = (instance_id, right, step_idx)
                if left_key not in feature_by_key or right_key not in feature_by_key:
                    continue
                pair_diffs.append(feature_by_key[left_key] - feature_by_key[right_key])
                pair_labels.append(label)
                pair_weights.append(gap)

    pair_diff_array = (
        np.vstack(pair_diffs).astype(np.float64)
        if pair_diffs
        else np.zeros((0, len(feature_names)), dtype=np.float64)
    )
    pair_label_array = np.asarray(pair_labels, dtype=np.float64)
    pair_weight_array = np.asarray(pair_weights, dtype=np.float64)

    return DatasetBundle(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        pair_diffs=pair_diff_array,
        pair_labels=pair_label_array,
        pair_weights=pair_weight_array,
        feature_names=feature_names,
        splits=split_spec,
    )


def samples_to_arrays(samples: Sequence[PrefixSample]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.vstack([sample.feature_vector for sample in samples]).astype(np.float64)
    y = np.asarray([sample.target_y for sample in samples], dtype=np.float64)
    return x, y


def filter_feature_matrices(
    feature_names: Sequence[str],
    x_train_raw: np.ndarray,
    x_val_raw: np.ndarray,
    x_test_raw: np.ndarray,
    pair_diffs_raw: np.ndarray,
    drop_confidence_features: bool,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_names = list(feature_names)
    if not drop_confidence_features:
        return (
            feature_names,
            x_train_raw,
            x_val_raw,
            x_test_raw,
            pair_diffs_raw,
            [],
        )

    keep_indices = [
        idx for idx, name in enumerate(feature_names) if "confidence" not in name.lower()
    ]
    dropped_feature_names = [
        name for idx, name in enumerate(feature_names) if idx not in keep_indices
    ]
    if not keep_indices:
        raise RuntimeError("Dropping confidence features removed every feature column")

    keep = np.asarray(keep_indices, dtype=np.int64)
    return (
        [feature_names[idx] for idx in keep_indices],
        x_train_raw[:, keep],
        x_val_raw[:, keep],
        x_test_raw[:, keep],
        pair_diffs_raw[:, keep],
        dropped_feature_names,
    )


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        return (x - self.mean_) / self.scale_

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else [],
        }


def soft_bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)).mean())


def pairwise_bce_loss(labels: np.ndarray, preds: np.ndarray, weights: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    eps = 1e-8
    preds = np.clip(preds, eps, 1.0 - eps)
    losses = -(labels * np.log(preds) + (1.0 - labels) * np.log(1.0 - preds))
    return float(np.sum(weights * losses) / max(np.sum(weights), 1e-8))


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    rank_sum = 0.0
    idx = 0
    n = len(y_true)
    while idx < n:
        end = idx + 1
        while end < n and sorted_scores[end] == sorted_scores[idx]:
            end += 1
        avg_rank = (idx + end - 1) / 2.0 + 1.0
        positive_count = int(sorted_labels[idx:end].sum())
        rank_sum += positive_count * avg_rank
        idx = end

    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = 0.0
    fp = 0.0
    ap = 0.0
    for label in y_true:
        if label == 1:
            tp += 1
            ap += tp / max(tp + fp, 1.0)
        else:
            fp += 1
    return float(ap / positives)


def choose_alarm_threshold(y_true_fail: np.ndarray, p_success: np.ndarray) -> Dict[str, float]:
    fail_score = 1.0 - p_success
    thresholds = np.unique(np.round(fail_score, 6))
    if thresholds.size > 400:
        thresholds = np.linspace(float(fail_score.min()), float(fail_score.max()), 400)

    best = None
    for threshold in thresholds:
        pred_fail = (fail_score >= threshold).astype(np.int64)
        tp = int(((pred_fail == 1) & (y_true_fail == 1)).sum())
        tn = int(((pred_fail == 0) & (y_true_fail == 0)).sum())
        fp = int(((pred_fail == 1) & (y_true_fail == 0)).sum())
        fn = int(((pred_fail == 0) & (y_true_fail == 1)).sum())

        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        recall = tpr
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        bal_acc = 0.5 * (tpr + tnr)

        candidate = {
            "threshold": float(threshold),
            "balanced_accuracy": float(bal_acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(tnr),
        }
        if best is None or (
            candidate["balanced_accuracy"],
            candidate["f1"],
            candidate["recall"],
            -candidate["threshold"],
        ) > (
            best["balanced_accuracy"],
            best["f1"],
            best["recall"],
            -best["threshold"],
        ):
            best = candidate
    if best is None:
        raise RuntimeError("Unable to choose threshold")
    return best


class LinearAlarmModel:
    def __init__(
        self,
        num_features: int,
        learning_rate: float,
        epochs: int,
        row_weight: float,
        pairwise_weight: float,
        l2: float,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.row_weight = row_weight
        self.pairwise_weight = pairwise_weight
        self.l2 = l2
        self.weights = np.zeros(num_features, dtype=np.float64)
        self.bias = 0.0
        self.best_epoch = 0
        self.best_val_auc = float("-inf")
        self.history: List[Dict[str, float]] = []

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict_success_prob(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.predict_score(x))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val_pure: np.ndarray,
        y_val_pure_fail: np.ndarray,
        pair_diffs: np.ndarray,
        pair_labels: np.ndarray,
        pair_weights: np.ndarray,
    ) -> None:
        m_w = np.zeros_like(self.weights)
        v_w = np.zeros_like(self.weights)
        m_b = 0.0
        v_b = 0.0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        best_weights = self.weights.copy()
        best_bias = self.bias

        pair_weight_total = max(float(pair_weights.sum()), 1e-8)
        n_rows = max(len(y_train), 1)

        for epoch in range(1, self.epochs + 1):
            train_scores = self.predict_score(x_train)
            p_train = sigmoid(train_scores)
            row_residual = (p_train - y_train) / n_rows
            grad_w = self.row_weight * (x_train.T @ row_residual)
            grad_b = self.row_weight * float(row_residual.sum())

            row_loss = soft_bce_loss(y_train, p_train)

            pair_loss = 0.0
            if pair_diffs.shape[0] > 0:
                pair_scores = sigmoid(pair_diffs @ self.weights)
                pair_residual = pair_weights * (pair_scores - pair_labels) / pair_weight_total
                grad_w += self.pairwise_weight * (pair_diffs.T @ pair_residual)
                pair_loss = pairwise_bce_loss(pair_labels, pair_scores, pair_weights)

            grad_w += self.l2 * self.weights

            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * (grad_w ** 2)
            m_b = beta1 * m_b + (1.0 - beta1) * grad_b
            v_b = beta2 * v_b + (1.0 - beta2) * (grad_b ** 2)

            m_w_hat = m_w / (1.0 - beta1**epoch)
            v_w_hat = v_w / (1.0 - beta2**epoch)
            m_b_hat = m_b / (1.0 - beta1**epoch)
            v_b_hat = v_b / (1.0 - beta2**epoch)

            self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.bias -= self.learning_rate * m_b_hat / (math.sqrt(v_b_hat) + eps)

            val_scores = self.predict_score(x_val_pure)
            val_auc = roc_auc_score_binary(y_val_pure_fail, -val_scores)
            total_loss = self.row_weight * row_loss + self.pairwise_weight * pair_loss

            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_row_bce": float(row_loss),
                    "train_pair_bce": float(pair_loss),
                    "train_total_loss": float(total_loss),
                    "val_auc": float(val_auc),
                }
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc = float(val_auc)
                self.best_epoch = epoch
                best_weights = self.weights.copy()
                best_bias = self.bias

        self.weights = best_weights
        self.bias = best_bias

    def observed_fisher_information(
        self,
        pair_diffs: np.ndarray,
        pair_weights: np.ndarray,
        row_features: np.ndarray | None = None,
    ) -> np.ndarray:
        dim = len(self.weights)
        fisher = np.eye(dim, dtype=np.float64) * max(self.l2, 1e-6)

        if pair_diffs.shape[0] > 0:
            pair_scores = sigmoid(pair_diffs @ self.weights)
            curvature = self.pairwise_weight * pair_weights * pair_scores * (1.0 - pair_scores)
            fisher += pair_diffs.T @ (pair_diffs * curvature[:, None])

        if row_features is not None and row_features.shape[0] > 0 and self.row_weight > 0:
            row_scores = self.predict_score(row_features)
            row_probs = sigmoid(row_scores)
            row_curvature = self.row_weight * row_probs * (1.0 - row_probs)
            fisher += row_features.T @ (row_features * row_curvature[:, None])

        return fisher

    def parameter_covariance(
        self,
        pair_diffs: np.ndarray,
        pair_weights: np.ndarray,
        row_features: np.ndarray | None = None,
    ) -> np.ndarray:
        fisher = self.observed_fisher_information(pair_diffs, pair_weights, row_features)
        return np.linalg.pinv(fisher)

    def score_standard_error(self, x: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[None, :]
        variances = np.einsum("ij,jk,ik->i", x, covariance, x)
        return np.sqrt(np.clip(variances, 0.0, None))

class LogisticCalibrator:
    def __init__(self, learning_rate: float, epochs: int, l2: float) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.scale = 1.0
        self.bias = 0.0
        self.best_loss = float("inf")
        self.history: List[Dict[str, float]] = []

    def fit(self, scores: np.ndarray, targets: np.ndarray) -> "LogisticCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)

        m_scale = 0.0
        v_scale = 0.0
        m_bias = 0.0
        v_bias = 0.0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        best_scale = self.scale
        best_bias = self.bias

        for epoch in range(1, self.epochs + 1):
            logits = self.scale * scores + self.bias
            preds = sigmoid(logits)
            residual = preds - targets

            grad_scale = float(np.mean(residual * scores) + self.l2 * self.scale)
            grad_bias = float(np.mean(residual))
            loss = soft_bce_loss(targets, preds) + 0.5 * self.l2 * self.scale * self.scale
            self.history.append({"epoch": float(epoch), "loss": float(loss)})

            if loss < self.best_loss:
                self.best_loss = float(loss)
                best_scale = self.scale
                best_bias = self.bias

            m_scale = beta1 * m_scale + (1.0 - beta1) * grad_scale
            v_scale = beta2 * v_scale + (1.0 - beta2) * (grad_scale ** 2)
            m_bias = beta1 * m_bias + (1.0 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1.0 - beta2) * (grad_bias ** 2)

            m_scale_hat = m_scale / (1.0 - beta1**epoch)
            v_scale_hat = v_scale / (1.0 - beta2**epoch)
            m_bias_hat = m_bias / (1.0 - beta1**epoch)
            v_bias_hat = v_bias / (1.0 - beta2**epoch)

            self.scale -= self.learning_rate * m_scale_hat / (math.sqrt(v_scale_hat) + eps)
            self.bias -= self.learning_rate * m_bias_hat / (math.sqrt(v_bias_hat) + eps)
            self.scale = max(self.scale, 1e-6)

        self.scale = best_scale
        self.bias = best_bias
        return self

    def predict_success_prob(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        return sigmoid(self.scale * scores + self.bias)

    def success_threshold_to_score_threshold(self, success_threshold: float) -> float:
        return float((logit(success_threshold) - self.bias) / self.scale)

    def score_threshold_to_success_threshold(self, score_threshold: float) -> float:
        return float(self.predict_success_prob(np.asarray([score_threshold]))[0])

    def to_dict(self) -> Dict[str, float]:
        return {
            "scale": float(self.scale),
            "bias": float(self.bias),
            "best_loss": float(self.best_loss),
        }

def threshold_candidates(values: np.ndarray, max_candidates: int = 120) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.0], dtype=np.float64)

    thresholds = np.unique(np.round(arr, 6))
    if thresholds.size > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        thresholds = np.unique(np.quantile(arr, quantiles))
    return thresholds.astype(np.float64)


def confusion_counts(y_fail: np.ndarray, pred_fail: np.ndarray) -> Tuple[int, int, int, int]:
    y_fail = np.asarray(y_fail, dtype=np.int64)
    pred_fail = np.asarray(pred_fail, dtype=np.int64)
    tp = int(((pred_fail == 1) & (y_fail == 1)).sum())
    tn = int(((pred_fail == 0) & (y_fail == 0)).sum())
    fp = int(((pred_fail == 1) & (y_fail == 0)).sum())
    fn = int(((pred_fail == 0) & (y_fail == 1)).sum())
    return tp, tn, fp, fn


def threshold_selection_key(candidate: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    return (
        candidate["balanced_accuracy"],
        min(candidate["recall"], candidate["specificity"]),
        candidate["f1"],
        candidate["specificity"],
        candidate["recall"],
    )


def choose_score_threshold(
    y_true_fail: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    thresholds = threshold_candidates(scores)
    best = None

    for threshold in thresholds:
        pred_fail = (scores < threshold).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(y_true_fail, pred_fail)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        bal_acc = 0.5 * (recall + specificity)
        candidate = {
            "threshold": float(threshold),
            "balanced_accuracy": float(bal_acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "alarm_rate": float(pred_fail.mean()),
        }
        if best is None or threshold_selection_key(candidate) > threshold_selection_key(best):
            best = candidate

    if best is None:
        raise RuntimeError("Unable to choose score threshold")
    return best


def binary_metrics(y_fail: np.ndarray, p_success: np.ndarray, threshold: float) -> Dict[str, float]:
    fail_score = 1.0 - p_success
    pred_fail = (fail_score >= threshold).astype(np.int64)
    return monitor_metrics(y_fail, p_success, pred_fail)


def monitor_metrics(
    y_fail: np.ndarray,
    p_success: np.ndarray,
    pred_fail: np.ndarray,
) -> Dict[str, float]:
    fail_score = 1.0 - p_success
    tp, tn, fp, fn = confusion_counts(y_fail, pred_fail)

    accuracy = (tp + tn) / max(len(y_fail), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    bal_acc = 0.5 * (recall + specificity)
    auc = roc_auc_score_binary(y_fail, fail_score)
    ap = average_precision_binary(y_fail, fail_score)
    brier = float(np.mean((p_success - (1.0 - y_fail)) ** 2))
    eps = 1e-8
    clipped = np.clip(p_success, eps, 1.0 - eps)
    log_loss = float(
        -np.mean((1.0 - y_fail) * np.log(clipped) + y_fail * np.log(1.0 - clipped))
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "brier": float(brier),
        "log_loss": float(log_loss),
        "alarm_rate": float(np.mean(pred_fail)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def apply_monitor_rule(
    lcb_scores: np.ndarray,
    score_drops: np.ndarray,
    lcb_threshold: float,
    drop_threshold: float,
    calibrated_success: np.ndarray | None = None,
    success_threshold: float | None = None,
    gate_by_success: bool = False,
) -> Dict[str, np.ndarray]:
    low_lcb = np.asarray(lcb_scores < lcb_threshold, dtype=np.int64)
    sharp_drop = np.asarray(
        np.isfinite(score_drops) & (score_drops < -drop_threshold), dtype=np.int64
    )

    low_calibrated_success = np.zeros_like(low_lcb)
    if calibrated_success is not None and success_threshold is not None:
        low_calibrated_success = np.asarray(
            np.asarray(calibrated_success) < success_threshold, dtype=np.int64
        )

    pred_fail = np.asarray((low_lcb == 1) | (sharp_drop == 1), dtype=np.int64)
    if gate_by_success:
        if calibrated_success is None or success_threshold is None:
            raise ValueError(
                "gate_by_success=True requires calibrated_success and success_threshold"
            )
        pred_fail = np.asarray(
            (low_calibrated_success == 1) & (pred_fail == 1), dtype=np.int64
        )

    return {
        "pred_fail": pred_fail,
        "low_calibrated_success": low_calibrated_success,
        "low_lcb": low_lcb,
        "sharp_drop": sharp_drop,
    }


def choose_composite_thresholds(
    y_true_fail: np.ndarray,
    lcb_scores: np.ndarray,
    score_drops: np.ndarray,
) -> Dict[str, float]:
    lcb_thresholds = threshold_candidates(lcb_scores)
    finite_drops = score_drops[np.isfinite(score_drops)]
    drop_thresholds = threshold_candidates(np.clip(-finite_drops, 0.0, None))
    drop_thresholds = np.unique(np.concatenate([np.asarray([0.0]), drop_thresholds]))

    low_lcb_flags = [(float(threshold), (lcb_scores < threshold).astype(np.int64)) for threshold in lcb_thresholds]
    sharp_drop_flags = [
        (float(threshold), np.asarray(np.isfinite(score_drops) & (score_drops < -threshold), dtype=np.int64))
        for threshold in drop_thresholds
    ]

    best = None
    for lcb_threshold, low_lcb in low_lcb_flags:
        for drop_threshold, sharp_drop in sharp_drop_flags:
            pred_fail = np.asarray((low_lcb == 1) | (sharp_drop == 1), dtype=np.int64)
            tp, tn, fp, fn = confusion_counts(y_true_fail, pred_fail)
            recall = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
            bal_acc = 0.5 * (recall + specificity)
            candidate = {
                "lcb_threshold": float(lcb_threshold),
                "drop_threshold": float(drop_threshold),
                "balanced_accuracy": float(bal_acc),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "alarm_rate": float(pred_fail.mean()),
            }
            if best is None or threshold_selection_key(candidate) > threshold_selection_key(best):
                best = candidate

    if best is None:
        raise RuntimeError("Unable to choose composite thresholds")
    return best


def safe_sort_metric(value: float) -> float:
    return float(value) if np.isfinite(value) else -1.0


def leaf_low_fp_selection_key(candidate: Dict[str, float]) -> Tuple[float, float, float, float, float, float, float, float]:
    return (
        -candidate["pure1_leaf_false_alarm_rate"],
        candidate["pure0_leaf_alarm_recall"],
        candidate["balanced_accuracy"],
        candidate["specificity"],
        safe_sort_metric(candidate.get("pure0_successful_warning_mean_early_pct", float("nan"))),
        candidate.get("drop_threshold", 0.0),
        -candidate.get("success_threshold", 0.0),
        -candidate.get("lcb_threshold", 0.0),
    )


def leaf_fallback_selection_key(candidate: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    return (
        candidate["pure0_leaf_alarm_recall"] - candidate["pure1_leaf_false_alarm_rate"],
        candidate["balanced_accuracy"],
        candidate["specificity"],
        candidate["recall"],
        safe_sort_metric(candidate.get("pure0_successful_warning_mean_early_pct", float("nan"))),
        -candidate.get("success_threshold", 0.0),
    )


def choose_leaf_aware_success_threshold(
    y_true_fail: np.ndarray,
    p_success: np.ndarray,
    trajectory_index: Sequence[Dict[str, object]],
    target_leaf_recall: float,
    max_candidates: int = 48,
) -> Dict[str, float]:
    thresholds = threshold_candidates(np.asarray(p_success), max_candidates=max_candidates)
    best = None
    fallback = None

    for success_threshold in thresholds:
        pred_fail = np.asarray(np.asarray(p_success) < success_threshold, dtype=np.int64)
        candidate = monitor_metrics(y_true_fail, p_success, pred_fail)
        candidate.update(trajectory_alarm_metrics_from_index(pred_fail, trajectory_index))
        candidate.update(
            {
                "success_threshold": float(success_threshold),
                "alarm_score_threshold": float(1.0 - success_threshold),
                "selection_target_leaf_recall": float(target_leaf_recall),
                "selection_mode": "min_leaf_false_alarm_subject_to_leaf_recall",
            }
        )

        if fallback is None or leaf_fallback_selection_key(candidate) > leaf_fallback_selection_key(fallback):
            fallback = candidate
        if candidate["pure0_leaf_alarm_recall"] >= target_leaf_recall:
            if best is None or leaf_low_fp_selection_key(candidate) > leaf_low_fp_selection_key(best):
                best = candidate

    if best is None:
        if fallback is None:
            raise RuntimeError("Unable to choose a leaf-aware success threshold")
        best = dict(fallback)
        best["selection_mode"] = "fallback_best_leaf_tradeoff"
    return best


def choose_leaf_aware_composite_thresholds(
    y_true_fail: np.ndarray,
    p_success: np.ndarray,
    lcb_scores: np.ndarray,
    score_drops: np.ndarray,
    trajectory_index: Sequence[Dict[str, object]],
    success_threshold: float,
    target_leaf_recall: float,
    max_candidates: int = 48,
    gate_by_success: bool = True,
) -> Dict[str, float]:
    lcb_thresholds = threshold_candidates(lcb_scores, max_candidates=max_candidates)
    finite_drops = score_drops[np.isfinite(score_drops)]
    drop_thresholds = threshold_candidates(
        np.clip(-finite_drops, 0.0, None),
        max_candidates=max_candidates,
    )
    drop_thresholds = np.unique(np.concatenate([np.asarray([0.0]), drop_thresholds]))

    low_calibrated_success = np.asarray(np.asarray(p_success) < success_threshold, dtype=np.int64)
    low_lcb_flags = [
        (float(threshold), np.asarray(lcb_scores < threshold, dtype=np.int64))
        for threshold in lcb_thresholds
    ]
    sharp_drop_flags = [
        (
            float(threshold),
            np.asarray(np.isfinite(score_drops) & (score_drops < -threshold), dtype=np.int64),
        )
        for threshold in drop_thresholds
    ]

    best = None
    fallback = None
    for lcb_threshold, low_lcb in low_lcb_flags:
        for drop_threshold, sharp_drop in sharp_drop_flags:
            pred_fail = np.asarray((low_lcb == 1) | (sharp_drop == 1), dtype=np.int64)
            if gate_by_success:
                pred_fail = np.asarray(
                    (low_calibrated_success == 1) & (pred_fail == 1), dtype=np.int64
                )

            candidate = monitor_metrics(y_true_fail, p_success, pred_fail)
            candidate.update(trajectory_alarm_metrics_from_index(pred_fail, trajectory_index))
            candidate.update(
                {
                    "lcb_threshold": float(lcb_threshold),
                    "drop_threshold": float(drop_threshold),
                    "success_threshold": float(success_threshold),
                    "alarm_score_threshold": float(1.0 - success_threshold),
                    "selection_target_leaf_recall": float(target_leaf_recall),
                    "gate_by_success": int(gate_by_success),
                    "selection_mode": "min_leaf_false_alarm_subject_to_leaf_recall",
                }
            )

            if fallback is None or leaf_fallback_selection_key(candidate) > leaf_fallback_selection_key(fallback):
                fallback = candidate
            if candidate["pure0_leaf_alarm_recall"] >= target_leaf_recall:
                if best is None or leaf_low_fp_selection_key(candidate) > leaf_low_fp_selection_key(best):
                    best = candidate

    if best is None:
        if fallback is None:
            raise RuntimeError("Unable to choose leaf-aware composite thresholds")
        best = dict(fallback)
        best["selection_mode"] = "fallback_best_leaf_tradeoff"
    return best


def build_node_final_score_map(

    instances: Dict[str, InstanceData],
    featurizer: PrefixFeaturizer,
    window_size: int,
    scaler: Standardizer,
    model: LinearAlarmModel,
    feature_names: Sequence[str] | None = None,
    drop_confidence_features: bool = False,
) -> Dict[Tuple[str, str], float]:
    keys: List[Tuple[str, str]] = []
    feature_vectors: List[np.ndarray] = []

    for instance in instances.values():
        for node_id in sorted_node_ids(instance.rows_by_node):
            path_rows = instance.path_rows_by_node[node_id]
            window_rows = path_rows[max(0, len(path_rows) - window_size) :]
            feature_vectors.append(featurizer.transform_window(window_rows, window_size))
            keys.append((instance.instance_id, node_id))

    if not keys:
        return {}

    stacked = np.vstack(feature_vectors).astype(np.float64)
    if feature_names is not None and drop_confidence_features:
        keep_indices = [
            idx for idx, name in enumerate(feature_names) if "confidence" not in name.lower()
        ]
        if not keep_indices:
            raise RuntimeError("Dropping confidence features removed every feature column")
        stacked = stacked[:, np.asarray(keep_indices, dtype=np.int64)]
    x = scaler.transform(stacked)
    scores = model.predict_score(x)
    return {key: float(score) for key, score in zip(keys, scores)}


def compute_parent_scores(
    samples: Sequence[PrefixSample],
    node_final_score_map: Dict[Tuple[str, str], float],
) -> np.ndarray:
    parent_scores = np.full(len(samples), np.nan, dtype=np.float64)
    for idx, sample in enumerate(samples):
        parent = node_parent(sample.node_id)
        if parent is None:
            continue
        key = (sample.instance_id, parent)
        if key in node_final_score_map:
            parent_scores[idx] = node_final_score_map[key]
    return parent_scores


def build_trajectory_index(
    samples: Sequence[PrefixSample],
    instances: Dict[str, InstanceData],
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for sample_idx, sample in enumerate(samples):
        grouped[sample.instance_id][sample.node_id].append((sample.step_idx, sample_idx))

    trajectory_index: List[Dict[str, object]] = []
    for instance_id, by_node in grouped.items():
        instance = instances[instance_id]
        leaf_nodes = [
            node_id for node_id in instance.rows_by_node if node_id not in instance.children_by_parent
        ]
        for leaf_node in leaf_nodes:
            path_nodes: List[str] = []
            current = leaf_node
            while current is not None:
                path_nodes.append(current)
                current = node_parent(current)
            path_nodes.reverse()

            path_events: List[Tuple[int, int]] = []
            for node_id in path_nodes:
                path_events.extend(by_node.get(node_id, []))
            path_events.sort(key=lambda item: item[0])

            trajectory_index.append(
                {
                    "root_kind": instance.root_kind,
                    "sample_indices": np.asarray([sample_idx for _, sample_idx in path_events], dtype=np.int64),
                    "path_steps": np.asarray([step for step, _ in path_events], dtype=np.int64),
                    "final_step": int(instance.rows_by_node[leaf_node][-1].step_idx),
                }
            )

    return trajectory_index


def trajectory_alarm_metrics_from_index(
    pred_fail: np.ndarray,
    trajectory_index: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    pred_fail = np.asarray(pred_fail, dtype=np.int64)

    pure0_total = 0
    pure1_total = 0
    pure0_detected = 0
    pure1_false = 0
    pure0_first_steps: List[int] = []
    pure1_first_steps: List[int] = []
    pure0_early_warning_pct: List[float] = []

    for item in trajectory_index:
        sample_indices = item["sample_indices"]
        path_steps = item["path_steps"]
        hit_positions = np.flatnonzero(pred_fail[sample_indices] == 1)
        first_alarm = None
        if hit_positions.size > 0:
            first_alarm = int(path_steps[int(hit_positions[0])])

        if item["root_kind"] == "pure0":
            pure0_total += 1
            if first_alarm is not None:
                pure0_detected += 1
                pure0_first_steps.append(first_alarm)
                pure0_early_warning_pct.append(
                    100.0 * (int(item["final_step"]) - first_alarm) / max(int(item["final_step"]), 1)
                )
        elif item["root_kind"] == "pure1":
            pure1_total += 1
            if first_alarm is not None:
                pure1_false += 1
                pure1_first_steps.append(first_alarm)

    return {
        "pure0_leaf_alarm_recall": pure0_detected / max(pure0_total, 1),
        "pure1_leaf_false_alarm_rate": pure1_false / max(pure1_total, 1),
        "pure0_leaf_alarm_median_step": float(np.median(pure0_first_steps))
        if pure0_first_steps
        else float("nan"),
        "pure1_leaf_false_alarm_median_step": float(np.median(pure1_first_steps))
        if pure1_first_steps
        else float("nan"),
        "pure0_leaf_paths": float(pure0_total),
        "pure1_leaf_paths": float(pure1_total),
        "pure0_successful_warning_mean_early_pct": float(np.mean(pure0_early_warning_pct))
        if pure0_early_warning_pct
        else float("nan"),
    }


def cusum_running_values(path_fail_scores: np.ndarray, cusum_drift: float) -> np.ndarray:
    running = np.zeros(len(path_fail_scores), dtype=np.float64)
    stat = 0.0
    for idx, score in enumerate(path_fail_scores):
        stat = max(0.0, stat + float(score) - float(cusum_drift))
        running[idx] = stat
    return running


def trajectory_cusum_metrics_from_precomputed(
    running_values_by_path: Sequence[np.ndarray],
    trajectory_index: Sequence[Dict[str, object]],
    cusum_threshold: float,
) -> Dict[str, float]:
    pure0_total = 0
    pure1_total = 0
    pure0_detected = 0
    pure1_false = 0
    pure0_first_steps: List[int] = []
    pure1_first_steps: List[int] = []
    pure0_early_warning_pct: List[float] = []

    for item, running in zip(trajectory_index, running_values_by_path):
        path_steps = item["path_steps"]
        hit_positions = np.flatnonzero(running >= cusum_threshold)
        first_alarm = None
        if hit_positions.size > 0:
            first_alarm = int(path_steps[int(hit_positions[0])])

        if item["root_kind"] == "pure0":
            pure0_total += 1
            if first_alarm is not None:
                pure0_detected += 1
                pure0_first_steps.append(first_alarm)
                pure0_early_warning_pct.append(
                    100.0 * (int(item["final_step"]) - first_alarm) / max(int(item["final_step"]), 1)
                )
        elif item["root_kind"] == "pure1":
            pure1_total += 1
            if first_alarm is not None:
                pure1_false += 1
                pure1_first_steps.append(first_alarm)

    return {
        "pure0_leaf_alarm_recall": pure0_detected / max(pure0_total, 1),
        "pure1_leaf_false_alarm_rate": pure1_false / max(pure1_total, 1),
        "pure0_leaf_alarm_median_step": float(np.median(pure0_first_steps))
        if pure0_first_steps
        else float("nan"),
        "pure1_leaf_false_alarm_median_step": float(np.median(pure1_first_steps))
        if pure1_first_steps
        else float("nan"),
        "pure0_leaf_paths": float(pure0_total),
        "pure1_leaf_paths": float(pure1_total),
        "pure0_successful_warning_mean_early_pct": float(np.mean(pure0_early_warning_pct))
        if pure0_early_warning_pct
        else float("nan"),
    }


def trajectory_cusum_metrics_from_index(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Dict[str, object]],
    cusum_drift: float,
    cusum_threshold: float,
) -> Dict[str, float]:
    fail_scores = np.asarray(fail_scores, dtype=np.float64)
    running_values_by_path = [
        cusum_running_values(fail_scores[item["sample_indices"]], cusum_drift)
        for item in trajectory_index
    ]
    return trajectory_cusum_metrics_from_precomputed(
        running_values_by_path,
        trajectory_index,
        cusum_threshold,
    )


def cusum_selection_key(candidate: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        -candidate["pure1_leaf_false_alarm_rate"],
        candidate["pure0_leaf_alarm_recall"],
        safe_sort_metric(candidate["pure0_successful_warning_mean_early_pct"]),
        -candidate["cusum_threshold"],
    )


def cusum_fallback_key(candidate: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        candidate["pure0_leaf_alarm_recall"] - candidate["pure1_leaf_false_alarm_rate"],
        safe_sort_metric(candidate["pure0_successful_warning_mean_early_pct"]),
        -candidate["cusum_threshold"],
    )


def choose_leaf_aware_cusum_thresholds(
    fail_scores: np.ndarray,
    trajectory_index: Sequence[Dict[str, object]],
    target_leaf_recall: float,
    max_candidates: int = 48,
) -> Dict[str, float]:
    fail_scores = np.asarray(fail_scores, dtype=np.float64)
    drift_candidates = threshold_candidates(
        fail_scores,
        max_candidates=max(8, min(max_candidates // 2, 20)),
    )
    drift_candidates = drift_candidates[(drift_candidates > 0.0) & (drift_candidates < 1.0)]
    drift_candidates = np.unique(
        np.concatenate(
            [
                drift_candidates,
                np.asarray([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),
            ]
        )
    )
    drift_candidates = drift_candidates[(drift_candidates > 0.0) & (drift_candidates < 1.0)]
    if drift_candidates.size == 0:
        drift_candidates = np.asarray([0.5], dtype=np.float64)

    best = None
    fallback = None
    for cusum_drift in drift_candidates:
        running_values_by_path = [
            cusum_running_values(fail_scores[item["sample_indices"]], float(cusum_drift))
            for item in trajectory_index
        ]
        max_running_values = np.asarray(
            [float(running.max()) if running.size > 0 else 0.0 for running in running_values_by_path],
            dtype=np.float64,
        )
        thresholds = threshold_candidates(max_running_values, max_candidates=max_candidates)
        thresholds = thresholds[thresholds > 0.0]
        if thresholds.size == 0:
            thresholds = np.asarray([0.0], dtype=np.float64)

        for cusum_threshold in thresholds:
            candidate = trajectory_cusum_metrics_from_precomputed(
                running_values_by_path,
                trajectory_index,
                float(cusum_threshold),
            )
            candidate.update(
                {
                    "cusum_drift": float(cusum_drift),
                    "cusum_threshold": float(cusum_threshold),
                    "selection_target_leaf_recall": float(target_leaf_recall),
                    "selection_mode": "min_leaf_false_alarm_subject_to_leaf_recall",
                }
            )
            if fallback is None or cusum_fallback_key(candidate) > cusum_fallback_key(fallback):
                fallback = candidate
            if candidate["pure0_leaf_alarm_recall"] >= target_leaf_recall:
                if best is None or cusum_selection_key(candidate) > cusum_selection_key(best):
                    best = candidate

    if best is None:
        if fallback is None:
            raise RuntimeError("Unable to choose leaf-aware CUSUM thresholds")
        best = dict(fallback)
        best["selection_mode"] = "fallback_best_leaf_tradeoff"
    return best


def trajectory_alarm_metrics(
    samples: Sequence[PrefixSample],
    pred_fail: np.ndarray,
    instances: Dict[str, InstanceData],
) -> Dict[str, float]:
    trajectory_index = build_trajectory_index(samples, instances)
    return trajectory_alarm_metrics_from_index(pred_fail, trajectory_index)


def save_predictions_csv(

    path: Path,
    samples: Sequence[PrefixSample],
    p_success: np.ndarray,
    raw_scores: np.ndarray,
    lcb_scores: np.ndarray,
    parent_scores: np.ndarray,
    score_drops: np.ndarray,
    low_calibrated_success: np.ndarray,
    low_lcb: np.ndarray,
    sharp_drop: np.ndarray,
    pred_fail: np.ndarray,
) -> None:
    fieldnames = [
        "instance_id",
        "root_kind",
        "node_id",
        "step_idx",
        "target_y",
        "is_final_step",
        "is_root_prefix",
        "pred_success",
        "pred_alarm",
        "raw_score",
        "score_lcb",
        "parent_score",
        "score_drop_from_parent",
        "rule_low_calibrated_success",
        "rule_low_lcb",
        "rule_sharp_drop",
        "pred_alarm_flag",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (
            sample,
            pred_success,
            raw_score,
            lcb_score,
            parent_score,
            score_drop,
            low_cal,
            low_lcb_flag,
            sharp_drop_flag,
            alarm_flag,
        ) in zip(
            samples,
            p_success,
            raw_scores,
            lcb_scores,
            parent_scores,
            score_drops,
            low_calibrated_success,
            low_lcb,
            sharp_drop,
            pred_fail,
        ):
            writer.writerow(
                {
                    "instance_id": sample.instance_id,
                    "root_kind": sample.root_kind,
                    "node_id": sample.node_id,
                    "step_idx": sample.step_idx,
                    "target_y": f"{sample.target_y:.6f}",
                    "is_final_step": int(sample.is_final_step),
                    "is_root_prefix": int(sample.is_root_prefix),
                    "pred_success": f"{float(pred_success):.6f}",
                    "pred_alarm": f"{1.0 - float(pred_success):.6f}",
                    "raw_score": f"{float(raw_score):.6f}",
                    "score_lcb": f"{float(lcb_score):.6f}",
                    "parent_score": f"{float(parent_score):.6f}",
                    "score_drop_from_parent": f"{float(score_drop):.6f}",
                    "rule_low_calibrated_success": int(low_cal),
                    "rule_low_lcb": int(low_lcb_flag),
                    "rule_sharp_drop": int(sharp_drop_flag),
                    "pred_alarm_flag": int(alarm_flag),
                }
            )

def save_feature_weights(path: Path, feature_names: Sequence[str], weights: np.ndarray) -> None:
    pairs = sorted(zip(feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "weight"])
        for feature, weight in pairs:
            writer.writerow([feature, f"{float(weight):.8f}"])


def summarize_samples(samples: Sequence[PrefixSample]) -> Dict[str, float]:
    root_kind_counts = Counter(sample.root_kind for sample in samples)
    return {
        "num_samples": float(len(samples)),
        "mixed_samples": float(root_kind_counts.get("mixed", 0)),
        "pure0_samples": float(root_kind_counts.get("pure0", 0)),
        "pure1_samples": float(root_kind_counts.get("pure1", 0)),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    row_weight = float(getattr(args, "row_weight", 0.2))
    pairwise_weight = float(getattr(args, "pairwise_weight", 1.0))
    calibration_learning_rate = float(getattr(args, "calibration_learning_rate", 0.05))
    calibration_epochs = int(getattr(args, "calibration_epochs", 400))
    calibration_l2 = float(getattr(args, "calibration_l2", 1e-3))
    confidence_z = float(getattr(args, "confidence_z", 1.96))
    target_leaf_recall = float(getattr(args, "target_leaf_recall", 0.85))
    rule_threshold_max_candidates = int(getattr(args, "rule_threshold_max_candidates", 48))
    drop_confidence_features = bool(getattr(args, "drop_confidence_features", False))

    instances = load_all_instances(data_dir)
    if not instances:
        raise RuntimeError(f"No CSV files found under {data_dir}")

    split_spec = stratified_pure_split(instances, args.pure_val_ratio, args.seed)
    featurizer = PrefixFeaturizer()
    bundle = build_samples(
        instances=instances,
        featurizer=featurizer,
        split_spec=split_spec,
        window_size=args.window_size,
        stride=args.stride,
        min_step=args.min_step,
        min_pair_gap=args.min_pair_gap,
    )

    x_train_raw, y_train = samples_to_arrays(bundle.train_samples)
    x_val_raw, _ = samples_to_arrays(bundle.val_samples)
    x_test_raw, _ = samples_to_arrays(bundle.test_samples)

    feature_names, x_train_raw, x_val_raw, x_test_raw, pair_diffs_raw, dropped_feature_names = filter_feature_matrices(
        bundle.feature_names,
        x_train_raw,
        x_val_raw,
        x_test_raw,
        bundle.pair_diffs,
        drop_confidence_features=drop_confidence_features,
    )

    scaler = Standardizer().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)
    pair_diffs = pair_diffs_raw / scaler.scale_

    y_val_fail = np.asarray([1 if sample.root_kind == "pure0" else 0 for sample in bundle.val_samples])
    y_test_fail = np.asarray([1 if sample.root_kind == "pure0" else 0 for sample in bundle.test_samples])

    model = LinearAlarmModel(
        num_features=x_train.shape[1],
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        row_weight=row_weight,
        pairwise_weight=pairwise_weight,
        l2=args.l2,
    )
    model.fit(
        x_train=x_train,
        y_train=y_train,
        x_val_pure=x_val,
        y_val_pure_fail=y_val_fail,
        pair_diffs=pair_diffs,
        pair_labels=bundle.pair_labels,
        pair_weights=bundle.pair_weights,
    )

    train_raw_scores = model.predict_score(x_train)
    val_raw_scores = model.predict_score(x_val)
    test_raw_scores = model.predict_score(x_test)

    calibrator = LogisticCalibrator(
        learning_rate=calibration_learning_rate,
        epochs=calibration_epochs,
        l2=calibration_l2,
    ).fit(train_raw_scores, y_train)

    train_p = calibrator.predict_success_prob(train_raw_scores)
    val_p = calibrator.predict_success_prob(val_raw_scores)
    test_p = calibrator.predict_success_prob(test_raw_scores)
    val_fail_scores = 1.0 - val_p
    test_fail_scores = 1.0 - test_p

    parameter_covariance = model.parameter_covariance(
        pair_diffs,
        bundle.pair_weights,
        row_features=x_train,
    )
    train_score_se = model.score_standard_error(x_train, parameter_covariance)
    val_score_se = model.score_standard_error(x_val, parameter_covariance)
    test_score_se = model.score_standard_error(x_test, parameter_covariance)

    train_lcb_scores = train_raw_scores - confidence_z * train_score_se
    val_lcb_scores = val_raw_scores - confidence_z * val_score_se
    test_lcb_scores = test_raw_scores - confidence_z * test_score_se

    node_final_score_map = build_node_final_score_map(
        instances=instances,
        featurizer=featurizer,
        window_size=args.window_size,
        scaler=scaler,
        model=model,
        feature_names=bundle.feature_names,
        drop_confidence_features=drop_confidence_features,
    )
    val_parent_scores = compute_parent_scores(bundle.val_samples, node_final_score_map)
    test_parent_scores = compute_parent_scores(bundle.test_samples, node_final_score_map)
    val_score_drops = val_raw_scores - val_parent_scores
    test_score_drops = test_raw_scores - test_parent_scores

    val_trajectory_index = build_trajectory_index(bundle.val_samples, instances)
    test_trajectory_index = build_trajectory_index(bundle.test_samples, instances)

    calibrated_leaf_low_fp_info = choose_leaf_aware_success_threshold(
        y_true_fail=y_val_fail,
        p_success=val_p,
        trajectory_index=val_trajectory_index,
        target_leaf_recall=target_leaf_recall,
        max_candidates=rule_threshold_max_candidates,
    )
    final_success_threshold = calibrated_leaf_low_fp_info["success_threshold"]
    val_calibrated_leaf_low_fp_pred = np.asarray(val_p < final_success_threshold, dtype=np.int64)
    test_calibrated_leaf_low_fp_pred = np.asarray(test_p < final_success_threshold, dtype=np.int64)

    cusum_threshold_info = choose_leaf_aware_cusum_thresholds(
        fail_scores=val_fail_scores,
        trajectory_index=val_trajectory_index,
        target_leaf_recall=target_leaf_recall,
        max_candidates=rule_threshold_max_candidates,
    )
    val_cusum_metrics = trajectory_cusum_metrics_from_index(
        val_fail_scores,
        val_trajectory_index,
        cusum_threshold_info["cusum_drift"],
        cusum_threshold_info["cusum_threshold"],
    )
    test_cusum_metrics = trajectory_cusum_metrics_from_index(
        test_fail_scores,
        test_trajectory_index,
        cusum_threshold_info["cusum_drift"],
        cusum_threshold_info["cusum_threshold"],
    )

    gated_threshold_info = choose_leaf_aware_composite_thresholds(
        y_true_fail=y_val_fail,
        p_success=val_p,
        lcb_scores=val_lcb_scores,
        score_drops=val_score_drops,
        trajectory_index=val_trajectory_index,
        success_threshold=final_success_threshold,
        target_leaf_recall=target_leaf_recall,
        max_candidates=rule_threshold_max_candidates,
        gate_by_success=True,
    )
    gated_lcb_threshold = gated_threshold_info["lcb_threshold"]
    gated_drop_threshold = gated_threshold_info["drop_threshold"]
    val_gated_rule = apply_monitor_rule(
        lcb_scores=val_lcb_scores,
        score_drops=val_score_drops,
        lcb_threshold=gated_lcb_threshold,
        drop_threshold=gated_drop_threshold,
        calibrated_success=val_p,
        success_threshold=final_success_threshold,
        gate_by_success=True,
    )
    test_gated_rule = apply_monitor_rule(
        lcb_scores=test_lcb_scores,
        score_drops=test_score_drops,
        lcb_threshold=gated_lcb_threshold,
        drop_threshold=gated_drop_threshold,
        calibrated_success=test_p,
        success_threshold=final_success_threshold,
        gate_by_success=True,
    )

    leaf_rule_metrics = {
        "val": {
            "calibrated_leaf_low_fp": trajectory_alarm_metrics_from_index(val_calibrated_leaf_low_fp_pred, val_trajectory_index),
            "cusum_leaf_low_fp": val_cusum_metrics,
            "gated_leaf_low_fp": trajectory_alarm_metrics_from_index(val_gated_rule["pred_fail"], val_trajectory_index),
        },
        "test": {
            "calibrated_leaf_low_fp": trajectory_alarm_metrics_from_index(test_calibrated_leaf_low_fp_pred, test_trajectory_index),
            "cusum_leaf_low_fp": test_cusum_metrics,
            "gated_leaf_low_fp": trajectory_alarm_metrics_from_index(test_gated_rule["pred_fail"], test_trajectory_index),
        },
    }

    final_rule_name = "calibrated_leaf_low_fp"
    final_leaf_alarm = {
        "val": leaf_rule_metrics["val"][final_rule_name],
        "test": leaf_rule_metrics["test"][final_rule_name],
    }

    summary = {
        "config": {
            "data_dir": str(data_dir),
            "window_size": args.window_size,
            "stride": args.stride,
            "min_step": args.min_step,
            "pure_val_ratio": args.pure_val_ratio,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "row_weight": row_weight,
            "pairwise_weight": pairwise_weight,
            "l2": args.l2,
            "min_pair_gap": args.min_pair_gap,
            "calibration_learning_rate": calibration_learning_rate,
            "calibration_epochs": calibration_epochs,
            "calibration_l2": calibration_l2,
            "confidence_z": confidence_z,
            "target_leaf_recall": target_leaf_recall,
            "rule_threshold_max_candidates": rule_threshold_max_candidates,
            "drop_confidence_features": drop_confidence_features,
        },
        "counts": {
            "num_instances": len(instances),
            "mixed_train_files": len(bundle.splits.train_mixed),
            "val_pure_files": len(bundle.splits.val_pure),
            "test_pure_files": len(bundle.splits.test_pure),
            "train_samples": len(bundle.train_samples),
            "val_samples": len(bundle.val_samples),
            "test_samples": len(bundle.test_samples),
            "pairwise_examples": int(bundle.pair_diffs.shape[0]),
            "num_features_before_filter": len(bundle.feature_names),
            "num_features_after_filter": len(feature_names),
        },
        "feature_filter": {
            "dropped_features": dropped_feature_names,
        },
        "sample_summary": {
            "train": summarize_samples(bundle.train_samples),
            "val": summarize_samples(bundle.val_samples),
            "test": summarize_samples(bundle.test_samples),
        },
        "thresholds": {
            "calibrated_leaf_low_fp": calibrated_leaf_low_fp_info,
            "success_probability_threshold": float(final_success_threshold),
            "cusum_leaf_low_fp": cusum_threshold_info,
            "gated_leaf_low_fp": gated_threshold_info,
            "confidence_z": float(confidence_z),
        },
        "model": {
            "best_epoch": model.best_epoch,
            "best_val_auc": model.best_val_auc,
            "bias": model.bias,
            "calibrator": calibrator.to_dict(),
        },
        "metrics": {
            "train_soft_target": {
                "row_bce": soft_bce_loss(y_train, train_p),
                "pairwise_bce": pairwise_bce_loss(
                    bundle.pair_labels,
                    sigmoid(pair_diffs @ model.weights),
                    bundle.pair_weights,
                ) if pair_diffs.shape[0] > 0 else 0.0,
                "y_mean": float(y_train.mean()),
                "pred_mean": float(train_p.mean()),
            },
            "leaf_rule_comparison": leaf_rule_metrics,
            "final_rule": {
                "name": final_rule_name,
                "formula": "low_calibrated_success",
                "leaf_alarm": final_leaf_alarm,
            },
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_mixed": bundle.splits.train_mixed,
                "val_pure": bundle.splits.val_pure,
                "test_pure": bundle.splits.test_pure,
            },
            handle,
            indent=2,
        )

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(model.history, handle, indent=2)

    with (output_dir / "scaler.json").open("w", encoding="utf-8") as handle:
        json.dump(scaler.to_dict(), handle)

    with (output_dir / "calibrator.json").open("w", encoding="utf-8") as handle:
        json.dump(calibrator.to_dict(), handle, indent=2)

    save_predictions_csv(
        output_dir / "val_predictions.csv",
        bundle.val_samples,
        val_p,
        val_raw_scores,
        val_lcb_scores,
        val_parent_scores,
        val_score_drops,
        val_calibrated_leaf_low_fp_pred,
        np.zeros_like(val_calibrated_leaf_low_fp_pred),
        np.zeros_like(val_calibrated_leaf_low_fp_pred),
        val_calibrated_leaf_low_fp_pred,
    )
    save_predictions_csv(
        output_dir / "test_predictions.csv",
        bundle.test_samples,
        test_p,
        test_raw_scores,
        test_lcb_scores,
        test_parent_scores,
        test_score_drops,
        test_calibrated_leaf_low_fp_pred,
        np.zeros_like(test_calibrated_leaf_low_fp_pred),
        np.zeros_like(test_calibrated_leaf_low_fp_pred),
        test_calibrated_leaf_low_fp_pred,
    )
    save_feature_weights(output_dir / "feature_weights.csv", feature_names, model.weights)

    top_weights = sorted(
        zip(feature_names, model.weights), key=lambda item: abs(item[1]), reverse=True
    )[:20]
    with (output_dir / "top_weights.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [{"feature": feature, "weight": float(weight)} for feature, weight in top_weights],
            handle,
            indent=2,
        )

    return summary


# Compatibility shim so the step-by-step notebook cells can keep using `tam.*`.


tam = SimpleNamespace(
    load_instance=load_instance,
    load_all_instances=load_all_instances,
    stratified_pure_split=stratified_pure_split,
    should_sample_step=should_sample_step,
    build_samples=build_samples,
    samples_to_arrays=samples_to_arrays,
    filter_feature_matrices=filter_feature_matrices,
    PrefixFeaturizer=PrefixFeaturizer,
    Standardizer=Standardizer,
    soft_bce_loss=soft_bce_loss,
    pairwise_bce_loss=pairwise_bce_loss,
    LinearAlarmModel=LinearAlarmModel,
    LogisticCalibrator=LogisticCalibrator,
    choose_alarm_threshold=choose_alarm_threshold,
    choose_score_threshold=choose_score_threshold,
    choose_composite_thresholds=choose_composite_thresholds,
    choose_leaf_aware_success_threshold=choose_leaf_aware_success_threshold,
    choose_leaf_aware_composite_thresholds=choose_leaf_aware_composite_thresholds,
    choose_leaf_aware_cusum_thresholds=choose_leaf_aware_cusum_thresholds,
    binary_metrics=binary_metrics,
    monitor_metrics=monitor_metrics,
    apply_monitor_rule=apply_monitor_rule,
    build_trajectory_index=build_trajectory_index,
    trajectory_alarm_metrics_from_index=trajectory_alarm_metrics_from_index,
    trajectory_cusum_metrics_from_index=trajectory_cusum_metrics_from_index,
    trajectory_alarm_metrics=trajectory_alarm_metrics,
    build_node_final_score_map=build_node_final_score_map,
    compute_parent_scores=compute_parent_scores,
    roc_auc_score_binary=roc_auc_score_binary,
    average_precision_binary=average_precision_binary,
    save_predictions_csv=save_predictions_csv,
    save_feature_weights=save_feature_weights,
    summarize_samples=summarize_samples,
    run_experiment=run_experiment,
)

# Change DATA_DIR when you add the later 300 lite + 500 verified CSV files.
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "monitor_results" / "leaf_low_fp_monitor"

# Prefix construction.
WINDOW_SIZE = 64
STRIDE = 8
MIN_STEP = 9

# File-level split. Mixed files train the monitor; pure files test alarm behavior.
PURE_VAL_RATIO = 0.5
SEED = 7

# Model hyperparameters.
LEARNING_RATE = 0.03
EPOCHS = 250
ROW_WEIGHT = 0.2
PAIRWISE_WEIGHT = 1.0
L2 = 1e-3
MIN_PAIR_GAP = 0.05

# Calibration + uncertainty hyperparameters.
CALIBRATION_LEARNING_RATE = 0.05
CALIBRATION_EPOCHS = 400
CALIBRATION_L2 = 1e-3
CONFIDENCE_Z = 1.96

# Final alarm rule tuning.
TARGET_LEAF_RECALL = 0.85
RULE_THRESHOLD_MAX_CANDIDATES = 48

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)






def notebook_globals() -> Dict[str, object]:
    return {
        "argparse": argparse,
        "json": json,
        "math": math,
        "Counter": Counter,
        "np": np,
        "pd": pd,
        "display": display,
        "sigmoid": sigmoid,
        "tam": tam,
        "ROOT": ROOT,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "WINDOW_SIZE": WINDOW_SIZE,
        "STRIDE": STRIDE,
        "MIN_STEP": MIN_STEP,
        "PURE_VAL_RATIO": PURE_VAL_RATIO,
        "SEED": SEED,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "ROW_WEIGHT": ROW_WEIGHT,
        "PAIRWISE_WEIGHT": PAIRWISE_WEIGHT,
        "L2": L2,
        "MIN_PAIR_GAP": MIN_PAIR_GAP,
        "CALIBRATION_LEARNING_RATE": CALIBRATION_LEARNING_RATE,
        "CALIBRATION_EPOCHS": CALIBRATION_EPOCHS,
        "CALIBRATION_L2": CALIBRATION_L2,
        "CONFIDENCE_Z": CONFIDENCE_Z,
        "TARGET_LEAF_RECALL": TARGET_LEAF_RECALL,
        "RULE_THRESHOLD_MAX_CANDIDATES": RULE_THRESHOLD_MAX_CANDIDATES,
    }
