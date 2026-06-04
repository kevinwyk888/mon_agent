"""Flatten a tree-search ``*.tree.json`` into a per-step CSV.

Library entry point used by both ``mon_agent.runner`` (per-instance, online)
and ``results/tree_to_csv.py`` (batch, post-hoc).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Columns kept (in order) — matches the user's feature schema, with `y` last.
COLUMNS = [
    "instance_id",
    "node_id",
    "depth",
    "temperature",
    "step_idx",
    "prompt_tokens",
    "completion_tokens",
    "prompt_tokens_cum",
    "completion_tokens_cum",
    "step_wall_s",
    "context_len_cum",
    "command_type",
    "target_file",
    "returncode",
    "exception_flag",
    "output_len",
    "output_elided_chars",
    "obs_tag",
    "repeat_cmd_score_recent",
    "repeat_file_score_recent",
    "failure_streak",
    "confidence",
    "y",
]

# Per-step keys we explicitly drop.
_DROPPED_KEYS = {"cost", "cost_cum", "command_raw", "test_delta"}


def _walk(node: dict, instance_id: str, rows: list[dict]) -> None:
    y = node.get("y", 0.0)
    node_id = node.get("node_id", "")
    depth = node.get("depth", 0)
    temperature = node.get("temperature", "")
    for step in node.get("step_logs", []) or []:
        row = {
            "instance_id": instance_id,
            "node_id": node_id,
            "depth": depth,
            "temperature": temperature,
            "y": y,
        }
        for k, v in step.items():
            if k in _DROPPED_KEYS:
                continue
            row[k] = v
        rows.append(row)
    for child in node.get("children", []) or []:
        _walk(child, instance_id, rows)


def convert_one(
    tree_json_path: Path,
    out_path: Path | None = None,
    csv_dir: Path | None = None,
) -> Path:
    """Convert one ``<inst>.tree.json`` to a flat ``<inst>.steps.csv``.

    Parameters
    ----------
    tree_json_path:
        Source nested-tree JSON.
    out_path:
        Explicit destination. Mutually exclusive with ``csv_dir``.
    csv_dir:
        Destination directory; CSV is written as ``<csv_dir>/<instance_id>.steps.csv``.
        Created if missing.
    """
    tree_json_path = Path(tree_json_path)
    data = json.loads(tree_json_path.read_text())
    tree = data.get("tree") or data  # tolerate either layout
    instance_id = (
        data.get("instance_id")
        or tree_json_path.stem.replace(".tree", "")
    )

    rows: list[dict] = []
    _walk(tree, instance_id, rows)

    # Stable ordering: by step_idx then node_id for deterministic diffs.
    rows.sort(key=lambda r: (r.get("step_idx", 0), r.get("node_id", "")))

    if out_path is None:
        if csv_dir is not None:
            csv_dir = Path(csv_dir)
            csv_dir.mkdir(parents=True, exist_ok=True)
            out_path = csv_dir / f"{instance_id}.steps.csv"
        else:
            out_path = tree_json_path.with_name(
                tree_json_path.name.replace(".tree.json", ".steps.csv")
            )
            if out_path == tree_json_path:  # fallback if naming didn't match
                out_path = tree_json_path.with_suffix(".steps.csv")
    else:
        out_path = Path(out_path)

    # Use the canonical column list, but tolerate unexpected extras by appending.
    extras = sorted({k for r in rows for k in r.keys() if k not in COLUMNS})
    fieldnames = COLUMNS + extras

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return out_path
