"""Compact prefix representation for monitoring.

A prefix after step k contains:
  1. issue summary (task description)
  2. recent window of step cards (last W steps)
  3. global counters
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class StepCard:
    """Compact summary of a single agent step."""

    step_idx: int
    action_type: str
    target_file: str
    returncode: int
    obs_tag: str
    test_delta: str


@dataclass
class GlobalCounters:
    """Cumulative counters tracked across the trajectory."""

    step_idx: int = 0
    prompt_tokens_cum: int = 0
    completion_tokens_cum: int = 0
    context_len_cum: int = 0
    repeat_cmd_score_recent: float = 0.0
    repeat_file_score_recent: float = 0.0
    failure_streak: int = 0


@dataclass
class Prefix:
    """Prefix representation after step k."""

    issue_summary: str
    recent_steps: list[StepCard] = field(default_factory=list)
    counters: GlobalCounters = field(default_factory=GlobalCounters)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prefix(
    issue_summary: str,
    step_logs: list[dict[str, Any]],
    window_size: int = 3,
) -> Prefix:
    """Build a prefix from the full step log history.

    Args:
        issue_summary: The original task description.
        step_logs: List of step-level log dicts (as produced by MonitoringAgent).
        window_size: Number of recent steps to include.
    """
    if not step_logs:
        return Prefix(issue_summary=issue_summary)

    latest = step_logs[-1]
    recent = step_logs[-window_size:]

    recent_cards = [
        StepCard(
            step_idx=s["step_idx"],
            action_type=s["command_type"],
            target_file=s["target_file"],
            returncode=s["returncode"],
            obs_tag=s["obs_tag"],
            test_delta=s["test_delta"],
        )
        for s in recent
    ]

    counters = GlobalCounters(
        step_idx=latest["step_idx"],
        prompt_tokens_cum=latest["prompt_tokens_cum"],
        completion_tokens_cum=latest["completion_tokens_cum"],
        context_len_cum=latest["context_len_cum"],
        repeat_cmd_score_recent=latest["repeat_cmd_score_recent"],
        repeat_file_score_recent=latest["repeat_file_score_recent"],
        failure_streak=latest["failure_streak"],
    )

    return Prefix(
        issue_summary=issue_summary,
        recent_steps=recent_cards,
        counters=counters,
    )
