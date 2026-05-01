"""Repetition scores over a sliding window of recent steps."""

from __future__ import annotations

from mon_agent.features.command_parser import classify_command, extract_target_file


def repeat_cmd_score(
    current_command: str,
    recent_commands: list[str],
) -> float:
    """Fraction of *recent_commands* whose coarse type matches *current_command*.

    Returns 0.0 when *recent_commands* is empty.
    """
    if not recent_commands:
        return 0.0
    cur_type = classify_command(current_command)
    matches = sum(1 for c in recent_commands if classify_command(c) == cur_type)
    return matches / len(recent_commands)


def repeat_file_score(
    current_command: str,
    recent_commands: list[str],
) -> float:
    """Fraction of *recent_commands* that touch the same target file.

    Returns 0.0 when the current command has no detectable file or
    *recent_commands* is empty.
    """
    cur_file = extract_target_file(current_command)
    if not cur_file or not recent_commands:
        return 0.0
    matches = sum(1 for c in recent_commands if extract_target_file(c) == cur_file)
    return matches / len(recent_commands)
