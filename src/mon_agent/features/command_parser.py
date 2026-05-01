"""Classify bash commands into coarse action types and extract target files."""

from __future__ import annotations

import re


# Ordered list of (pattern, label).  First match wins.
_COMMAND_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(pytest|python\s+-m\s+pytest|tox|unittest|make\s+test)", re.I), "test"),
    (re.compile(r"\b(sed\s+-i|patch\b|ed\b)", re.I), "edit"),
    (re.compile(r"\b(cat|head|tail|less|more|bat)\b", re.I), "read"),
    (re.compile(r"\b(grep|rg|ag|ack|find|fd|locate)\b", re.I), "search"),
    (re.compile(r"\b(ls|tree|wc|du|file|stat)\b", re.I), "explore"),
    (re.compile(r"\bpython\b", re.I), "run"),
    (re.compile(r"\b(pip|apt|conda|npm|cargo)\b", re.I), "install"),
    (re.compile(r"\bgit\s+diff\b", re.I), "diff"),
]

# Heuristic: if the command contains a heredoc write or redirect‐create, it is
# most likely an edit (creating / overwriting a file).
_HEREDOC_RE = re.compile(r"(cat\s*<<|>\s*/|tee\s)", re.I)


def classify_command(command: str) -> str:
    """Return a coarse action type for *command*.

    Possible labels: ``read``, ``search``, ``explore``, ``edit``, ``test``,
    ``run``, ``install``, ``diff``, ``other``.
    """
    if not command or not command.strip():
        return "other"

    # Heredoc / redirect → edit
    if _HEREDOC_RE.search(command):
        return "edit"

    for pattern, label in _COMMAND_PATTERNS:
        if pattern.search(command):
            return label

    return "other"


# Grab the first thing that looks like a file path.
_FILE_RE = re.compile(
    r"""(?:^|\s)                    # start of string or whitespace
        (/?                         # optional leading /
         (?:[\w.@~-]+/)*           # zero or more dir/ segments
         [\w.@~-]+                 # filename (must have at least one char)
         \.[\w]+                   # .extension
        )""",
    re.VERBOSE,
)


def extract_target_file(command: str) -> str:
    """Return the first file‐path‐like token from *command*, or ``""``."""
    if not command:
        return ""
    m = _FILE_RE.search(command)
    return m.group(1) if m else ""
