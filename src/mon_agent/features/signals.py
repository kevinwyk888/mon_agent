"""Observation‐level signals: obs_tag and test_delta."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# obs_tag — classify a single environment observation
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"SyntaxError|IndentationError", re.I), "syntax_error"),
    (re.compile(r"Traceback \(most recent call last\)", re.I), "exception"),
    (re.compile(r"FAILED|ERRORS?\b.*test", re.I), "test_fail"),
    (re.compile(r"command not found|No such file or directory", re.I), "cmd_error"),
]


def classify_obs_tag(
    output: str,
    returncode: int,
    exception_info: str,
) -> str:
    """Return a short tag summarising the observation.

    Possible values: ``success``, ``syntax_error``, ``exception``,
    ``test_fail``, ``cmd_error``, ``empty_output``, ``other_error``.
    """
    if exception_info:
        return "exception"

    if returncode == 0:
        if not output or not output.strip():
            return "empty_output"
        return "success"

    # returncode != 0 — try to classify
    for pattern, tag in _ERROR_PATTERNS:
        if pattern.search(output):
            return tag

    return "other_error"


# ---------------------------------------------------------------------------
# test_delta — did test results improve, worsen, or stay the same?
# ---------------------------------------------------------------------------

_TEST_RESULT_RE = re.compile(
    r"(\d+)\s+passed"
    r"(?:.*?(\d+)\s+failed)?"
    r"(?:.*?(\d+)\s+error)?",
    re.I,
)


def _parse_test_counts(output: str) -> tuple[int, int, int] | None:
    """Extract (passed, failed, errors) from pytest‐style output."""
    m = _TEST_RESULT_RE.search(output)
    if not m:
        return None
    passed = int(m.group(1))
    failed = int(m.group(2) or 0)
    errors = int(m.group(3) or 0)
    return (passed, failed, errors)


def compute_test_delta(
    current_output: str,
    previous_output: str,
) -> str:
    """Compare test results between two consecutive outputs.

    Returns one of: ``improved``, ``unchanged``, ``worse``, ``NA``.
    """
    cur = _parse_test_counts(current_output)
    prev = _parse_test_counts(previous_output)

    if cur is None or prev is None:
        return "NA"

    cur_pass, cur_fail, cur_err = cur
    prev_pass, prev_fail, prev_err = prev

    cur_bad = cur_fail + cur_err
    prev_bad = prev_fail + prev_err

    if cur_bad < prev_bad or (cur_bad == prev_bad and cur_pass > prev_pass):
        return "improved"
    elif cur_bad == prev_bad and cur_pass == prev_pass:
        return "unchanged"
    else:
        return "worse"
