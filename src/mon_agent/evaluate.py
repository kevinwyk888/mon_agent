"""Run the official SWE-bench harness on a single (instance, patch) pair.

Used by Monte-Carlo forks to replace the proxy "agent claims it submitted"
success signal with the real "FAIL_TO_PASS / PASS_TO_PASS tests pass" signal.

The harness is invoked as a subprocess (``python -m swebench.harness.run_evaluation``)
because it manages its own Docker containers and we don't want to hold its
imports in the agent process. Each call:

  1. Writes a one-line predictions file with the patch under model name
     ``model_name_or_path``.
  2. Runs the harness with ``--instance_ids <id>`` against that predictions
     file in an isolated working directory.
  3. Reads the per-instance ``report.json`` produced under
     ``logs/run_evaluation/<run_id>/<model_name>/<instance_id>/report.json``
     and returns ``report[instance_id]["resolved"]``.

If anything fails (timeout, harness crash, missing report, parse error) we
return ``EvalResult(resolved=False, error=...)`` and let the caller decide how
to score it. The MC code treats any non-resolved outcome as failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(name: str) -> str:
    """Sanitize names for use as filesystem components / harness model id."""
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "x"


@dataclass
class EvalResult:
    resolved: bool
    error: str = ""
    wall_s: float = 0.0
    report_path: str = ""
    raw_report: dict | None = None


def evaluate_submission(
    instance_id: str,
    patch: str,
    *,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    run_id: str = "mc_eval",
    model_name: str = "mc_fork",
    max_workers: int = 1,
    timeout_s: int = 1800,
    namespace: str | None = None,
    work_dir: str | os.PathLike | None = None,
    python_exe: str | None = None,
    keep_work_dir: bool = False,
    extra_args: list[str] | None = None,
) -> EvalResult:
    """Run the SWE-bench harness on a single (instance_id, patch) and return
    whether it is resolved (FAIL_TO_PASS pass + PASS_TO_PASS still pass).

    Parameters
    ----------
    instance_id : str
        Canonical SWE-bench instance id (e.g. ``marshmallow-code__marshmallow-1343``).
    patch : str
        Unified-diff text. If empty, this function short-circuits to ``resolved=False``.
    dataset_name : str
        Dataset name passed to the harness (default SWE-bench Lite).
    run_id : str
        Harness ``--run_id``. Used in the on-disk report path; pick something
        unique per call to avoid caching collisions across forks.
    model_name : str
        ``model_name_or_path`` written into the predictions file. Determines
        the harness output sub-directory.
    max_workers : int
        Forwarded to ``--max_workers``.
    timeout_s : int
        Hard wall-clock cap for the subprocess.
    namespace : str | None
        Optional ``--namespace`` for the harness Docker images.
    work_dir : path | None
        Where to write the predictions file and harness logs. If None, a
        temp dir is created and removed after evaluation (unless
        ``keep_work_dir=True``).
    python_exe : str | None
        Python interpreter used to invoke the harness. Defaults to
        ``sys.executable``.
    keep_work_dir : bool
        Keep the working directory (and its harness logs) after the call.
    extra_args : list[str] | None
        Additional CLI args appended verbatim.
    """
    t0 = time.monotonic()

    if not patch or not patch.strip():
        return EvalResult(resolved=False, error="empty_patch", wall_s=0.0)

    owns_work_dir = work_dir is None
    work_path = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="mc_eval_"))
    work_path.mkdir(parents=True, exist_ok=True)

    safe_model = _safe(model_name)
    safe_run = _safe(run_id)
    safe_inst = _safe(instance_id)

    preds_path = work_path / f"preds_{safe_inst}_{safe_run}.json"
    preds_path.write_text(
        json.dumps(
            {
                instance_id: {
                    "instance_id": instance_id,
                    "model_name_or_path": safe_model,
                    "model_patch": patch,
                }
            }
        )
    )

    py = python_exe or sys.executable
    cmd: list[str] = [
        py, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(preds_path),
        "--max_workers", str(max_workers),
        "--run_id", safe_run,
        "--instance_ids", instance_id,
        "--cache_level", "instance",
    ]
    if namespace:
        cmd += ["--namespace", namespace]
    if extra_args:
        cmd += list(extra_args)

    error = ""
    proc_stdout = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_path),
            timeout=timeout_s,
            capture_output=True,
            text=True,
            check=False,
        )
        proc_stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            error = f"harness_rc={proc.returncode}"
    except subprocess.TimeoutExpired:
        error = f"harness_timeout>{timeout_s}s"
    except Exception as e:  # pragma: no cover - defensive
        error = f"harness_exception:{type(e).__name__}:{e}"

    # Locate the per-instance report. The harness writes to
    #   <cwd>/logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json
    candidates = [
        work_path / "logs" / "run_evaluation" / safe_run / safe_model / instance_id / "report.json",
        # Some harness versions namespace by sanitized instance id too.
        work_path / "logs" / "run_evaluation" / safe_run / safe_model / safe_inst / "report.json",
    ]
    report_path = next((p for p in candidates if p.exists()), None)

    resolved = False
    raw_report: dict | None = None
    if report_path is not None:
        try:
            raw_report = json.loads(report_path.read_text())
            entry = raw_report.get(instance_id, raw_report)
            resolved = bool(entry.get("resolved", False))
        except Exception as e:
            error = error or f"report_parse:{type(e).__name__}:{e}"
    else:
        error = error or "report_missing"
        if proc_stdout:
            logger.debug(
                "Harness output for %s (no report.json):\n%s",
                instance_id,
                proc_stdout[-2000:],
            )

    if owns_work_dir and not keep_work_dir:
        shutil.rmtree(work_path, ignore_errors=True)

    return EvalResult(
        resolved=resolved,
        error=error,
        wall_s=round(time.monotonic() - t0, 2),
        report_path=str(report_path) if report_path else "",
        raw_report=raw_report,
    )
