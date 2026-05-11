"""Singularity-backed SWE-bench evaluator.

Mirrors the public surface of :mod:`mon_agent.evaluate` (``EvalResult``,
``evaluate_submission``) but runs the per-instance test container with
``singularity exec`` instead of the Docker SDK that the upstream harness uses.

This is needed on HPC clusters (e.g. UMich Great Lakes) where Docker is not
available but Singularity is. We:

  1. Use ``swebench.harness.test_spec.test_spec.make_test_spec(inst, namespace="swebench")``
     to obtain ``spec.instance_image_key`` (the public Docker Hub URL,
     e.g. ``swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-2419:latest``)
     and ``spec.eval_script`` (the bash script that checks out the base
     commit, applies the test patch, and runs pytest with sentinel markers).
  2. Cache a ``.sif`` per instance under ``sif_cache_dir`` (default
     ``~/.cache/swebench_sif``). First call does
     ``singularity pull <sif> docker://<image_key>``; subsequent calls reuse it.
  3. Stage a workdir with ``patch.diff``, ``eval.sh``, and a small ``run.sh``
     that tries the same ``GIT_APPLY_CMDS`` fallback chain as the upstream
     harness, then execs ``eval.sh`` and tees its output to
     ``test_output.txt`` (the file format expected by
     :func:`swebench.harness.grading.get_eval_report`).
  4. Run with ``singularity exec --writable-tmpfs --containall --bind
     <work>:/eval_io <sif> bash /eval_io/run.sh``. ``--writable-tmpfs``
     overlays a tmpfs so the conda env can ``pip install -e .`` and tests can
     write to ``/testbed`` without touching the read-only image.
  5. Parse via ``grading.get_eval_report`` and return whether
     ``resolved`` is true.

If anything goes wrong (no singularity on PATH, pull failure, exec failure,
report parse failure) we return ``EvalResult(resolved=False, error=...)`` so
the caller can record the failure mode without crashing the agent.
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_pg(
    cmd: list[str],
    *,
    timeout_s: int,
    capture_output: bool = True,
) -> tuple[int, str, str, bool]:
    """Run ``cmd`` in its own process group; on timeout kill the whole group.

    Returns ``(returncode, stdout, stderr, timed_out)``.

    Why a new process group: ``subprocess.run(timeout=...)`` only kills the
    immediate child. When that child is ``singularity exec`` (or ``bash``
    inside the container without ``--pid``) it forks grandchildren — e.g.
    Django's ``tests/runtests.py`` multiprocessing pool — that survive in the
    host PID namespace as orphans, still holding the stdout/stderr pipes.
    The parent worker thread then deadlocks in ``communicate()`` waiting for
    EOF that never comes. Spawning with ``start_new_session=True`` puts the
    whole tree in a fresh pgid; on timeout we ``killpg`` the group so every
    descendant dies and the pipes close cleanly.
    """
    stdout_arg = subprocess.PIPE if capture_output else None
    stderr_arg = subprocess.PIPE if capture_output else None
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_arg,
        stderr=stderr_arg,
        text=True,
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            try:
                out, err = proc.communicate(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            # Final fallback: detach pipes so the worker thread does not hang.
            try:
                proc.kill()
            except Exception:
                pass
            out, err = "", ""
    return proc.returncode if proc.returncode is not None else -9, out or "", err or "", timed_out

# Same fallback chain the upstream harness uses to apply the model patch.
_GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

_DEFAULT_SIF_CACHE = os.path.expanduser("~/.cache/swebench_sif")
_SAFE_NAME_RE = _re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "x"


@dataclass
class EvalResult:
    resolved: bool
    error: str = ""
    wall_s: float = 0.0
    report_path: str = ""
    raw_report: dict | None = None


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _is_real_binary(path: str) -> bool:
    """Return True if *path* is an executable binary, not a text shim.

    On UMich Great Lakes, ``/usr/local/bin/singularity`` is an Ansible-managed
    text stub printing "module load singularity" instructions; calling it
    raises ``Exec format error``. We probe the first two bytes for the ELF
    magic to filter such shims.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head[:4] == b"\x7fELF"
    except Exception:
        return False


# Common locations for real singularity/apptainer binaries on HPC clusters
# (in order of preference). Used as a fallback when ``shutil.which`` returns
# a text stub.
_CANDIDATE_SING_BINARIES = [
    "/opt/singularity/4.3.4/bin/singularity",
    "/opt/singularity/4.1.5/bin/singularity",
    "/usr/bin/singularity",
    "/usr/bin/apptainer",
]


def _which_singularity() -> str | None:
    # 1. Honor PATH first, but skip text stubs.
    for name in ("singularity", "apptainer"):
        p = shutil.which(name)
        if p and _is_real_binary(p):
            return p
    # 2. Fall back to known-good HPC paths.
    for p in _CANDIDATE_SING_BINARIES:
        if _is_real_binary(p):
            return p
    # 3. Glob /opt for any singularity install.
    import glob
    for p in sorted(glob.glob("/opt/singularity/*/bin/singularity"), reverse=True):
        if _is_real_binary(p):
            return p
    return None


def _load_instance(dataset_name: str, instance_id: str) -> dict | None:
    """Look up a SWE-bench instance row from a HF dataset.

    Tries the ``dev`` and ``test`` splits to cover Lite/full layouts.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("datasets package missing: %s", e)
        return None

    for split in ("dev", "test"):
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception:
            continue
        try:
            for row in ds:
                if row.get("instance_id") == instance_id:
                    return dict(row)
        except Exception:
            continue
    return None


def _ensure_sif(image_key: str, sif_path: Path, *, sing_exe: str, timeout_s: int) -> str:
    """Pull the image if ``sif_path`` does not exist. Returns "" on success
    or an error string on failure."""
    if sif_path.exists() and sif_path.stat().st_size > 0:
        return ""
    sif_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sif_path.with_suffix(sif_path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [sing_exe, "pull", "--force", str(tmp), f"docker://{image_key}"]
    logger.info("singularity pull: %s", " ".join(cmd))
    rc, _out, err, timed_out = _run_pg(cmd, timeout_s=timeout_s)
    if timed_out:
        return f"pull_timeout>{timeout_s}s"
    if rc != 0:
        tail = (err or _out or "")[-1500:]
        return f"pull_rc={rc}:{tail.strip()[:500]}"
    try:
        tmp.replace(sif_path)
    except Exception as e:
        return f"pull_rename:{e}"
    return ""


def _build_run_script(eval_script_path_in_container: str = "/eval_io/eval.sh") -> str:
    """Bash that applies the model patch (with fallbacks) then runs eval.sh.

    Mirrors what the upstream harness does in Python. Output ordering matters:
    ``get_eval_report`` only consumes between the ``>>>>> Start/End Test Output``
    markers that ``eval.sh`` itself emits, so noise from the patch-apply stage
    is harmless.
    """
    apply_block = "\n".join(
        f"  {cmd} /eval_io/patch.diff && {{ echo APPLY_PATCH_PASS; APPLIED=1; break; }}"
        for cmd in _GIT_APPLY_CMDS
    )
    return f"""#!/bin/bash
set -uo pipefail
cd /testbed
git config --global --add safe.directory /testbed >/dev/null 2>&1 || true

APPLIED=0
for _ in 1; do
{apply_block}
done
if [ "$APPLIED" != "1" ]; then
  echo APPLY_PATCH_FAIL
fi

bash {eval_script_path_in_container}
"""


def _run_container(
    sing_exe: str,
    sif_path: Path,
    work_path: Path,
    *,
    timeout_s: int,
    extra_singularity_args: list[str] | None = None,
) -> tuple[int, str, str]:
    """Exec ``run.sh`` inside the SIF, returning (rc, stdout, stderr)."""
    cmd = [
        sing_exe,
        "exec",
        "--writable-tmpfs",
        "--containall",
        "--cleanenv",
        # ``--pid`` puts the container in its own PID namespace, so when
        # singularity exits every descendant (Django's multiprocessing pool,
        # pytest-xdist workers, etc.) is reaped by the kernel instead of
        # being orphaned to the host. Without this, killing the singularity
        # process leaves grandchildren behind, hanging worker pipes.
        "--pid",
        "--bind", f"{work_path}:/eval_io",
        str(sif_path),
        "bash", "/eval_io/run.sh",
    ]
    if extra_singularity_args:
        # insert before the SIF path
        sif_idx = cmd.index(str(sif_path))
        cmd[sif_idx:sif_idx] = list(extra_singularity_args)
    logger.info("singularity exec: %s", " ".join(cmd))
    rc, out, err, timed_out = _run_pg(cmd, timeout_s=timeout_s)
    if timed_out:
        return 124, out, (err + f"\nexec_timeout>{timeout_s}s").strip()
    return rc, out, err


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_submission(
    instance_id: str,
    patch: str,
    *,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    run_id: str = "sing_eval",
    model_name: str = "mc_fork",
    timeout_s: int = 1800,
    pull_timeout_s: int = 1800,
    sif_cache_dir: str | os.PathLike = _DEFAULT_SIF_CACHE,
    work_dir: str | os.PathLike | None = None,
    keep_work_dir: bool = False,
    singularity_exe: str | None = None,
    extra_singularity_args: list[str] | None = None,
    # accepted for API compatibility with evaluate.evaluate_submission
    namespace: str | None = None,
    max_workers: int = 1,
    python_exe: str | None = None,
    extra_args: list[str] | None = None,
    harness_run_id_prefix: str | None = None,  # ignored
) -> EvalResult:
    """Evaluate a single SWE-bench (instance, patch) using Singularity.

    ``namespace`` / ``max_workers`` / ``python_exe`` / ``extra_args`` are
    accepted for signature parity with older Docker-based variants but have
    no effect here (the namespace is fixed to ``swebench`` because that is
    what is on Docker Hub).
    """
    t0 = time.monotonic()

    if not patch or not patch.strip():
        return EvalResult(resolved=False, error="empty_patch", wall_s=0.0)

    sing_exe = singularity_exe or _which_singularity()
    if not sing_exe:
        return EvalResult(
            resolved=False,
            error="singularity_not_found (try: module load singularity)",
            wall_s=round(time.monotonic() - t0, 2),
        )

    # Build TestSpec → image URL + eval script.
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
        from swebench.harness import grading
    except Exception as e:
        return EvalResult(
            resolved=False,
            error=f"swebench_import:{type(e).__name__}:{e}",
            wall_s=round(time.monotonic() - t0, 2),
        )

    instance = _load_instance(dataset_name, instance_id)
    if instance is None:
        return EvalResult(
            resolved=False,
            error=f"instance_not_found:{instance_id}",
            wall_s=round(time.monotonic() - t0, 2),
        )

    spec = make_test_spec(instance, namespace="swebench", instance_image_tag="latest")
    image_key = spec.instance_image_key  # e.g. swebench/sweb.eval.x86_64.<sanitized>:latest

    # Cache .sif per instance.
    sif_dir = Path(os.fspath(sif_cache_dir)).expanduser()
    sif_path = sif_dir / f"{_safe(instance_id)}.sif"

    pull_err = _ensure_sif(image_key, sif_path, sing_exe=sing_exe, timeout_s=pull_timeout_s)
    if pull_err:
        return EvalResult(
            resolved=False,
            error=pull_err,
            wall_s=round(time.monotonic() - t0, 2),
        )

    # Stage workdir.
    owns_work = work_dir is None
    work_path = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix=f"sing_eval_{_safe(instance_id)}_")
    )
    work_path.mkdir(parents=True, exist_ok=True)

    (work_path / "patch.diff").write_text(patch)
    (work_path / "eval.sh").write_text(spec.eval_script)
    (work_path / "run.sh").write_text(_build_run_script())

    rc, stdout, stderr = _run_container(
        sing_exe,
        sif_path,
        work_path,
        timeout_s=timeout_s,
        extra_singularity_args=extra_singularity_args,
    )

    test_output_path = work_path / "test_output.txt"
    test_output_path.write_text(stdout + "\n" + stderr)

    error = ""
    if rc != 0 and rc != 1:
        # rc=1 is normal when tests fail; treat anything else as infra error.
        error = f"exec_rc={rc}:{(stderr or stdout)[-300:].strip()}"

    # Build the prediction dict get_eval_report expects and run it.
    pred = {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": _safe(model_name),
    }
    raw_report: dict | None = None
    resolved = False
    try:
        raw_report = grading.get_eval_report(
            spec, pred, str(test_output_path), include_tests_status=True
        )
        entry = raw_report.get(instance_id, raw_report) if isinstance(raw_report, dict) else {}
        resolved = bool(entry.get("resolved", False))
    except Exception as e:
        if not error:
            error = f"report_parse:{type(e).__name__}:{e}"

    # Persist the report alongside test_output for later inspection.
    report_path = work_path / "report.json"
    if raw_report is not None:
        try:
            report_path.write_text(json.dumps(raw_report, indent=2))
        except Exception:
            pass

    if owns_work and not keep_work_dir:
        shutil.rmtree(work_path, ignore_errors=True)
        report_path_str = ""
    else:
        report_path_str = str(report_path) if raw_report is not None else ""

    return EvalResult(
        resolved=resolved,
        error=error,
        wall_s=round(time.monotonic() - t0, 2),
        report_path=report_path_str,
        raw_report=raw_report,
    )
