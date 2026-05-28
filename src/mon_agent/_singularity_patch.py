"""Defensive patches + SIF-cache integration for mini-SWE-agent's Singularity env.

Two problems this module fixes
------------------------------

1. ``SingularityEnvironment.__init__`` calls ``self._build_sandbox()`` which can
   raise (e.g. when Docker Hub returns 429 TOOMANYREQUESTS).  Because
   ``self.sandbox_dir`` is only assigned after the call succeeds, the implicit
   ``__del__`` path then crashes with

       AttributeError: 'SingularityEnvironment' object has no attribute 'sandbox_dir'

   masking the original error and corrupting batch progress.  ``apply_patches``
   makes ``__init__`` initialise ``self.sandbox_dir = None`` up-front and makes
   ``cleanup`` a no-op when nothing was built.

2. The upstream env always builds the sandbox from ``docker://docker.io/...``
   directly, which means every fresh instance triggers an unauthenticated
   Docker Hub pull.  On Slurm batches of 100+ instances that quickly trips
   ``TOOMANYREQUESTS``.  ``build_env_for_instance`` pre-pulls each instance's
   image into the same ``.sif`` cache the harness uses (``${SIF_CACHE}``) and
   then points ``environment.image`` at the local file, so:

       - subsequent jobs reuse the cache and make zero docker.io requests, and
       - the agent runtime and the harness evaluator share one cache directory.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from copy import deepcopy
from pathlib import Path

logger = logging.getLogger("mon_agent.singularity_patch")

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(name: str) -> str:
    """Sanitize an instance_id for filesystem use (matches evaluate_singularity._safe)."""
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "x"


# ---------------------------------------------------------------------------
# (1) Defensive monkey-patch for SingularityEnvironment
# ---------------------------------------------------------------------------

_PATCHED = False


def apply_patches() -> None:
    """Idempotently patch mini-SWE-agent's SingularityEnvironment.

    Safe to call multiple times; only the first call mutates the class.
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        from minisweagent.environments import singularity as _sing_mod  # type: ignore
    except Exception:
        logger.debug("minisweagent.environments.singularity not importable; skipping patch")
        return

    cls = _sing_mod.SingularityEnvironment
    _orig_init = cls.__init__
    _orig_cleanup = cls.cleanup

    def _safe_init(self, *args, **kwargs):  # type: ignore[no-redef]
        # Assign BEFORE _build_sandbox so __del__/cleanup never AttributeError
        # if the build raises (e.g. Docker Hub rate limit).
        self.sandbox_dir = None
        _orig_init(self, *args, **kwargs)

    def _safe_cleanup(self):  # type: ignore[no-redef]
        sandbox_dir = getattr(self, "sandbox_dir", None)
        if not sandbox_dir:
            return
        try:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        except Exception:  # pragma: no cover - cleanup must never raise
            pass

    cls.__init__ = _safe_init  # type: ignore[assignment]
    cls.cleanup = _safe_cleanup  # type: ignore[assignment]
    _PATCHED = True
    logger.debug("Applied SingularityEnvironment defensive patches")


# ---------------------------------------------------------------------------
# (2) SIF prefetch -> local-image env construction
# ---------------------------------------------------------------------------


def _pull_sif(
    image_key: str,
    sif_path: Path,
    *,
    sing_exe: str = "singularity",
    timeout_s: int = 1800,
    retries: int = 4,
) -> None:
    """Pull ``docker://<image_key>`` into ``sif_path`` with backoff.

    Uses a sibling ``.lock`` file so concurrent workers serialise on the same
    image without re-pulling.  Backoff doubles on each retry and is extended
    when Docker Hub returns ``TOOMANYREQUESTS`` (the 6h anonymous pull-rate
    cap is the dominant failure mode on shared clusters).
    """
    if sif_path.exists() and sif_path.stat().st_size > 0:
        return

    sif_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = sif_path.with_suffix(sif_path.suffix + ".lock")

    # Coarse-grained inter-process lock; we don't need fcntl niceties because
    # the inner check below makes duplicate pulls cheap (just an `exists`).
    import fcntl  # local import: not available on Windows but Slurm is Linux

    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)
        # Another worker may have completed the pull while we waited.
        if sif_path.exists() and sif_path.stat().st_size > 0:
            return

        tmp = sif_path.with_suffix(sif_path.suffix + ".part")
        if tmp.exists():
            tmp.unlink()

        last_err = ""
        backoff = 30.0  # seconds
        for attempt in range(1, retries + 1):
            cmd = [sing_exe, "pull", "--force", str(tmp), f"docker://{image_key}"]
            logger.info("singularity pull (attempt %d/%d): %s", attempt, retries, " ".join(cmd))
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, timeout=timeout_s, text=True
                )
            except subprocess.TimeoutExpired:
                last_err = f"pull_timeout>{timeout_s}s"
                logger.error("singularity pull timed out for %s", image_key)
            else:
                if proc.returncode == 0:
                    tmp.replace(sif_path)
                    return
                last_err = (proc.stderr or proc.stdout or "")[-1500:].strip()
                logger.error(
                    "singularity pull failed (rc=%d) for %s: %s",
                    proc.returncode, image_key, last_err[:500],
                )

            # Longer cool-off for Docker Hub rate limit (resets in ~6h, but
            # short bursts often clear in a few minutes).
            sleep_s = backoff * 4 if "TOOMANYREQUESTS" in last_err else backoff
            if attempt < retries:
                logger.info("Backing off %.0fs before retry", sleep_s)
                time.sleep(sleep_s)
                backoff *= 2

        # All retries exhausted
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to pull docker://{image_key} into {sif_path} after {retries} attempts: {last_err}"
        )


def _resolve_sif_cache_dir(config: dict) -> str:
    """Look up an SIF cache dir from config / env, returning '' if disabled."""
    cand = (
        (config.get("environment") or {}).get("sif_cache_dir")
        or (config.get("tree_search") or {}).get("harness_sif_cache_dir")
        or os.environ.get("MSWEA_SIF_CACHE_DIR")
        or ""
    )
    return str(cand or "")


def build_env_for_instance(config: dict, instance: dict):
    """Drop-in replacement for ``minisweagent.run.benchmarks.swebench.get_sb_environment``
    that prefers a local cached ``.sif`` over a fresh Docker Hub pull.

    Falls back to the upstream behaviour when no cache dir is configured or
    the environment class is not Singularity.
    """
    # Lazy imports so this module is importable even if minisweagent isn't installed.
    from jinja2 import StrictUndefined, Template  # type: ignore
    from minisweagent.environments import get_environment  # type: ignore
    from minisweagent.run.benchmarks.swebench import (  # type: ignore
        get_swebench_docker_image_name,
    )

    # Per-call deepcopy: avoids mutating the shared config dict across worker threads.
    env_config = deepcopy(config.get("environment") or {})
    env_config.setdefault("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)

    sif_cache_dir = _resolve_sif_cache_dir(config)
    used_local_sif = False

    if env_config["environment_class"] in {"singularity", "contree"} and sif_cache_dir:
        instance_id = instance.get("instance_id") or image_name
        sif_path = Path(sif_cache_dir).expanduser() / f"{_safe(instance_id)}.sif"
        sing_exe = env_config.get("executable") or os.getenv(
            "MSWEA_SINGULARITY_EXECUTABLE", "singularity"
        )
        try:
            _pull_sif(image_name, sif_path, sing_exe=sing_exe)
            env_config["image"] = str(sif_path)  # local file -> no docker.io traffic
            used_local_sif = True
        except Exception as e:
            logger.warning(
                "Local SIF prefetch failed for %s (%s); falling back to docker:// pull",
                instance_id, e,
            )

    if not used_local_sif:
        if env_config["environment_class"] in {"docker", "swerex_modal"}:
            env_config["image"] = image_name
        elif env_config["environment_class"] in {"singularity", "contree"}:
            env_config["image"] = f"docker://{image_name}"

    env = get_environment(env_config)

    startup_command = (config.get("run") or {}).get("env_startup_command")
    if startup_command:
        rendered = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(rendered)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env
