"""Monte-Carlo fork rollouts for prefix-level success-rate estimation.

This implements Phase-2 from `feasible_method.md`:

    Y_k^MC = (1/M) * sum_{m=1..M} 1[continuation m succeeds]

At chosen step indices we

  1. snapshot the working tree of the agent's environment (git commit-tree),
  2. spawn ``M`` fork rollouts that share the same env but use a higher-temperature
     copy of the model so they actually diverge,
  3. between/after rollouts reset the env back to the snapshot so the main
     trajectory is unaffected,
  4. report the proxy success rate (model produced a non-empty submission with
     ``exit_status == "Submitted"``).

Notes
-----
* The "success" signal here is ``submission != ""`` AND ``exit_status == "Submitted"``.
  This is a *proxy* — it means the model believes it solved the task, not that
  SWE-bench's grader agrees.  The full submissions are saved per-fork so they
  can be re-scored offline with the official harness.
* Forks run **sequentially** because they share one Singularity sandbox.
  Cross-instance parallelism still works through the runner's ThreadPoolExecutor.
* Forks do NOT recursively fork (the ``_is_fork`` flag short-circuits MC inside
  ``MonitoringAgent.step``).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from minisweagent.exceptions import InterruptAgentFlow

if TYPE_CHECKING:  # avoid circular import at runtime
    from mon_agent.agent import MonitoringAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MCForkConfig:
    """Settings for Monte-Carlo prefix evaluation."""

    enabled: bool = False
    fork_every: int = 5
    """Run an MC estimate every N main-trajectory steps."""
    samples: int = 4
    """Number of fork rollouts per snapshot (M)."""
    temperature: float = 0.7
    """Sampling temperature used inside forks."""
    top_p: float = 0.95
    max_fork_steps: int = 20
    """Static cap on fork steps. Used as fallback when dynamic cap is disabled
    (max_fork_steps_total == 0)."""
    max_fork_steps_min: int = 0
    """Lower bound on per-snapshot fork-step cap (only used when total > 0)."""
    max_fork_steps_total: int = 0
    """If > 0, fork at step k uses cap = max(max_fork_steps_min, max_fork_steps_total - k)."""
    fork_cost_budget: float = 1.0
    """Maximum extra cost (USD) per individual fork rollout."""
    snapshot_cwd: str = "/testbed"
    """Working directory to snapshot via git in the container."""
    seed_base: int = 1234
    """Base RNG seed; seed_i = seed_base + i."""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MCForkConfig":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def fork_step_cap(self, current_step: int) -> int:
        """Per-fork step cap for a snapshot taken at *current_step*.

        If max_fork_steps_total > 0, returns max(max_fork_steps_min,
        max_fork_steps_total - current_step). Otherwise returns max_fork_steps.
        """
        if self.max_fork_steps_total > 0:
            return max(self.max_fork_steps_min, self.max_fork_steps_total - current_step)
        return self.max_fork_steps


@dataclass
class ForkResult:
    sample_idx: int
    success: bool
    exit_status: str
    submission: str
    n_steps: int
    cost: float
    wall_s: float
    attempted_submit: bool = False
    error: str = ""
    step_logs: list[dict] = field(default_factory=list)
    """Per-step monitoring records produced by the fork itself (same schema as
    the main `<inst>.steps.jsonl`). Empty when the fork crashed before any
    step was logged."""


@dataclass
class MCResult:
    step_idx: int
    y_mc: float
    n_samples: int
    samples: list[ForkResult] = field(default_factory=list)
    snapshot_sha: str = ""
    total_cost: float = 0.0
    total_wall_s: float = 0.0
    fork_step_cap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "y_mc": round(self.y_mc, 4),
            "n_samples": self.n_samples,
            "snapshot_sha": self.snapshot_sha,
            "total_cost": round(self.total_cost, 4),
            "total_wall_s": round(self.total_wall_s, 2),
            "fork_step_cap": self.fork_step_cap,
            "samples": [s.__dict__ for s in self.samples],
        }


# ---------------------------------------------------------------------------
# Env snapshot/restore via git inside the container
# ---------------------------------------------------------------------------


# Snapshot strategy: capture the working tree as a *tree object* (not a commit)
# and reset the index back to HEAD so that:
#   - HEAD never moves (the agent's eventual `git diff` against HEAD still sees
#     all of its modifications),
#   - the index matches HEAD (so `git diff` between working tree and index also
#     reflects the agent's working changes),
#   - the snapshot tree SHA is reusable across multiple restores.
# Restore replays that tree onto the working tree in three phases:
#   1. force-checkout HEAD onto all tracked files (undo tracked modifications),
#   2. ``git clean -fd`` untracked files left behind by the previous fork
#      (without ``-x`` so editable-install metadata such as ``*.egg-info`` is
#      preserved),
#   3. ``git checkout <tree> -- .`` overlays the snapshot tree onto both the
#      working tree and the index, restoring snapshot-time tracked modifications
#      *and* snapshot-time untracked files,
#   4. ``git reset --mixed HEAD`` rewinds the index to HEAD so the agent's
#      ``git diff`` semantics are preserved.
_SNAPSHOT_CMD = (
    "cd {cwd} && "
    "git add -A && "
    "tree=$(git write-tree) && "
    "git reset --mixed -q HEAD >/dev/null 2>&1 && "
    "printf '%s\\n' \"$tree\""
)
_RESTORE_CMD = (
    "cd {cwd} && "
    "git checkout -f -q HEAD -- . && "
    "git clean -fd --quiet && "
    "git checkout -q {sha} -- . && "
    "git reset --mixed -q HEAD >/dev/null 2>&1"
)


def snapshot_workdir(env: Any, cwd: str) -> str:
    """Capture the current working tree as a git tree object and return its SHA.

    HEAD is left untouched, and the index is reset back to HEAD so subsequent
    ``git diff`` calls by the main agent still see the agent's working changes.
    Raises RuntimeError on failure (e.g. cwd is not a git repo).
    """
    out = env.execute({"command": _SNAPSHOT_CMD.format(cwd=cwd)})
    if out.get("returncode", 1) != 0:
        raise RuntimeError(
            f"Failed to snapshot workdir at {cwd}: rc={out.get('returncode')} "
            f"output={out.get('output', '')[:500]}"
        )
    sha = (out.get("output") or "").strip().splitlines()[-1].strip()
    if not sha or len(sha) < 7:
        raise RuntimeError(f"Invalid snapshot sha: {sha!r}")
    return sha


def restore_workdir(env: Any, cwd: str, sha: str) -> None:
    """Restore the working tree to a previously captured tree SHA.

    The index is then reset back to HEAD so the main agent's ``git diff``
    semantics are preserved. Untracked files created since the snapshot are
    removed (ignored files such as *.egg-info are kept).
    """
    out = env.execute({"command": _RESTORE_CMD.format(cwd=cwd, sha=sha)})
    if out.get("returncode", 1) != 0:
        raise RuntimeError(
            f"Failed to restore workdir to {sha}: {out.get('output', '')[:500]}"
        )


# ---------------------------------------------------------------------------
# Model cloning with new sampling kwargs
# ---------------------------------------------------------------------------


def clone_model_with_sampling(
    model: Any, temperature: float, top_p: float | None = None, seed: int | None = None
) -> Any:
    """Return a new model instance of the same class with overridden sampling
    kwargs. Falls back to returning *model* unchanged if cloning fails."""
    try:
        config_dict = model.config.model_dump()
    except Exception as e:
        logger.warning("Cannot clone model (no model_dump): %s — reusing parent model", e)
        return model

    mk = dict(config_dict.get("model_kwargs") or {})
    mk["temperature"] = temperature
    if top_p is not None:
        mk["top_p"] = top_p
    if seed is not None:
        mk["seed"] = seed
    config_dict["model_kwargs"] = mk

    try:
        return type(model)(**config_dict)
    except Exception as e:
        logger.warning("Model clone failed (%s) — reusing parent model", e)
        return model


# ---------------------------------------------------------------------------
# Core: rollout a single fork
# ---------------------------------------------------------------------------


def _fork_rollout_once(
    parent: "MonitoringAgent",
    fork_model: Any,
    max_fork_steps: int,
    fork_cost_budget: float,
    sample_idx: int,
) -> ForkResult:
    """Run one continuation from parent's current state. Caller must have
    already restored env to the parent's snapshot before this call."""
    from mon_agent.agent import MonitoringAgent  # local import to avoid cycles

    t0 = time.monotonic()

    config_dict = parent.config.model_dump()
    config_dict.pop("output_path", None)  # never overwrite trajectory from a fork
    config_dict["step_limit"] = parent.n_calls + max_fork_steps
    config_dict["cost_limit"] = parent.cost + fork_cost_budget

    fork = MonitoringAgent(
        fork_model,
        parent.env,
        task=getattr(parent, "_task_text", ""),
        is_fork=True,
        **config_dict,
    )
    fork.messages = copy.deepcopy(parent.messages)
    fork.cost = parent.cost
    fork.n_calls = parent.n_calls
    fork._failure_streak = parent._failure_streak
    fork._last_test_output = parent._last_test_output
    fork.extra_template_vars = copy.deepcopy(getattr(parent, "extra_template_vars", {}))
    # Ensure templates relying on {{task}} still render even if the parent has
    # not yet populated extra_template_vars via DefaultAgent.run().
    fork.extra_template_vars.setdefault("task", getattr(parent, "_task_text", ""))

    error_str = ""
    try:
        while True:
            try:
                fork.step()
            except InterruptAgentFlow as e:
                # Submitted / LimitsExceeded / FormatError carry exit/format
                # messages that DefaultAgent.run appends without crashing.
                fork.add_messages(*e.messages)
            except Exception as e:
                fork.handle_uncaught_exception(e)
                break
            if fork.messages and fork.messages[-1].get("role") == "exit":
                break
    except Exception as outer:  # last-resort safety net
        error_str = f"{type(outer).__name__}: {outer}"
        logger.warning("Fork %d crashed: %s", sample_idx, error_str)

    last_extra: dict[str, Any] = {}
    if fork.messages:
        last_extra = fork.messages[-1].get("extra", {}) or {}
    submission = last_extra.get("submission", "") or ""
    exit_status = last_extra.get("exit_status", "") or ""
    attempted_submit = exit_status == "Submitted"
    success = bool(submission.strip()) and attempted_submit

    return ForkResult(
        sample_idx=sample_idx,
        success=success,
        exit_status=exit_status,
        submission=submission,
        n_steps=max(0, fork.n_calls - parent.n_calls),
        cost=max(0.0, fork.cost - parent.cost),
        wall_s=round(time.monotonic() - t0, 2),
        attempted_submit=attempted_submit,
        error=error_str,
        step_logs=list(fork.step_logs),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def estimate_y_mc(
    parent: "MonitoringAgent",
    cfg: MCForkConfig,
) -> MCResult | None:
    """Run M fork rollouts from parent's current prefix and return Y_k^MC.

    Returns ``None`` if the snapshot fails (the main run is left untouched)."""
    step_idx = parent.n_calls

    try:
        sha = snapshot_workdir(parent.env, cfg.snapshot_cwd)
    except Exception as e:
        logger.warning("MC snapshot failed at step %d: %s — skipping", step_idx, e)
        return None

    effective_cap = cfg.fork_step_cap(step_idx)
    logger.info(
        "MC fork cap at step %d = %d (min=%d, total=%d, static=%d)",
        step_idx, effective_cap, cfg.max_fork_steps_min, cfg.max_fork_steps_total, cfg.max_fork_steps,
    )

    samples: list[ForkResult] = []
    t0 = time.monotonic()
    try:
        for i in range(cfg.samples):
            try:
                restore_workdir(parent.env, cfg.snapshot_cwd, sha)
            except Exception as e:
                logger.warning("MC restore before sample %d failed: %s", i, e)
                samples.append(
                    ForkResult(
                        sample_idx=i,
                        success=False,
                        exit_status="RestoreFailed",
                        submission="",
                        n_steps=0,
                        cost=0.0,
                        wall_s=0.0,
                        error=str(e),
                    )
                )
                continue

            fork_model = clone_model_with_sampling(
                parent.model,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=cfg.seed_base + i,
            )
            result = _fork_rollout_once(
                parent=parent,
                fork_model=fork_model,
                max_fork_steps=effective_cap,
                fork_cost_budget=cfg.fork_cost_budget,
                sample_idx=i,
            )
            samples.append(result)
            logger.info(
                "MC step=%d sample=%d/%d success=%s steps=%d cost=$%.3f",
                step_idx, i + 1, cfg.samples, result.success, result.n_steps, result.cost,
            )
    finally:
        # Always restore main run's state, even on partial failure.
        try:
            restore_workdir(parent.env, cfg.snapshot_cwd, sha)
        except Exception as e:
            logger.error(
                "MC final restore failed at step %d (sha=%s): %s — main run state may be corrupted!",
                step_idx, sha, e,
            )

    n = len(samples) or 1
    y_mc = sum(1 for s in samples if s.success) / n
    total_cost = sum(s.cost for s in samples)
    return MCResult(
        step_idx=step_idx,
        y_mc=y_mc,
        n_samples=len(samples),
        samples=samples,
        snapshot_sha=sha,
        total_cost=total_cost,
        total_wall_s=round(time.monotonic() - t0, 2),
        fork_step_cap=effective_cap,
    )
