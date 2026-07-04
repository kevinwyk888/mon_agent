"""Binary-tree branching Monte-Carlo search.

Replaces the flat "main + M independent forks every K steps" scheme with a
proper recursive tree:

  * At every K steps the current prefix splits into B (default 2) children.
  * Each child runs K more steps from the snapshot, then splits again.
  * Recursion stops on a branch when:
      - the branch terminates (Submitted / LimitsExceeded raised by mini-swe-agent
        for any reason / uncaught exception), OR
      - cumulative step count reaches ``step_budget`` (default 60).
  * Leaf success = harness ``resolved`` (if eval enabled and patch non-empty)
    else proxy ``submission != "" and exit_status == "Submitted"``.
  * ``y(leaf) = 1.0 if success else 0.0``,
    ``y(internal) = mean(y(child) for child in children)``.

Output (next to the trajectory file):
  * ``<inst>.tree.json``   – full nested tree with all node info
  * ``<inst>.tree.jsonl``  – one line per node (flattened) carrying
                              ``node_id``, ``step_start``, ``step_end``,
                              ``features`` (per-step records), ``y``, ``success``.

Notes
-----
* All branches share the single Singularity sandbox; they are evaluated
  sequentially with snapshot/restore between siblings.
* The root node (id ``"0"``) runs the very first segment (steps 1..K) from
  the initial state; from step K onward we are in a pure tree (no main path).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from minisweagent.exceptions import InterruptAgentFlow

from mon_agent.mc_fork import (
    clone_model_with_sampling,
    restore_workdir,
    snapshot_workdir,
)

if TYPE_CHECKING:  # pragma: no cover
    from mon_agent.agent import MonitoringAgent

logger = logging.getLogger(__name__)


# Branching factor is fixed: every internal node splits into exactly two children.
_BRANCHING = 2

# Per-child sampling temperatures are configurable via TreeSearchConfig.
# The last digit of node_id selects which one applies: ".0" → left,
# ".1" → right. The root has no parent split and uses the left temperature.
_TOP_P = 1.0


def _temperature_for(node_id: str, cfg: "TreeSearchConfig") -> float:
    """Return the sampling temperature for a node based on ``cfg``.

    node_id is a dotted path like ``"0"``, ``"0.0"``, ``"0.1.0"``. The last
    component is the child index (0 = left, 1 = right). The root ``"0"`` is
    treated as a left-style baseline.
    """
    last = node_id.rsplit(".", 1)[-1]
    return cfg.temperature_right if last == "1" else cfg.temperature_left


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TreeSearchConfig:
    """Settings for binary-tree MC search."""

    enabled: bool = False
    fork_every: int = 10
    """K – split every K steps."""
    step_budget: int = 60
    """Hard cap on total steps along any root→leaf path."""
    fork_cost_budget: float = 5.0
    """Max additional $ cost per child segment (between two splits)."""
    snapshot_cwd: str = "/testbed"
    seed_base: int = 1234
    temperature_left: float = 0.2
    """Sampling temperature for left ('.0') children and the root."""
    temperature_right: float = 0.6
    """Sampling temperature for right ('.1') children."""

    # ------------------------------------------------------------------
    # SWE-bench harness evaluation (real success signal for y at leaves)
    # ------------------------------------------------------------------
    evaluate_with_harness: bool = False
    dataset_name: str = "princeton-nlp/SWE-bench_Lite"
    harness_run_id_prefix: str = "tree_eval"
    harness_max_workers: int = 1
    harness_timeout_s: int = 1800
    harness_namespace: str = ""
    harness_work_dir: str = ""
    harness_python: str = ""
    harness_keep_logs: bool = False
    # Only "singularity" is supported (HPC clusters w/o docker).
    # Field kept for config compatibility; value is ignored.
    harness_backend: str = "singularity"
    harness_sif_cache_dir: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TreeSearchConfig":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    node_id: str
    depth: int
    step_start: int
    step_end: int
    step_logs: list[dict] = field(default_factory=list)
    children: list["TreeNode"] = field(default_factory=list)
    terminated: bool = False
    """True if this segment ended via an `exit` message (Submitted or otherwise)."""
    exit_status: str = ""
    submission: str = ""
    proxy_success: bool = False
    harness_evaluated: bool = False
    harness_resolved: bool | None = None
    harness_error: str = ""
    harness_wall_s: float = 0.0
    harness_report_path: str = ""
    success: bool = False
    """Final leaf-level success used to compute y."""
    y: float = 0.0
    error: str = ""
    n_steps: int = 0
    cost: float = 0.0
    wall_s: float = 0.0
    temperature: float = 0.0
    """Sampling temperature used to roll this segment out (0.0 left, 0.3 right)."""
    alarm_events: list[dict] = field(default_factory=list)
    """Per-step alarm monitor records emitted during this segment (may be empty)."""
    n_interventions: int = 0
    """Deprecated intervention count field kept for tree-CSV compatibility."""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_nested_dict(self) -> dict:
        d = asdict(self)
        d["children"] = [c.to_nested_dict() for c in self.children]
        return d

    def iter_flat(self):
        """Yield (node, parent_id) for every node in the subtree."""
        stack: list[tuple["TreeNode", str | None]] = [(self, None)]
        while stack:
            node, parent_id = stack.pop()
            yield node, parent_id
            for c in node.children:
                stack.append((c, node.node_id))

    def to_flat_record(self, parent_id: str | None) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": parent_id,
            "depth": self.depth,
            "step_start": self.step_start,
            "step_end": self.step_end,
            "n_steps": self.n_steps,
            "is_leaf": not self.children,
            "n_children": len(self.children),
            "terminated": self.terminated,
            "exit_status": self.exit_status,
            "success": self.success,
            "proxy_success": self.proxy_success,
            "harness_evaluated": self.harness_evaluated,
            "harness_resolved": self.harness_resolved,
            "harness_error": self.harness_error,
            "harness_wall_s": self.harness_wall_s,
            "harness_report_path": self.harness_report_path,
            "y": round(self.y, 4),
            "submission": self.submission,
            "cost": round(self.cost, 4),
            "wall_s": round(self.wall_s, 2),
            "temperature": self.temperature,
            "error": self.error,
            "n_interventions": self.n_interventions,
            "features": self.step_logs,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _seed_root_messages(agent: "MonitoringAgent", task: str) -> None:
    """Mirror DefaultAgent.run's prompt-seeding without entering its loop."""
    agent.extra_template_vars |= {"task": task}
    agent.messages = []
    agent.add_messages(
        agent.model.format_message(
            role="system", content=agent._render_template(agent.config.system_template)
        ),
        agent.model.format_message(
            role="user", content=agent._render_template(agent.config.instance_template)
        ),
    )


def _make_segment_agent(
    parent_env: Any,
    fork_model: Any,
    template_agent: "MonitoringAgent",
    parent_messages: list[dict],
    parent_n_calls: int,
    parent_cost: float,
    step_limit: int,
    cost_limit: float,
    instance_id: str,
    node_id: str = "0",
    depth: int = 0,
    temperature: float = 0.0,
    path_prefix_logs: list[dict] | None = None,
) -> "MonitoringAgent":
    """Build a MonitoringAgent that resumes from (parent_messages, parent_n_calls)."""
    from mon_agent.agent import MonitoringAgent  # local import to avoid cycles

    config_dict = template_agent.config.model_dump()
    config_dict.pop("output_path", None)  # never let segments overwrite trajectory
    config_dict["step_limit"] = step_limit
    config_dict["cost_limit"] = cost_limit

    seg = MonitoringAgent(
        fork_model,
        parent_env,
        task=getattr(template_agent, "_task_text", ""),
        is_fork=True,  # disable nested splitting hooks
        instance_id=instance_id,
        **config_dict,
    )
    seg.messages = copy.deepcopy(parent_messages)
    seg.cost = parent_cost
    seg.n_calls = parent_n_calls
    seg.extra_template_vars = copy.deepcopy(
        getattr(template_agent, "extra_template_vars", {})
    )
    seg.extra_template_vars.setdefault(
        "task", getattr(template_agent, "_task_text", "")
    )
    # Node context + ancestor prefix so the alarm monitor sees the full path.
    seg._node_id = node_id
    seg._node_depth = depth
    seg._node_temperature = temperature
    seg._path_prefix_logs = list(path_prefix_logs or [])
    return seg


def _drive_until_split(
    seg: "MonitoringAgent",
    progress_cb: Callable[["MonitoringAgent"], None] | None = None,
) -> tuple[bool, str, str, str]:
    """Step the segment until either (a) it raises an exit message
    (Submitted / LimitsExceeded / etc.) or (b) it hits its step_limit which
    will itself produce a LimitsExceeded exit on the next query.

    Returns (terminated, exit_status, submission, error_str). ``terminated``
    means an exit message was appended; the caller still has to inspect
    ``exit_status`` to know whether the run is permanently done (``Submitted``)
    or merely paused at the split point (``LimitsExceeded``).
    """
    error_str = ""
    try:
        while True:
            if progress_cb is not None:
                try:
                    progress_cb(seg)
                except Exception:  # pragma: no cover - progress is best-effort
                    pass
            try:
                seg.step()
            except InterruptAgentFlow as e:
                seg.add_messages(*e.messages)
            except Exception as e:
                seg.handle_uncaught_exception(e)
                break
            if seg.messages and seg.messages[-1].get("role") == "exit":
                break
    except Exception as outer:  # last-resort safety net
        error_str = f"{type(outer).__name__}: {outer}"
        logger.warning("Tree segment crashed: %s", error_str)

    last_extra: dict = {}
    if seg.messages:
        last_extra = seg.messages[-1].get("extra", {}) or {}
    submission = last_extra.get("submission", "") or ""
    exit_status = last_extra.get("exit_status", "") or ""
    terminated = bool(seg.messages and seg.messages[-1].get("role") == "exit")
    return terminated, exit_status, submission, error_str


def _strip_trailing_exit(messages: list[dict]) -> list[dict]:
    """Return a copy of *messages* with any trailing ``role="exit"`` removed.

    A LimitsExceeded message is appended at the split point but must NOT be
    inherited by children (otherwise their loop exits immediately).
    """
    if messages and messages[-1].get("role") == "exit":
        return list(messages[:-1])
    return list(messages)


def _evaluate_leaf(
    cfg: TreeSearchConfig,
    instance_id: str,
    submission: str,
    exit_status: str,
    parent_step_idx: int,
    node_id: str,
    model_name: str,
) -> tuple[bool, bool, bool | None, str, float, str]:
    """Returns (success, proxy_success, harness_resolved, harness_error, harness_wall_s, harness_report_path)."""
    proxy = bool(submission.strip()) and exit_status == "Submitted"
    if not (cfg.evaluate_with_harness and proxy):
        return proxy, proxy, None, "", 0.0, ""

    from mon_agent.evaluate_singularity import evaluate_submission as _eval
    backend = "singularity"

    safe_node = node_id.replace(".", "_")
    run_id = (
        f"{cfg.harness_run_id_prefix}__{instance_id}"
        f"__step{parent_step_idx}__n{safe_node}"
    )
    eval_kwargs = dict(
        instance_id=instance_id,
        patch=submission,
        dataset_name=cfg.dataset_name,
        run_id=run_id,
        model_name=model_name,
        max_workers=cfg.harness_max_workers,
        timeout_s=cfg.harness_timeout_s,
        namespace=cfg.harness_namespace or None,
        work_dir=cfg.harness_work_dir or None,
        python_exe=cfg.harness_python or None,
        keep_work_dir=cfg.harness_keep_logs,
    )
    if backend == "singularity" and cfg.harness_sif_cache_dir:
        eval_kwargs["sif_cache_dir"] = cfg.harness_sif_cache_dir
    try:
        ev = _eval(**eval_kwargs)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Harness eval crashed (%s/%s): %s", instance_id, node_id, e)
        return False, proxy, None, f"eval_crash:{type(e).__name__}:{e}", 0.0, ""

    return (
        bool(ev.resolved),
        proxy,
        bool(ev.resolved),
        ev.error,
        ev.wall_s,
        ev.report_path,
    )


def _propagate_y(node: TreeNode) -> float:
    if not node.children:
        node.y = 1.0 if node.success else 0.0
        return node.y
    node.y = sum(_propagate_y(c) for c in node.children) / len(node.children)
    return node.y


# ---------------------------------------------------------------------------
# Recursive expansion
# ---------------------------------------------------------------------------


def _expand_node(
    *,
    template_agent: "MonitoringAgent",
    cfg: TreeSearchConfig,
    instance_id: str,
    snapshot_sha: str,
    step_start: int,
    parent_messages: list[dict],
    parent_n_calls: int,
    parent_cost: float,
    node_id: str,
    depth: int,
    model_name: str,
    progress_cb: Callable[[str, "MonitoringAgent"], None] | None = None,
    path_prefix_logs: list[dict] | None = None,
) -> TreeNode:
    """Run one segment from a snapshot; recurse if it ends at a split point."""
    t0 = time.monotonic()
    path_prefix_logs = list(path_prefix_logs or [])

    # Restore env to snapshot before running this segment.
    try:
        restore_workdir(template_agent.env, cfg.snapshot_cwd, snapshot_sha)
    except Exception as e:
        logger.warning("Tree restore failed at node %s: %s", node_id, e)
        return TreeNode(
            node_id=node_id, depth=depth,
            step_start=step_start, step_end=step_start,
            terminated=False, success=False, y=0.0,
            error=f"restore_failed:{type(e).__name__}:{e}",
            wall_s=round(time.monotonic() - t0, 2),
        )

    # Diversify: every node gets its own seed derived from its node_id.
    # Use a stable hash (sha256) so the same node_id always maps to the same
    # seed across runs — Python's built-in ``hash()`` for strings is salted
    # per-process via PYTHONHASHSEED and is therefore not reproducible.
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    seed = cfg.seed_base + int.from_bytes(digest[:4], "big")
    fork_model = clone_model_with_sampling(
        template_agent.model,
        temperature=_temperature_for(node_id, cfg),
        top_p=_TOP_P,
        seed=seed,
    )

    next_split = min(step_start + cfg.fork_every, cfg.step_budget)
    seg = _make_segment_agent(
        parent_env=template_agent.env,
        fork_model=fork_model,
        template_agent=template_agent,
        parent_messages=parent_messages,
        parent_n_calls=parent_n_calls,
        parent_cost=parent_cost,
        step_limit=next_split,
        cost_limit=parent_cost + cfg.fork_cost_budget,
        instance_id=instance_id,
        node_id=node_id,
        depth=depth,
        temperature=_temperature_for(node_id, cfg),
        path_prefix_logs=path_prefix_logs,
    )

    terminated, exit_status, submission, error_str = _drive_until_split(
        seg,
        progress_cb=(
            (lambda s: progress_cb(node_id, s)) if progress_cb is not None else None
        ),
    )

    node = TreeNode(
        node_id=node_id,
        depth=depth,
        step_start=step_start,
        step_end=seg.n_calls,
        step_logs=list(seg.step_logs),
        terminated=terminated,
        exit_status=exit_status,
        submission=submission,
        error=error_str,
        n_steps=max(0, seg.n_calls - parent_n_calls),
        cost=max(0.0, seg.cost - parent_cost),
        wall_s=round(time.monotonic() - t0, 2),
        temperature=_temperature_for(node_id, cfg),
        alarm_events=list(getattr(seg, "_alarm_events", [])),
        n_interventions=0,
    )

    # ------------------------------------------------------------------
    # Decide leaf vs. split
    # ------------------------------------------------------------------
    is_leaf = False
    reason = ""
    if exit_status == "Submitted":
        # Real success path: stop here, score it.
        is_leaf, reason = True, "submitted"
    elif seg.n_calls >= cfg.step_budget:
        is_leaf, reason = True, "budget_exhausted"
    elif terminated and exit_status != "LimitsExceeded":
        # Crash, FormatError, etc. — treat as failure leaf.
        is_leaf, reason = True, f"terminated:{exit_status or 'unknown'}"
    elif error_str:
        # Segment raised an uncaught exception (e.g. infra/API failure) and
        # ``handle_uncaught_exception`` did not append an ``exit`` message.
        # Without this branch the node would be misclassified as a normal
        # ``LimitsExceeded`` split point and we'd recurse on a corrupted
        # message history. Bail out as a failure leaf instead.
        is_leaf, reason = True, f"crash:{error_str[:120]}"

    if is_leaf:
        success, proxy, hres, herr, hwall, hrep = _evaluate_leaf(
            cfg=cfg,
            instance_id=instance_id,
            submission=submission,
            exit_status=exit_status,
            parent_step_idx=step_start,
            node_id=node_id,
            model_name=model_name,
        )
        node.success = success
        node.proxy_success = proxy
        node.harness_evaluated = cfg.evaluate_with_harness and proxy
        node.harness_resolved = hres
        node.harness_error = herr
        node.harness_wall_s = hwall
        node.harness_report_path = hrep
        logger.info(
            "Tree leaf %s depth=%d steps=%d..%d reason=%s success=%s y=%.2f",
            node_id, depth, step_start, seg.n_calls, reason, success,
            1.0 if success else 0.0,
        )
        return node

    # ------------------------------------------------------------------
    # Otherwise this node is an internal split point.
    # ------------------------------------------------------------------
    try:
        new_sha = snapshot_workdir(template_agent.env, cfg.snapshot_cwd)
    except Exception as e:
        logger.warning("Tree snapshot failed at node %s: %s — leaf", node_id, e)
        node.error = node.error or f"snapshot_failed:{type(e).__name__}:{e}"
        node.success = False
        return node

    # Children must NOT inherit the trailing exit message we got from
    # LimitsExceeded; otherwise their first while-loop iteration sees an
    # "exit" role and bails out.
    child_messages = _strip_trailing_exit(seg.messages)
    child_n_calls = seg.n_calls
    child_cost = seg.cost
    # Children inherit this segment's steps so the alarm monitor scores the
    # full root->node prefix window, not just their own local segment.
    child_prefix_logs = path_prefix_logs + list(seg.step_logs)

    for b in range(_BRANCHING):
        child_id = f"{node_id}.{b}"
        child = _expand_node(
            template_agent=template_agent,
            cfg=cfg,
            instance_id=instance_id,
            snapshot_sha=new_sha,
            step_start=child_n_calls,
            parent_messages=child_messages,
            parent_n_calls=child_n_calls,
            parent_cost=child_cost,
            node_id=child_id,
            depth=depth + 1,
            model_name=model_name,
            progress_cb=progress_cb,
            path_prefix_logs=child_prefix_logs,
        )
        node.children.append(child)

    return node


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _pick_representative_submission(root: TreeNode) -> tuple[str, str, str]:
    """Choose one submission to put in preds.json.

    Priority:
      1. any leaf with success=True (highest depth = most-evolved patch)
      2. any leaf with non-empty submission
      3. empty
    """
    successes: list[TreeNode] = []
    nonempty: list[TreeNode] = []
    for node, _ in root.iter_flat():
        if node.children:
            continue
        if node.success and node.submission.strip():
            successes.append(node)
        elif node.submission.strip():
            nonempty.append(node)

    if successes:
        # Prefer the deepest (most steps invested) successful leaf.
        node = max(successes, key=lambda n: n.depth)
        return node.submission, node.exit_status or "Submitted", node.node_id
    if nonempty:
        node = max(nonempty, key=lambda n: n.depth)
        return node.submission, node.exit_status, node.node_id
    return "", "TreeNoSubmission", ""


def run_tree(
    agent: "MonitoringAgent",
    task: str,
    cfg: TreeSearchConfig,
    instance_id: str,
    output_dir: Path | None = None,
    progress_cb: Callable[[str, "MonitoringAgent"], None] | None = None,
) -> tuple[TreeNode, dict]:
    """Build the binary search tree starting from the agent's initial state.

    Parameters
    ----------
    agent : MonitoringAgent
        Used as the *template* (env, model, config). It must NOT have been
        run before — we seed system+instance prompts ourselves.
    task : str
        Task / problem statement passed to the prompt template.
    cfg : TreeSearchConfig
    instance_id : str
        SWE-bench instance id (used for harness eval and run_id).
    output_dir : Path | None
        If provided, writes ``<inst>.tree.json`` and ``<inst>.tree.jsonl``.

    Returns
    -------
    (root, info)
        ``root`` is the populated TreeNode; ``info`` is a dict matching the
        signature of ``DefaultAgent.run`` (``exit_status``, ``submission``)
        so it can be plugged into the runner without further changes.
    """
    if not cfg.enabled:
        raise ValueError("run_tree called with cfg.enabled=False")

    # Seed prompts on the (template) agent so its messages contain system+user.
    _seed_root_messages(agent, task)
    agent._task_text = task  # used by templates inside forks

    try:
        root_sha = snapshot_workdir(agent.env, cfg.snapshot_cwd)
    except Exception as e:
        logger.error("Tree root snapshot failed: %s", e)
        raise

    model_cfg = getattr(getattr(agent, "model", None), "config", None)
    model_name = getattr(model_cfg, "model_name", None) or "tree_fork"

    t0 = time.monotonic()
    root = _expand_node(
        template_agent=agent,
        cfg=cfg,
        instance_id=instance_id,
        snapshot_sha=root_sha,
        step_start=0,
        parent_messages=list(agent.messages),
        parent_n_calls=0,
        parent_cost=0.0,
        node_id="0",
        depth=0,
        model_name=model_name,
        progress_cb=progress_cb,
    )
    # Always restore env to root snapshot so the agent's env is in a known state.
    try:
        restore_workdir(agent.env, cfg.snapshot_cwd, root_sha)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Tree final restore failed: %s", e)

    _propagate_y(root)
    total_wall = round(time.monotonic() - t0, 2)
    n_nodes = sum(1 for _ in root.iter_flat())
    n_leaves = sum(1 for n, _ in root.iter_flat() if not n.children)
    n_success = sum(1 for n, _ in root.iter_flat() if not n.children and n.success)
    n_interventions = sum(n.n_interventions for n, _ in root.iter_flat())
    logger.info(
        "Tree done: nodes=%d leaves=%d success=%d y_root=%.4f interventions=%d wall=%.1fs",
        n_nodes, n_leaves, n_success, root.y, n_interventions, total_wall,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        nested_path = output_dir / f"{instance_id}.tree.json"
        flat_path = output_dir / f"{instance_id}.tree.jsonl"
        nested_path.write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "config": asdict(cfg),
                    "stats": {
                        "n_nodes": n_nodes,
                        "n_leaves": n_leaves,
                        "n_success": n_success,
                        "n_interventions": n_interventions,
                        "y_root": round(root.y, 4),
                        "wall_s": total_wall,
                        "snapshot_sha": root_sha,
                    },
                    "tree": root.to_nested_dict(),
                },
                indent=2,
            )
        )
        with open(flat_path, "w") as f:
            for node, parent_id in root.iter_flat():
                f.write(json.dumps(node.to_flat_record(parent_id)) + "\n")

    submission, exit_status, picked_id = _pick_representative_submission(root)
    info = {
        "exit_status": exit_status if submission else "TreeNoSubmission",
        "submission": submission,
        "tree_y_root": root.y,
        "tree_n_leaves": n_leaves,
        "tree_n_success": n_success,
        "tree_n_interventions": n_interventions,
        "tree_picked_node": picked_id,
    }
    return root, info
