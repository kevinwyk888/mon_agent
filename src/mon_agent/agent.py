"""MonitoringAgent — wraps DefaultAgent with step-level logging and prefix tracking."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent

from mon_agent.features.command_parser import classify_command, extract_target_file
from mon_agent.features.repetition import repeat_cmd_score, repeat_file_score
from mon_agent.features.signals import classify_obs_tag, compute_test_delta
from mon_agent.prefix import Prefix, build_prefix

logger = logging.getLogger(__name__)

WINDOW_SIZE = 3

OUTPUT_LIMITER_RE = re.compile(
    r"(\|\s*(head|tail|wc)\b|\|\s*sed\s+-n|"
    r"\|\s*grep\b.*\s(-m|--max-count)\b|(^|[^0-9])>\s*\S)",
    re.IGNORECASE,
)
RECURSIVE_LS_RE = re.compile(
    r"(^|[;&|]\s*)ls\s+[^;&\n]*(-[A-Za-z]*R[A-Za-z]*|--recursive)\b"
)
FIND_ROOT_RE = re.compile(r"(^|[;&|]\s*)find\s+(\.|/testbed)(\s|$)")
SELECTIVE_FIND_RE = re.compile(r"\s-(name|iname|regex)\s+", re.IGNORECASE)

GUARD_MESSAGE = """Command blocked before execution because it is likely to produce a very large observation and exhaust the model context.

Use a bounded command instead. Examples:
- find . -maxdepth 2 -type f | head -200
- rg -n "keyword" .
- sed -n '1,160p' path/to/file
- python - <<'PY'
from pathlib import Path
for p in list(Path('.').rglob('*.py'))[:200]:
    print(p)
PY

Blocked command:
{command}
"""


class MonitoringAgent(DefaultAgent):
    """DefaultAgent + per-step signal logging and prefix construction.

    Every call to :meth:`step` appends a structured record to
    ``self.step_logs``.  At save time the step log and final prefix are
    written alongside the normal trajectory file.
    """

    def __init__(self, *args: Any, task: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.step_logs: list[dict[str, Any]] = []
        self._task_text: str = task
        self._failure_streak: int = 0
        self._last_test_output: str = ""

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def step(self) -> list[dict]:
        # Snapshot state before the step
        cost_before = self.cost
        n_calls_before = self.n_calls
        t0 = time.monotonic()

        # Execute the real step (query + execute_actions)
        result = super().step()
        step_wall_s = time.monotonic() - t0

        # Collect signals from messages appended by super().step()
        self._log_step(cost_before, n_calls_before, step_wall_s)

        # Incrementally save step logs alongside the trajectory.
        # DefaultAgent.run() already saves the traj after each step via
        # output_path; here we also append the latest step record so that
        # data is never lost even if the job is killed.
        if self.config.output_path:
            step_log_path = Path(str(self.config.output_path)).with_suffix(".steps.jsonl")
            step_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(step_log_path, "a") as f:
                f.write(json.dumps(self.step_logs[-1]) + "\n")

        return result

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions, blocking commands that would flood the context."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            guard_output = self._guard_action(action)
            outputs.append(
                guard_output if guard_output is not None else self.env.execute(action)
            )
        return self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )

    def _guard_action(self, action: dict[str, Any]) -> dict[str, Any] | None:
        command = action.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return None

        if OUTPUT_LIMITER_RE.search(command):
            return None

        reason = ""
        if RECURSIVE_LS_RE.search(command):
            reason = "recursive ls without an output limiter"
        elif self._is_broad_find(command):
            reason = "broad find without an output limiter"

        if not reason:
            return None

        logger.info("Blocked command guard (%s): %s", reason, command)
        return {
            "output": GUARD_MESSAGE.format(command=command),
            "returncode": 2,
            "exception_info": "",
            "extra": {"guarded_command": True, "guard_reason": reason},
        }

    def _is_broad_find(self, command: str) -> bool:
        if not FIND_ROOT_RE.search(command):
            return False
        return not SELECTIVE_FIND_RE.search(command)

    # ------------------------------------------------------------------
    # Step logging
    # ------------------------------------------------------------------

    def _log_step(self, cost_before: float, n_calls_before: int, step_wall_s: float = 0.0) -> None:
        """Extract signals from the messages added during this step."""
        step_idx = self.n_calls  # 1-indexed (incremented inside query())

        # --- Find the assistant message and observation(s) from this step ---
        # Messages layout after a step:
        #   ... [assistant_msg] [tool_msg_1] [tool_msg_2] ...
        # We walk backwards to find the most recent assistant message.
        assistant_msg: dict[str, Any] = {}
        obs_msgs: list[dict[str, Any]] = []

        for msg in reversed(self.messages):
            role = msg.get("role", "")
            if role == "assistant" and not assistant_msg:
                assistant_msg = msg
                break
            elif role == "tool":
                obs_msgs.append(msg)

        obs_msgs.reverse()  # Chronological order

        # --- Extract action info ---
        actions = assistant_msg.get("extra", {}).get("actions", [])
        command_raw = actions[0].get("command", "") if actions else ""
        command_type = classify_command(command_raw)
        target_file = extract_target_file(command_raw)

        # --- Extract observation info (use first action's result) ---
        # The observation is embedded in the tool message content by the model's
        # observation_template.  We also look at the raw output stored during
        # execute_actions.
        returncode = 0
        output_text = ""
        exception_info = ""
        output_elided_chars = 0

        if obs_msgs:
            content = obs_msgs[0].get("content", "")
            output_text = content

            # Try to extract returncode from the template output
            import re

            rc_match = re.search(r"<returncode>(\-?\d+)</returncode>", content)
            if rc_match:
                returncode = int(rc_match.group(1))
            exc_match = re.search(
                r"<exception>(.*?)</exception>", content, re.DOTALL
            )
            if exc_match:
                exception_info = exc_match.group(1).strip()
            # When the observation template truncates a runaway command, it
            # emits "<elided_chars>N characters elided</elided_chars>". A large
            # N is itself a degradation signal worth logging.
            elided_match = re.search(
                r"<elided_chars>\s*(\d+)\s+characters elided", content
            )
            if elided_match:
                output_elided_chars = int(elided_match.group(1))

        # --- Derived signals ---
        obs_tag = classify_obs_tag(output_text, returncode, exception_info)

        # Failure streak
        is_failed = returncode != 0 or bool(exception_info)
        self._failure_streak = (self._failure_streak + 1) if is_failed else 0

        # Repetition scores (over recent window)
        recent_cmds = [s["command_raw"] for s in self.step_logs[-WINDOW_SIZE:]]
        rep_cmd = repeat_cmd_score(command_raw, recent_cmds)
        rep_file = repeat_file_score(command_raw, recent_cmds)

        # Test delta
        test_delta = compute_test_delta(output_text, self._last_test_output)
        if command_type == "test":
            self._last_test_output = output_text

        # Context length (rough estimate: total chars in messages)
        context_len_cum = sum(len(str(m.get("content", ""))) for m in self.messages)

        # Token counts from the LLM response
        extra = assistant_msg.get("extra", {})
        usage = extra.get("response", {}).get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Cumulative token counts
        prev_prompt_cum = self.step_logs[-1]["prompt_tokens_cum"] if self.step_logs else 0
        prev_completion_cum = self.step_logs[-1]["completion_tokens_cum"] if self.step_logs else 0

        # --- Build record ---
        record: dict[str, Any] = {
            "step_idx": step_idx,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_cum": prev_prompt_cum + prompt_tokens,
            "completion_tokens_cum": prev_completion_cum + completion_tokens,
            "step_wall_s": round(step_wall_s, 2),
            "context_len_cum": context_len_cum,
            "command_raw": command_raw,
            "command_type": command_type,
            "target_file": target_file,
            "returncode": returncode,
            "exception_flag": int(bool(exception_info)),
            "output_len": len(output_text),
            "output_elided_chars": output_elided_chars,
            "obs_tag": obs_tag,
            "test_delta": test_delta,
            "repeat_cmd_score_recent": round(rep_cmd, 3),
            "repeat_file_score_recent": round(rep_file, 3),
            "failure_streak": self._failure_streak,
        }

        self.step_logs.append(record)
        logger.info("Step %d: %s | %s | rc=%d | %s", step_idx, command_type, obs_tag, returncode, target_file)

    # ------------------------------------------------------------------
    # Prefix access
    # ------------------------------------------------------------------

    def get_prefix(self) -> Prefix:
        """Build the current prefix representation."""
        return build_prefix(self._task_text, self.step_logs, window_size=WINDOW_SIZE)

    # ------------------------------------------------------------------
    # Serialization — extend parent to include step logs + prefix
    # ------------------------------------------------------------------

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        data["step_logs"] = self.step_logs
        data["prefix_final"] = self.get_prefix().to_dict() if self.step_logs else None
        return data

    def save_step_logs(self, path: Path) -> None:
        """Write step logs as JSONL to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self.step_logs:
                f.write(json.dumps(record) + "\n")
