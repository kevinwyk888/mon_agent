
# Feasible Method for Prefix-Level Degradation Monitoring on mini-SWE-agent

## Goal

Build a practical first-stage monitoring system for `mini-swe-agent` on `SWE-bench Lite` that can detect hidden degradation **before terminal failure**.

The core idea is:

1. collect **step-level observables** from each agent step,
2. define a **prefix-level score** \(Y_k\) that estimates future solvability,
3. monitor the sequence of \(Y_k\) or its increments online,
4. trigger simple intervention such as **early stop** when persistent degradation is detected.

This is intentionally designed as the **simplest viable version** that can be implemented and tested first.

---

## Why mini-SWE-agent

`mini-swe-agent` is a good pilot platform because:

- the workflow is extremely simple and linear,
- the message history is directly the trajectory,
- each step is essentially:
  - model query,
  - parse action,
  - execute command,
  - append observation,
- the environment returns structured execution results such as `output`, `returncode`, and `exception_info`,
- the agent already tracks step count and cost.

This makes it easy to insert logging and monitoring.

---

## Prefix Definition

Use the following **minimal, practical prefix definition** after step \(k\):

\[
\text{prefix}_k = (\text{issue summary},\ \text{recent window summary},\ \text{global counters})
\]

### 1. Issue summary
Always keep:

- original issue / task description
- optional: task ID and repo name

### 2. Recent window summary
Use only the **most recent 3 steps**.

For each recent step, keep a compact step card:

- `action_type`
- `target_file`
- `returncode`
- `obs_tag`
- `test_delta`

where:

- `action_type` is a coarse category such as `read`, `search`, `edit`, `test`
- `target_file` is the main file touched by the command
- `returncode` is command exit status
- `obs_tag` is a short label such as `success`, `test_fail`, `syntax_error`, `no_new_info`
- `test_delta` is `improved`, `unchanged`, `worse`, or `NA`

### 3. Global counters
Keep only these 6 counters:

- `step_idx`
- `cost_cum`
- `context_len_cum`
- `repeat_cmd_score_recent`
- `repeat_file_score_recent`
- `failure_streak`

This is the recommended **minimum prefix representation**.

---

## Step-Level Signals to Collect

For each step \(k\), log the following minimal set.

### Basic identity
- `step_idx`

### Cost and context
- `cost_step`
- `cost_cum`
- `context_len_cum`

### Action
- `command_raw`
- `command_type`
- `target_file`

### Repetition
- `repeat_cmd_score_recent`
- `repeat_file_score_recent`

### Execution result
- `returncode`
- `exception_flag`
- `output_len`

### Error / progress
- `obs_tag`
- `test_delta`
- `failure_streak`

### Optional semantic score
- `issue_action_alignment`

---

## Recommended Definitions for Key Signals

### `repeat_cmd_score_recent`
This measures whether the current command is repetitive relative to the recent window.

A practical first implementation:

- window size \(w = 3\)
- normalize the command string
- compare with recent commands of the same type

Example:

\[
\text{repeat\_cmd\_score\_recent}(k)
=
\frac{1}{w-1}\sum_{j=k-w}^{k-1} \mathbf{1}(\text{norm}(c_j)=\text{norm}(c_k))
\]

In the simplest version, use repeated **command type** instead of exact command.

### `repeat_file_score_recent`
This measures whether the current target file has been repeatedly touched in the recent window.

Simple version:

- extract the main file path from the command,
- compute the fraction of recent steps that touch the same file.

### `failure_streak`
This is the number of consecutive failed steps.

Define a step as failed if:

- `returncode != 0`, or
- `exception_flag == 1`

Then:

- if failed: increment streak by 1
- else: reset to 0

---

## Formal Definition of Y_k

Main definition:

\[
Y_k := \Pr(\text{eventual success} \mid \text{prefix}_k)
\]

Interpretation: after step \(k\), how likely is the task still solvable from the current prefix?

---

## How to Estimate Y_k — Recursive Binary Branching

Instead of a flat "main + M independent forks every K steps", we use a **recursive binary tree** of rollouts that share one Singularity sandbox via git snapshot/restore (see `src/mon_agent/tree_search.py`).

### Tree construction

- Every \(K\) steps the current prefix splits into \(B=2\) children:
  - **left child**: temperature 0.0 (deterministic baseline),
  - **right child**: temperature 0.3 (diverse sibling).
- Each child runs \(K\) more steps from the parent's git snapshot, then splits again.
- Recursion stops on a branch when:
  - the branch terminates (`Submitted` / limits exceeded / exception), or
  - cumulative step count reaches `step_budget` (default 60).
- Node IDs are dotted paths (`0`, `0.0`, `0.1.0`, ...); the last digit selects left/right and therefore the temperature.

### Leaf and internal labels

- **Leaf success**: harness `resolved` if eval enabled and the patch is non-empty, else the proxy `submission != "" and exit_status == "Submitted"`.
- **Leaf score**: \(y(\text{leaf}) = 1\) if success else \(0\).
- **Internal score**: \(y(v) = \operatorname{mean}\bigl(y(c) : c \in \text{children}(v)\bigr)\).

This gives a tree-MC estimate

\[
\hat Y_k^{\text{tree}}(v) = \text{average leaf success in the subtree rooted at } v.
\]

for every node \(v\) covering steps \([s_v, e_v]\).

### Per-step Y_k (global view)

For a single instance, we also collapse the tree to a per-step score by averaging over all paths whose prefix contains step \(k\):

\[
Y_k^{\text{global}}
= \frac{\#\{\text{leaves whose ancestor chain covers step } k \text{ and that succeeded}\}}
        {\#\{\text{leaves whose ancestor chain covers step } k\}}
\]

This is what `results/compute_y_global.py` produces from `<inst>.tree.jsonl`.

---

## Phased Training Target

1. **Phase 1 (cheap label)**: use eventual success of the original run as a per-prefix label
   \(\tilde Y_k = \mathbf{1}(\text{run succeeds})\) and fit \(\hat Y_k = f_\theta(\text{prefix}_k)\).
2. **Phase 2 (tree label)**: replace the label with \(\hat Y_k^{\text{tree}}\) / \(Y_k^{\text{global}}\) computed from the binary tree above.

Start from Phase 1; switch to Phase 2 once a small batch of trees is collected. Do **not** start from degradation-risk labels — they are much harder to define cleanly.

---

## Recommended Model for Y_k

Start with a lightweight tabular or shallow model:

- logistic regression
- XGBoost
- small MLP

Input:
- structured prefix features only

Do **not** start with a full-text large model predictor in the first stage.

The purpose of the pilot is to validate whether the chosen observables contain useful information.

---

## Online Monitoring Strategy

Do not start with residual modeling.

Use one of these directly:

### Option A: monitor \(\hat Y_k\)
If \(\hat Y_k\) stays low for several steps, flag degradation.

### Option B: monitor progress increments
Define:

\[
\Delta_k = \hat Y_k - \hat Y_{k-1}
\]

Interpretation:
- positive: progress
- near zero: stagnation
- negative: degradation

This is highly interpretable.

### Option C: simple CUSUM on \(\Delta_k\)
Define:

\[
S_k = \max\{0, S_{k-1} + (c - \Delta_k)\}
\]

If \(S_k\) exceeds a threshold, trigger intervention.

This helps distinguish persistent degradation from one noisy bad step.

---

## Recommended First Intervention

Use only **early stop** in the first stage.

Reason:
- easiest to implement,
- easiest to evaluate,
- directly measures whether the monitor avoids wasting steps and cost on obviously bad trajectories.

Later, a second intervention can be added:
- replan once,
- or compress / reset context.

---

## Recommended Experiment Sequence

### Stage 1
Run `mini-swe-agent` on a small subset of `SWE-bench Lite` with the **binary tree runner** (`tree_search.py`) and collect, per instance:
- `<inst>.steps.jsonl` (per-step features for every node),
- `<inst>.tree.json` / `<inst>.tree.jsonl` (full tree with `y`, `success`).

### Stage 2
Flatten trees into prefix samples and train a cheap \(\hat Y_k\) predictor, first against eventual-success labels, then against tree labels \(\hat Y_k^{\text{tree}}\) / \(Y_k^{\text{global}}\).

### Stage 3
Check whether \(\hat Y_k\) or \(\Delta_k\) correlates with hidden degradation patterns such as repeated command/file loops, persistent execution failures, or long context growth with no test improvement — and whether **left/right sibling divergence** in \(y\) flags risky prefixes early.

### Stage 4
Add a simple monitoring rule and compare:
- baseline (single trajectory),
- monitor-only,
- monitor + early stop,
- monitor + branch-pruning inside the tree (drop low-\(y\) subtrees early to save budget).

---

## Success Criteria for the Pilot

The pilot is successful if at least one of the following is true:

1. \(\hat Y_k\) clearly separates successful and failing prefixes.
2. \(\Delta_k\) becomes persistently negative before terminal failure.
3. early stop reduces wasted steps / cost on bad runs.
4. the monitor provides interpretable evidence of hidden degradation.

---

## Short Summary

The most feasible method is:

1. define a compact prefix from issue + recent 3-step window + 6 global counters,
2. estimate \(Y_k\) by a **recursive binary-branching tree** (B=2, temperatures 0.0 / 0.3, split every K steps, capped by `step_budget`), with leaf success from harness or `Submitted` proxy and internal nodes = mean of children,
3. train \(\hat Y_k = f_\theta(\text{prefix}_k)\) against the tree labels (or eventual-success labels as a Phase-1 shortcut),
4. monitor \(\hat Y_k\) or \(\Delta_k\) online and start with early stop as the first intervention.

This is the cleanest first implementation on `mini-swe-agent`.
