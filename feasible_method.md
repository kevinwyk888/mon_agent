
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

Use the following main definition:

\[
Y_k := \Pr(\text{eventual success} \mid \text{prefix}_k)
\]

Interpretation:

- after step \(k\), how likely is the task to still be solvable from the current prefix?

This is the recommended primary monitoring score.

---

## How to Estimate Y_k

### Phase 1: Cheap approximation
Use the eventual result of the original run as the prefix label.

For each prefix in a trajectory:

\[
\tilde Y_k = \mathbf{1}(\text{the run eventually succeeds})
\]

Then train a small predictor:

\[
\hat Y_k = f_\theta(\text{prefix}_k)
\]

This is coarse, but very easy to implement.

### Phase 2: Better approximation
Use continuation rollouts from intermediate prefixes.

For a saved prefix \(\text{prefix}_k\), continue the agent from that state for \(M\) runs under a fixed remaining budget:

\[
\hat Y_k^{MC} = \frac{1}{M}\sum_{m=1}^M \mathbf{1}(\text{continuation } m \text{ succeeds})
\]

Then train:

\[
\hat Y_k = f_\theta(\text{prefix}_k)
\]

This is closer to the true definition of future solvability.

---

## Recommended Training Target for the First Implementation

Start with:

\[
\hat Y_k = \Pr(\text{eventual success} \mid \text{prefix}_k)
\]

using the **cheap approximation** first.

Do **not** start with degradation-risk labeling first, because those labels are much harder to define cleanly.

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
Run `mini-swe-agent` on a small subset of `SWE-bench Lite` and collect trajectories.

### Stage 2
Extract prefix-level samples and train a cheap \(Y_k\) predictor using eventual success labels.

### Stage 3
Evaluate whether \(\hat Y_k\) or \(\Delta_k\) correlates with hidden degradation patterns such as:

- repeated command loops,
- repeated file loops,
- persistent execution failures,
- long context growth with no test improvement.

### Stage 4
Add a simple monitoring rule and compare:
- baseline
- monitor-only
- monitor + early stop

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
2. train \(Y_k\) as a future solvability score,
3. monitor \(Y_k\) or \(\Delta_k\) online,
4. start with early stop as the first intervention.

This is the cleanest first implementation on `mini-swe-agent`.
