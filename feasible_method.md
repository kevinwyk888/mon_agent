
# Feasible Method for Prefix-Level Degradation Monitoring on mini-SWE-agent

## Goal

Build a practical first-stage monitoring system for `mini-swe-agent` on `SWE-bench Lite` that can detect hidden degradation **before terminal failure**.

The core idea is:

1. collect **step-level observables** from each agent step,
2. estimate a **prefix-level score** \(Y_k\) that proxies future solvability via a recursive binary tree of rollouts,
3. use the per-(node, step) records as the basic dataset for any downstream monitor.

This document describes the **data-generation pipeline** (already implemented
in `src/mon_agent/tree_search.py` and `src/mon_agent/runner.py`) and **fixes
the experimental knobs** that we will hold constant for the first batch of
runs. How to actually consume the resulting CSVs — feature engineering,
predictor choice, intervention rule — is intentionally left open below.

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

- original issue / task description,
- task ID and repo name.

### 2. Recent window summary
Use only the **most recent 3 steps**.

For each recent step, keep a compact step card:

- `action_type` (`read`, `search`, `edit`, `run`, `test`, `other`),
- `target_file` (main file touched by the command),
- `returncode` (command exit status),
- `obs_tag` (`success`, `exception`, `test_fail`, `syntax_error`, ...),
- `test_delta` (`improved`, `unchanged`, `worse`, `NA`).

### 3. Global counters
Keep only these 6 counters:

- `step_idx`,
- `cost_cum` (proxied by `prompt_tokens_cum` + `completion_tokens_cum`),
- `context_len_cum`,
- `repeat_cmd_score_recent`,
- `repeat_file_score_recent`,
- `failure_streak`.

All step-level fields are emitted into the per-instance CSV described in
[Output CSV Format](#output-csv-format).

---

## Formal Definition of Y_k

Main definition:

\[
Y_k := \Pr(\text{eventual success} \mid \text{prefix}_k).
\]

Interpretation: after step \(k\), how likely is the task still solvable from
the current prefix?

---

## How to Estimate Y_k — Recursive Binary Branching

Instead of a flat "main + M independent forks every K steps", we use a
**recursive binary tree** of rollouts that share one Singularity sandbox via
`git` snapshot/restore (see `src/mon_agent/tree_search.py`).

### Tree construction

- Every \(K\) steps the current prefix splits into \(B = 2\) children:
  - **left child**: sampling temperature \(T_L\),
  - **right child**: sampling temperature \(T_R\).
- Each child runs \(K\) more steps from the parent's git snapshot, then splits again.
- Recursion stops on a branch when:
  - the branch terminates (`Submitted` / limits exceeded / exception), or
  - cumulative step count reaches `step_budget` (default 60).
- Node IDs are dotted paths (`0`, `0.0`, `0.1.0`, ...); the last digit
  selects left/right and therefore which of \(T_L, T_R\) was used.

### Fixed knob choice for the first batch

For the first generation pass we **freeze** the temperatures:

| knob | value | rationale |
|------|-------|-----------|
| \(T_L\) (left, `.0` children) | **0.3** | matches the default already used in `configs/swebench_lite.yaml` and `scripts/run_mc_fork.sbatch` |
| \(T_R\) (right, `.1` children) | **0.3** | same as \(T_L\); equal-temperature siblings keep \(y\) estimates comparable across branches and avoid temperature being a confound when comparing left/right divergence |
| `fork_every` (\(K\))           | 10    | already the sbatch default |
| `step_budget`                  | 60    | already the sbatch default |
| `fork_cost_budget`             | 5.0   | already the sbatch default |

Asymmetric temperatures (e.g. 0.0 vs 0.3) remain a future ablation but are
out of scope for the initial dataset.

### Leaf and internal labels

- **Leaf success**: harness `resolved` if eval enabled and the patch is non-empty, else the proxy `submission != "" and exit_status == "Submitted"`.
- **Leaf score**: \(y(\text{leaf}) = 1\) if success else \(0\).
- **Internal score**: \(y(v) = \operatorname{mean}\bigl(y(c) : c \in \text{children}(v)\bigr)\).

This gives a tree-MC estimate

\[
\hat Y_k^{\text{tree}}(v) = \text{average leaf success in the subtree rooted at } v
\]

for every node \(v\) covering steps \([s_v, e_v]\).

### Per-step Y_k (global view)

For a single instance we also collapse the tree to a per-step score by
averaging over all paths whose prefix contains step \(k\):

\[
Y_k^{\text{global}}
= \frac{\#\{\text{leaves whose ancestor chain covers step } k \text{ and that succeeded}\}}
        {\#\{\text{leaves whose ancestor chain covers step } k\}}.
\]

This is what `results/compute_y_global.py` produces from `<inst>.tree.jsonl`.

---

## Output CSV Format

The runner writes one CSV per instance into `${CSV_OUT_DIR}` as soon as that
instance's `tree.json` is on disk (see `mon_agent.tree_to_csv.convert_one`).
File name: `<instance_id>.steps.csv`. Schema is **one row per (node, step)**
across the entire tree, so each instance produces on the order of
(# tree nodes) × (segment length) rows (typically 50–150 rows for the default
knobs above; large trees can exceed 1k rows when many leaves are explored).

### Column dictionary

| # | column | type | description |
|---|--------|------|-------------|
| 1 | `instance_id` | str | SWE-bench task id, e.g. `django__django-11179`. |
| 2 | `node_id` | str | Dotted tree path. `0` = root, `0.1` = right child of root, `0.1.0` = left child of `0.1`, etc. The last digit picks the sibling (and thus the temperature). |
| 3 | `depth` | int | Tree depth (= number of dots in `node_id`). |
| 4 | `temperature` | float | Sampling temperature used for this node's segment (`T_L` or `T_R`). |
| 5 | `step_idx` | int | 1-based step index along this node's segment (resets at the start of each node). |
| 6 | `prompt_tokens` | int | Prompt tokens charged on this step. |
| 7 | `completion_tokens` | int | Completion tokens charged on this step. |
| 8 | `prompt_tokens_cum` | int | Cumulative prompt tokens **within this node's segment**. |
| 9 | `completion_tokens_cum` | int | Cumulative completion tokens within this node's segment. |
| 10 | `step_wall_s` | float | Wall-clock seconds for the step (LLM round-trip + tool exec). |
| 11 | `context_len_cum` | int | Conversation context length (chars) up to and including this step. |
| 12 | `command_type` | str | Coarse action tag (`read`, `search`, `edit`, `run`, `other`, ...). |
| 13 | `target_file` | str | Best-effort main target of the command (file path, module, or symbol). |
| 14 | `returncode` | int | Tool/exec return code (0 on success). |
| 15 | `exception_flag` | int | 1 if mini-swe-agent flagged an exception, else 0. |
| 16 | `output_len` | int | Length (chars) of the observation returned to the model. |
| 17 | `output_elided_chars` | int | Characters dropped by the observation-truncation rule (0 if not truncated). |
| 18 | `obs_tag` | str | Short observation label (`success`, `exception`, `test_fail`, ...). |
| 19 | `repeat_cmd_score_recent` | float | Fraction of the last 3 commands (excluding current) that match the current command. |
| 20 | `repeat_file_score_recent` | float | Same but on `target_file`. |
| 21 | `failure_streak` | int | Number of consecutive failed steps ending at this step. |
| 22 | `confidence` | float | Self-reported confidence in `[0, 1]` that the action just taken was a correct, helpful step toward solving the task. Produced by a separate one-shot LLM probe (system + user prompt, max 256 tokens, up to 2 attempts) called immediately after the step finishes. `NaN` when both attempts fail to emit a parseable number. Probe cost is added to the agent's running `cost` total. |
| 23 | `y` | float | Subtree success rate for **this node** (constant within a node's rows; equals the leaf success label at leaves). |

Two points worth remembering when consuming the CSV:

- Token / wall-time cumulatives are **per node**, not per path. To recover a
  root→leaf cumulative, walk `node_id` from the root and sum across
  segments.
- `y` is the **node-level** label \(y(v)\) defined above. To get the
  per-step `Y_k^{global}` use `results/compute_y_global.py` against
  `<inst>.tree.jsonl`; do not try to recompute it from `steps.csv` directly.

### Example rows

> Note: the excerpt below comes from `results/tree_run_50411867/`, which
> predates the `confidence` column. Newer runs (e.g. `tree_run_51265704`
> and later) include `confidence` as the second-to-last column, just
> before `y`.

We pick a **mixed-`y`** instance on purpose: trees where every leaf
succeeds (root `y = 1`) or every leaf fails (root `y = 0`) carry little
prefix-level signal, and the interesting cases are the ones where sibling
subtrees disagree under the same fixed temperature. The excerpt below is
from `results/tree_run_50411867/django__django-11815.steps.csv` (root
segment + first fork + the four grandchildren at the next fork; long rows
wrapped for display):

```csv
instance_id,node_id,depth,temperature,step_idx,prompt_tokens,completion_tokens,prompt_tokens_cum,completion_tokens_cum,step_wall_s,context_len_cum,command_type,target_file,returncode,exception_flag,output_len,output_elided_chars,obs_tag,repeat_cmd_score_recent,repeat_file_score_recent,failure_streak,y
django__django-11815,0,0,0.3,1,1590,211,1590,211,2.47,6874,read,,0,0,1021,0,success,0.0,0.0,0,0.59375
django__django-11815,0,0,0.3,2,2180,80,3770,291,1.25,7812,search,,0,0,938,0,success,0.0,0.0,0,0.59375
django__django-11815,0,0,0.3,3,2543,92,6313,383,1.41,14424,read,/testbed/django/db/migrations/serializer.py,0,0,6612,6415,success,0.5,0.0,0,0.59375
django__django-11815,0,0,0.3,9,6695,78,38341,1642,1.53,22451,edit,,0,0,1862,0,success,0.333,0.0,0,0.59375
django__django-11815,0,0,0.3,10,7271,80,45612,1722,1.65,23103,search,,0,0,652,0,success,0.333,0.0,0,0.59375
django__django-11815,0.0,1,0.3,11,7534,86,7534,86,1.56,24704,search,/testbed/tests/migrations/test_writer.py,0,0,1601,0,success,0.0,0.0,0,0.375
django__django-11815,0.1,1,0.3,11,7534,80,7534,80,1.38,24704,search,/testbed/tests/migrations/test_writer.py,0,0,1601,0,success,0.0,0.0,0,0.8125
django__django-11815,0.0,1,0.3,12,8054,840,15588,926,8.20,25104,other,/testbed/django/db/migrations/serializer.py,0,0,400,0,success,0.0,0.0,0,0.375
django__django-11815,0.1,1,0.3,12,8048,82,15582,162,1.39,26975,other,/testbed/tests/migrations/test_writer.py,0,0,2271,0,success,0.0,1.0,0,0.8125
django__django-11815,0.0.0,2,0.3,21,11970,498,11970,498,3.60,29408,run,django.conf,1,0,2024,0,exception,0.0,0.0,1,0.25
django__django-11815,0.0.1,2,0.3,21,11970,491,11970,491,3.64,27883,run,django.conf,0,0,499,0,success,0.0,0.0,0,0.5
django__django-11815,0.1.0,2,0.3,21,11748,200,11748,200,2.49,28896,run,django.db.migrations.serializer,1,0,380,0,exception,0.0,0.0,1,0.75
django__django-11815,0.1.1,2,0.3,21,11748,153,11748,153,2.62,28805,run,django.db.migrations.serializer,1,0,289,0,exception,0.0,0.0,1,0.875
```

What to read out of the example:

- **Root**: `node_id = 0` has `y = 0.59375`. Across the full tree under this
  prefix, ~60% of leaves resolved the SWE-bench task. This single number
  averages over everything that happens after step 10, including the very
  different left/right subtrees below.
- **First fork (depth 1)**: at step 11 the root splits into `0.0` (`y =
  0.375`) and `0.1` (`y = 0.8125`). Both children inherit the parent's
  sandbox via a git snapshot, both are sampled at \(T = 0.3\), and they
  even open with the same `search` command on the same file — yet their
  downstream success rates differ by more than 2×. **This is the raw
  material for any prefix-level monitor**: a per-step classifier that only
  looks at the row at step 11 sees identical features for the two
  siblings, so any useful signal must come from richer structure (sibling
  disagreement, sequence history, calibration to tree statistics).
- **Second fork (depth 2)**: the four grandchildren span `y ∈ {0.25, 0.5,
  0.75, 0.875}`. Notice that `0.1.0` opens with an **exception**
  (`returncode = 1`, `failure_streak = 1`) but still leads to a `y = 0.75`
  subtree, while `0.0.0` opens with the same kind of exception and ends up
  at `y = 0.25`. Local step-level failures therefore do not in themselves
  determine the prefix's solvability — the whole reason we are estimating
  `Y_k` from the tree rather than counting errors.
- **Within-node constancy of `y`**: every row of a given node carries the
  same `y` value (it is a node-level label, not a step-level one). Per-step
  variation, if needed, has to be reconstructed via `Y_k^{global}` from
  `<inst>.tree.jsonl`, **not** by aggregating `steps.csv` rows.

---

## Phased Training Target

1. **Phase 1 (cheap label)**: use eventual success of the original run as a
   per-prefix label \(\tilde Y_k = \mathbf{1}(\text{run succeeds})\) and fit
   \(\hat Y_k = f_\theta(\text{prefix}_k)\).
2. **Phase 2 (tree label)**: replace the label with \(\hat Y_k^{\text{tree}}\)
   / \(Y_k^{\text{global}}\) computed from the binary tree above.

Start from Phase 1; switch to Phase 2 once a small batch of trees is
collected. Do **not** start from degradation-risk labels — they are much
harder to define cleanly.

---

## Downstream Use of `steps.csv` — TBD

Intentionally left open for now. The dataset described above is the input;
the question of **what model / monitor consumes it** is unresolved. A few
non-binding observations to keep in mind when we come back to this:

- The feature set is small (≤ 21 numeric / categorical columns) and the
  cross-instance variance dominates within-instance variance, so naïve
  per-step classifiers (logistic regression, shallow XGBoost) on raw rows
  are expected to **mostly memorize instance-level priors** rather than
  detect prefix-level degradation.
- Useful structure that simple tabular learners do not capture out of the
  box:
  - sequential dependence (`failure_streak`, repeat scores already encode a
    little, but the full step sequence carries more),
  - tree topology and sibling-divergence signals (left vs right `y` gap at
    the same fork),
  - cross-instance normalization (token / context counters are not
    comparable across repos without it).
- A reasonable order of attempts, when we get to it, is roughly:
  sequence model over a single trajectory → per-fork sibling-disagreement
  features → calibration of the per-step prediction against the
  tree-derived \(Y_k^{global}\). None of this is committed; this section is
  a placeholder.

The intervention rule (early stop, replan, etc.) and the success criteria
for the monitor are likewise deferred until we have a candidate predictor.

---

## Recommended Experiment Sequence

### Stage 1 — data generation (this is what the current pipeline does)
Run `mini-swe-agent` on `SWE-bench Lite` with the binary tree runner
(`tree_search.py`) under the fixed knobs above and collect, per instance:
- `<inst>.steps.csv` (the table specified in
  [Output CSV Format](#output-csv-format)),
- `<inst>.steps.jsonl` (the same content in JSONL, slightly richer),
- `<inst>.tree.json` / `<inst>.tree.jsonl` (full tree with `y`, `success`).

### Stage 2 — data inspection
Sanity-check distributions of the step-level signals (`repeat_*`,
`failure_streak`, `obs_tag`, `command_type`) across successful vs failing
subtrees. Confirm that `y` propagates correctly and that token counters are
non-decreasing within a node.

### Stage 3 — predictor / monitor design
**Deferred** — see the TBD section above.

### Stage 4 — intervention evaluation
**Deferred** — depends on Stage 3.

---

## Short Summary

The most feasible **data-generation** method, which is what we commit to in
this document, is:

1. define a compact prefix from issue + recent 3-step window + 6 global
   counters,
2. estimate \(Y_k\) by a **recursive binary tree** of rollouts with
   \(B = 2\), **fixed equal temperatures** \(T_L = T_R = 0.3\), split every
   \(K = 10\) steps, capped by `step_budget = 60`, with leaf success from
   the SWE-bench harness (or `Submitted` proxy as fallback) and internal
   nodes set to the mean of their children,
3. dump every (node, step) into a per-instance CSV with the schema in
   [Output CSV Format](#output-csv-format).

What we do with those CSVs — predictor architecture, monitoring rule,
intervention — is **deliberately left open** in this version of the
document.
