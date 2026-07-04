# Prefix Alarm Monitor

[`prefix_alarm_monitor.ipynb`](prefix_alarm_monitor.ipynb) is the current
source of truth for the first offline monitor trained from the tree-search
CSVs. [`prefix_alarm_monitor_lib.py`](prefix_alarm_monitor_lib.py) mirrors the
notebook implementation so the same pipeline can be run as
`tam.run_experiment(args)`.

The monitor consumes per-prefix features from `*.steps.csv`, learns a
calibrated probability of eventual success, and turns low predicted
solvability into leaf-level alarms.

## What The Notebook Does

The notebook:

1. reads all `*.steps.csv` files under `DATA_DIR`,
2. uses only mixed-root files (`0 < root_y < 1`) for training,
3. uses only pure-root files (`root_y = 0` or `root_y = 1`) for validation
   and test,
4. rebuilds each prefix along its root-to-node path,
5. truncates each prefix to a 64-step trailing window,
6. skips prefixes with `step_idx <= 8`,
7. samples remaining prefixes every 8 steps plus each node-final step,
8. learns a raw prefix score with weighted sibling Bradley-Terry comparisons
   plus a small row-level anchor,
9. fits a logistic calibrator from raw score to `p_success`,
10. tunes three leaf-level alarm rules on the validation split,
11. keeps `calibrated_leaf_low_fp` as the default final rule,
12. reports leaf-trajectory metrics on validation and test,
13. runs a no-confidence feature ablation, and
14. includes a CUSUM backtracking diagnostic that estimates how far before
    the actual CUSUM alarm the contributing risk window begins.

## Data Layout

Each row of `*.steps.csv` is one step of one node in the search tree. A leaf
trajectory is the sequence of sampled prefixes along the root-to-leaf path.
Splits are file-level, so prefixes from one SWE-bench instance never appear in
two splits.

- **Train**: mixed-root files only. The prefix label is the node's subtree
  success rate `target_y`, treated as a soft probability.
- **Validation / test**: pure-root files only. `pure0` means the root subtree
  has final success rate 0, so it is treated as a failing trajectory.
  `pure1` means the root subtree has final success rate 1, so it is treated
  as a successful trajectory.

The latest full-feature run uses:

| item | count |
| --- | ---: |
| total instances | 707 |
| mixed train files | 253 |
| pure validation files | 227 |
| pure test files | 227 |
| train prefixes | 75,036 |
| validation prefixes | 38,222 |
| test prefixes | 39,136 |
| sibling-pair examples | 7,565 |
| features | 148 |

## Features

`PrefixFeaturizer` computes a fixed-width feature vector over the trailing
64-step window. The current feature groups are:

| group | examples |
| --- | --- |
| window shape | `window_len`, `window_fill_ratio` |
| current numeric state | step index, depth, temperature, tokens, context length, return code, exception flag, output length, repeat scores, failure streak, confidence |
| current target-file shape | whether a target exists, path length/depth, `.py`, `test`, `__init__`, absolute-path-like and symbol-like flags |
| window numeric stats | last/mean/min/max/std/delta for tokens, wall time, context, return code, output length, repeat scores, failure streak, confidence |
| window binary stats | missing-confidence rate, nonzero-returncode rate, missing-target rate |
| command and observation mix | latest one-hot command/observation plus window fractions |
| diversity / churn | unique target ratio, same-target ratio, command switch ratio, observation switch ratio |

There are no embeddings and no model-token probabilities. The only LLM-derived
feature is the agent's self-reported `confidence`; missing confidence is kept
as an explicit signal rather than dropping rows.

## Model

The score model is linear on standardized features:

```text
s(x) = w^T x + b
```

Training uses Adam with `learning_rate = 0.03`, `epochs = 250`, and
`l2 = 1e-3`. The objective is:

```text
loss = pairwise_weight * sibling_Bradley_Terry_loss
     + row_weight      * soft_row_BCE
     + L2
```

with:

```text
pairwise_weight = 1.0
row_weight      = 0.2
min_pair_gap    = 0.05
```

For each mixed-file sibling pair, the pairwise term forms:

```text
diff   = x_left - x_right
label  = 1[left_y > right_y]
weight = |left_y - right_y|
```

The row anchor is a small binary cross-entropy against the soft subtree
success rate `target_y`. It fixes the otherwise arbitrary global shift in the
Bradley-Terry score.

After training, a logistic calibrator maps the raw score to a success
probability:

```text
p_success = sigmoid(calibration_scale * s(x) + calibration_bias)
```

Latest full-feature model:

```text
best_epoch = 13
validation AUC = 0.8811
calibration_scale = 0.8812
calibration_bias  = -0.3424
```

For the gated rule, the notebook also computes a score standard error from
the observed Fisher information:

```text
se(x) = sqrt(x^T covariance x)
lcb_score = s(x) - confidence_z * se(x)
confidence_z = 1.96
```

## Monitor Rules

All rules alarm when a prefix looks unlikely to succeed. Thresholds are chosen
on the validation split using leaf-trajectory metrics.

### `calibrated_leaf_low_fp`

Single-prefix calibrated probability rule:

```text
alarm_i = 1[p_success_i < theta_p]
```

`theta_p` is chosen to minimize successful-leaf false alarms subject to:

```text
pure0_leaf_alarm_recall >= target_leaf_recall
target_leaf_recall = 0.85
```

Latest full-feature threshold:

```text
theta_p = 0.3512
```

This remains the default because it gives the best balance between recall and
early warning.

### `cusum_leaf_low_fp`

Sequential CUSUM rule over each leaf path:

```text
fail_score_i = 1 - p_success_i
increment_i  = fail_score_i - k
C_i          = max(0, C_{i-1} + increment_i)
alarm_i      = 1[C_i >= h]
```

Both `k` (`cusum_drift`) and `h` (`cusum_threshold`) are tuned on validation
with the same leaf-level low-false-positive criterion.

Latest full-feature thresholds:

```text
cusum_drift     ~= 6.63e-12
cusum_threshold ~= 4.8645
```

In this run the drift is effectively zero, so the rule mostly accumulates
`1 - p_success` until the cumulative evidence crosses the threshold. This
cuts false alarms substantially, but alarms later.

### `gated_leaf_low_fp`

Confirmation rule:

```text
low_success_i = 1[p_success_i < theta_p]
low_lcb_i     = 1[lcb_score_i < theta_lcb]
sharp_drop_i  = 1[(s_i - s_parent_i) < -theta_drop]

alarm_i = low_success_i AND (low_lcb_i OR sharp_drop_i)
```

Latest full-feature thresholds:

```text
theta_lcb  = -2561.9781
theta_drop = 0.0641
theta_p    = 0.3512
```

The very low LCB threshold means the LCB branch is almost inactive in this
run; most extra filtering comes from the score-drop gate.

## Leaf-Level Metrics

Only leaf-trajectory monitoring is reported. For each pure leaf path, the
path is marked alarmed if any sampled prefix on that path triggers.

Metrics:

```text
pure0_leaf_alarm_recall
    fraction of failing leaf paths that get at least one alarm

pure1_leaf_false_alarm_rate
    fraction of successful leaf paths that get at least one alarm

pure0_leaf_alarm_median_step
    median first alarm step among alarmed failing paths

pure0_successful_warning_mean_early_pct
    mean over alarmed failing paths of:
        100 * (final_step - first_alarm_step) / final_step
```

`pure0_successful_warning_mean_early_pct` is conditional on successful true
alarms. Missed failing paths are not included in that average.

## Latest Effectiveness

Latest full-feature run:

```text
output: monitor_results/leaf_low_fp_monitor/
data:   707 instances, 253 mixed train files, 227 pure val files, 227 pure test files
```

Final default rule:

```text
calibrated_leaf_low_fp
p_success < 0.3512
```

| split | leaf TPR (`pure0`) | leaf FPR (`pure1`) | median true alarm step | mean early-warning |
| --- | ---: | ---: | ---: | ---: |
| val | 85.2% (7370 / 8654) | 18.8% (232 / 1232) | 40 | 38.8% |
| test | 84.3% (7554 / 8957) | 18.0% (209 / 1159) | 40 | 38.5% |

Rule comparison on the test set:

| rule | leaf TPR | leaf FPR | median true alarm step | mean early-warning |
| --- | ---: | ---: | ---: | ---: |
| `calibrated_leaf_low_fp` | 84.3% | 18.0% | 40 | 38.5% |
| `cusum_leaf_low_fp` | 82.1% | 10.4% | 49 | 22.3% |
| `gated_leaf_low_fp` | 84.1% | 17.3% | 40 | 38.0% |

Reading:

- `calibrated_leaf_low_fp` is the most balanced default: high recall and
  early warnings, but more false alarms.
- `cusum_leaf_low_fp` reduces successful-path false alarms from 18.0% to
  10.4%, a relative reduction of about 43%, but loses about 2.2 points of
  failing-path recall and alarms around 9 steps later.
- `gated_leaf_low_fp` is a light filter over the calibrated rule. It slightly
  lowers false alarms while keeping almost the same recall and timing.

## CUSUM Backtracking Diagnostic

The notebook includes a `CUSUM Alarm Backtracking` cell. For each CUSUM-alarmed
leaf path, it traces backward from the first alarm to the shortest recent
contribution window whose increments are enough to cross the CUSUM threshold.

For an alarm at sampled position `tau`:

```text
increment_t = fail_score_t - cusum_drift
```

The backtracked `risk_start` is the latest start position such that:

```text
sum_{t=risk_start}^{tau} increment_t >= cusum_threshold
```

The diagnostic then reports:

```text
rollback_steps = alarm_step - risk_start_step
```

Latest all-path output from the notebook:

| split | alarmed leaf paths | mean rollback steps | median rollback steps | mean rollback sampled prefixes | median rollback sampled prefixes | median risk start step | median alarm step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 7572 | 34.2 | 32 | 8.51 | 8 | 16 | 48 |
| test | 7478 | 34.6 | 32 | 8.58 | 8 | 16 | 49 |

This is a post-hoc explanation of where the accumulated evidence came from.
It does not mean the monitor could have fired at the risk-start step online.
It means the alarm at roughly step 48-49 is mostly explained by the previous
about 9 sampled prefixes, or about 32-35 real steps.

## No-Confidence Ablation

The no-confidence run drops all features whose names contain `confidence`.
It is saved under:

```text
monitor_results/no_confidence_leaf_low_fp_monitor/
```

Test comparison:

| setup | calibrated FPR | CUSUM FPR | gated FPR |
| --- | ---: | ---: | ---: |
| full features | 18.0% | 10.4% | 17.3% |
| no confidence | 20.2% | 10.4% | 19.2% |

The confidence-derived features are not causing the current false alarms; if
anything, dropping them makes the calibrated and gated rules less selective.

## Feature Weights

Top standardized weights by absolute value in the latest full-feature run:

| feature | weight |
| --- | ---: |
| `win__cmd_frac__diff` | 0.314 |
| `win__cmd_frac__test` | 0.293 |
| `win__target_missing__mean` | -0.283 |
| `win__repeat_cmd_score_recent__mean` | -0.276 |
| `win__cmd_frac__other` | -0.255 |
| `win__command_switch_ratio` | -0.247 |
| `cur__prompt_tokens_cum` | -0.244 |
| `win__cmd_frac__read` | -0.234 |
| `cur__cmd__edit` | 0.232 |
| `win__output_len__mean` | 0.222 |
| `win__confidence__std` | 0.217 |
| `win__confidence__mean` | 0.212 |

No single feature dominates. The head is spread across command mix, target
behavior, repetition, context/token scale, output length, and confidence.

## Run

Open [`prefix_alarm_monitor.ipynb`](prefix_alarm_monitor.ipynb), set
`DATA_DIR` and `OUTPUT_DIR` near the top if needed, then run the notebook.

The same implementation is also exposed in
[`prefix_alarm_monitor_lib.py`](prefix_alarm_monitor_lib.py):

```python
summary = tam.run_experiment(args)
```

Current defaults:

| setting | value |
| --- | ---: |
| `window_size` | 64 |
| `stride` | 8 |
| `min_step` | 9 |
| `row_weight` | 0.2 |
| `pairwise_weight` | 1.0 |
| `epochs` | 250 |
| `learning_rate` | 0.03 |
| `l2` | 1e-3 |
| `min_pair_gap` | 0.05 |
| `calibration_epochs` | 400 |
| `calibration_learning_rate` | 0.05 |
| `calibration_l2` | 1e-3 |
| `confidence_z` | 1.96 |
| `target_leaf_recall` | 0.85 |
| `rule_threshold_max_candidates` | 48 |
| `pure_val_ratio` | 0.5 |
| `seed` | 7 |

## Outputs

Each run writes to `OUTPUT_DIR`:

| file | contents |
| --- | --- |
| `summary.json` | config, counts, thresholds, model info, leaf-rule metrics, final rule |
| `splits.json` | train / validation / test file split |
| `training_history.json` | epoch-by-epoch row BCE, pairwise BCE, and validation AUC |
| `scaler.json` | training-split feature mean and scale |
| `calibrator.json` | fitted logistic calibration parameters |
| `val_predictions.csv`, `test_predictions.csv` | per-prefix success probability, raw score, LCB, parent score, score drop, and final-rule flags |
| `feature_weights.csv` | learned standardized weight for every feature |
| `top_weights.json` | top 20 features by absolute standardized weight |

The CUSUM backtracking table is displayed by the notebook cell and is not
currently written into `summary.json`.

## How To Extend

If more `lite` or `verified` CSVs are added, point `DATA_DIR` at the new
directory and rerun. Keep the split semantics unless you intentionally want a
different experiment:

- train on mixed-root files,
- tune thresholds on pure validation files,
- report once on pure test files.

For false-alarm reduction, the current evidence suggests rule design matters
more than simple feature deletion. CUSUM gives the clearest reduction in
successful-path false alarms, while the no-confidence ablation does not help.

For feature-set experiments, edit `CURRENT_NUMERIC_KEYS`,
`WINDOW_STATS_KEYS`, `TARGET_FILE_FEATURE_NAMES`, `COMMAND_TYPES`, or
`OBS_TAGS` in the notebook/library. Re-run the whole pipeline and compare
leaf-level FPR at the same target recall; prefix AUC alone is not enough.
