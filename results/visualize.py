#!/usr/bin/env python3
"""Visualize MonitoringAgent trajectory signals.

Usage:
    python results/visualize.py results/e2e_47191683
    python results/visualize.py /scratch/.../results/lite_20260410_*

Generates:
    <result_dir>/analysis/
        signals_timeline.png   — 15 signals over steps
        phase_summary.png      — phase segmentation bar chart
        trajectory_report.html — interactive HTML report
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found, skipping PNG plots. Install with: pip install matplotlib")


def load_steps(result_dir: Path) -> list:
    """Load step logs from any instance in result_dir."""
    steps_files = list(result_dir.rglob("*.steps.jsonl"))
    # Prefer the one NOT inside .traj.steps.jsonl (the dedicated file)
    steps_files = [f for f in steps_files if ".traj." not in f.name] or steps_files
    if not steps_files:
        raise FileNotFoundError(f"No .steps.jsonl found in {result_dir}")
    all_steps = {}
    for sf in steps_files:
        instance_id = sf.name.replace(".steps.jsonl", "")
        with open(sf) as f:
            all_steps[instance_id] = [json.loads(line) for line in f if line.strip()]
    return all_steps


def load_traj(result_dir: Path) -> dict:
    """Load trajectory JSON files."""
    trajs = {}
    for tf in result_dir.rglob("*.traj.json"):
        instance_id = tf.name.replace(".traj.json", "")
        with open(tf) as f:
            trajs[instance_id] = json.load(f)
    return trajs


def classify_phase(step: dict, all_steps: list[dict]) -> str:
    """Heuristic phase classification."""
    idx = step["step_idx"]
    total = len(all_steps)
    # Find first edit of source file (not reproduce/test scripts)
    first_source_edit = total + 1
    for s in all_steps:
        if s["command_type"] == "edit" and s["target_file"] and \
           "reproduce" not in s["target_file"] and "test" not in s["target_file"] and \
           "debug" not in s["target_file"] and "fix_" not in s["target_file"] and \
           s["returncode"] == 0:
            first_source_edit = s["step_idx"]
            break

    if idx >= total - 1 and "patch" in step.get("command_raw", "").lower():
        return "Submit"
    elif idx >= first_source_edit:
        return "Fix & Verify"
    elif step["command_type"] == "read" and step["obs_tag"] == "success":
        return "Understand"
    else:
        return "Explore"


def plot_signals(instance_id: str, steps: list[dict], out_dir: Path):
    """Generate signal timeline plot."""
    if not HAS_MPL:
        return

    n = len(steps)
    x = [s["step_idx"] for s in steps]

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(5, 1, hspace=0.35)

    # 1. Context growth
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, [s["context_len_cum"] for s in steps], "b-o", markersize=4)
    ax1.set_ylabel("Context Length")
    ax1.set_title(f"Trajectory Signals — {instance_id} ({n} steps)")
    ax1.grid(True, alpha=0.3)

    # 2. Output length per step
    ax2 = fig.add_subplot(gs[1])
    colors = ["green" if s["returncode"] == 0 else "red" for s in steps]
    ax2.bar(x, [s["output_len"] for s in steps], color=colors, alpha=0.7)
    ax2.set_ylabel("Output Length")
    ax2.legend(["green=success, red=fail"], loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Failure streak
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(x, [s["failure_streak"] for s in steps], alpha=0.4, color="red")
    ax3.plot(x, [s["failure_streak"] for s in steps], "r-o", markersize=4)
    ax3.set_ylabel("Failure Streak")
    ax3.grid(True, alpha=0.3)

    # 4. Repetition scores
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(x, [s["repeat_cmd_score_recent"] for s in steps], "m-o", markersize=4, label="cmd_repeat")
    ax4.plot(x, [s["repeat_file_score_recent"] for s in steps], "c-s", markersize=4, label="file_repeat")
    ax4.set_ylabel("Repetition Score")
    ax4.legend(fontsize=8)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)

    # 5. Command type timeline
    ax5 = fig.add_subplot(gs[4])
    type_map = {"search": 0, "read": 1, "edit": 2, "run": 3, "install": 4, "test": 5, "navigate": 6}
    type_colors = {"search": "#4CAF50", "read": "#2196F3", "edit": "#FF9800",
                   "run": "#9C27B0", "install": "#795548", "test": "#F44336", "navigate": "#607D8B"}
    for s in steps:
        t = s["command_type"]
        y = type_map.get(t, 7)
        c = type_colors.get(t, "#999999")
        marker = "x" if s["returncode"] != 0 else "o"
        ax5.scatter(s["step_idx"], y, color=c, marker=marker, s=60, zorder=3)
    ax5.set_yticks(list(type_map.values()))
    ax5.set_yticklabels(list(type_map.keys()), fontsize=8)
    ax5.set_ylabel("Command Type")
    ax5.set_xlabel("Step")
    ax5.grid(True, alpha=0.3)

    plt.savefig(out_dir / "signals_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'signals_timeline.png'}")


def generate_html_report(instance_id: str, steps: list[dict], traj: dict, out_dir: Path):
    """Generate interactive HTML report."""
    # Extract messages for trajectory view
    messages = traj.get("messages", [])
    info = traj.get("info", {})

    # Build step detail rows
    step_rows = []
    for s in steps:
        cmd_preview = s["command_raw"].strip().replace("\n", "\\n")[:80]
        phase = classify_phase(s, steps)
        rc_class = "success" if s["returncode"] == 0 else "fail"
        step_rows.append(f"""
        <tr class="{rc_class}">
            <td>{s['step_idx']}</td>
            <td><span class="phase-{phase.lower().replace(' & ', '-').replace(' ', '-')}">{phase}</span></td>
            <td>{s['command_type']}</td>
            <td class="cmd" title="{s['command_raw'].strip()[:200]}">{cmd_preview}</td>
            <td>{s['returncode']}</td>
            <td>{s['obs_tag']}</td>
            <td>{s['context_len_cum']:,}</td>
            <td>{s['output_len']:,}</td>
            <td>{s['failure_streak']}</td>
            <td>{s['repeat_cmd_score_recent']:.2f}</td>
            <td>{s['repeat_file_score_recent']:.2f}</td>
        </tr>""")

    # Summary stats
    n_steps = len(steps)
    n_success = sum(1 for s in steps if s["returncode"] == 0)
    max_streak = max(s["failure_streak"] for s in steps)
    max_ctx = steps[-1]["context_len_cum"]
    submission = info.get("submission", "")
    exit_status = info.get("exit_status", "unknown")

    # Command type distribution
    type_counts = {}
    for s in steps:
        type_counts[s["command_type"]] = type_counts.get(s["command_type"], 0) + 1

    type_dist = " | ".join(f"{t}: {c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trajectory Report — {instance_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.summary .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
.metric {{ text-align: center; }}
.metric .value {{ font-size: 2em; font-weight: bold; color: #1a73e8; }}
.metric .label {{ font-size: 0.85em; color: #666; }}
.status-Submitted {{ color: #0a0; font-weight: bold; }}
.status-ContextWindowExceededError {{ color: #c00; font-weight: bold; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #1a73e8; color: white; padding: 10px 8px; text-align: left; font-size: 0.85em; }}
td {{ padding: 8px; border-bottom: 1px solid #eee; font-size: 0.85em; }}
tr.fail {{ background: #fff0f0; }}
tr.success {{ background: #f0fff0; }}
tr:hover {{ background: #e8f0fe !important; }}
.cmd {{ font-family: monospace; font-size: 0.8em; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.phase-explore {{ background: #e3f2fd; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
.phase-understand {{ background: #f3e5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
.phase-fix-verify {{ background: #fff3e0; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
.phase-submit {{ background: #e8f5e9; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
.patch {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.85em; white-space: pre; overflow-x: auto; }}
.patch .add {{ color: #6a9955; }}
.patch .del {{ color: #f44747; }}
img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>Trajectory Report: {instance_id}</h1>

<div class="summary">
<div class="grid">
    <div class="metric"><div class="value status-{exit_status}">{exit_status}</div><div class="label">Exit Status</div></div>
    <div class="metric"><div class="value">{n_steps}</div><div class="label">Total Steps</div></div>
    <div class="metric"><div class="value">{n_success}/{n_steps}</div><div class="label">Success Rate ({n_success/n_steps*100:.0f}%)</div></div>
    <div class="metric"><div class="value">{max_streak}</div><div class="label">Max Failure Streak</div></div>
    <div class="metric"><div class="value">{max_ctx:,}</div><div class="label">Final Context Length</div></div>
</div>
<p style="margin-top:15px;color:#666;font-size:0.9em">Command types: {type_dist}</p>
</div>

{"<img src='signals_timeline.png' alt='Signal Timeline'><br><br>" if HAS_MPL else ""}

<h2>Step-by-Step Log</h2>
<table>
<thead>
<tr>
    <th>#</th><th>Phase</th><th>Type</th><th>Command</th><th>RC</th>
    <th>Obs</th><th>Context</th><th>OutLen</th><th>Streak</th><th>RepCmd</th><th>RepFile</th>
</tr>
</thead>
<tbody>
{"".join(step_rows)}
</tbody>
</table>

<h2>Submitted Patch</h2>
<div class="patch">{format_patch_html(submission) if submission else "<em>No patch submitted</em>"}</div>

</body>
</html>"""

    html_path = out_dir / "trajectory_report.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Saved: {html_path}")


def format_patch_html(patch: str) -> str:
    """Syntax highlight a diff patch."""
    lines = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(f'<span class="add">{line}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            lines.append(f'<span class="del">{line}</span>')
        else:
            lines.append(line)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    if not result_dir.exists():
        print(f"ERROR: {result_dir} does not exist")
        sys.exit(1)

    all_steps = load_steps(result_dir)
    all_trajs = load_traj(result_dir)

    for instance_id, steps in all_steps.items():
        print(f"\nProcessing: {instance_id} ({len(steps)} steps)")
        out_dir = result_dir / "analysis"
        out_dir.mkdir(exist_ok=True)

        if HAS_MPL:
            plot_signals(instance_id, steps, out_dir)

        traj = all_trajs.get(instance_id, {"info": {}, "messages": []})
        generate_html_report(instance_id, steps, traj, out_dir)

    print(f"\nDone! Open the HTML report in your browser:")
    for f in (result_dir / "analysis").glob("*.html"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
