#!/usr/bin/env python3
"""Compare control and CUSUM rollback retry experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, help="Control output dir or summary.json")
    parser.add_argument("--intervention", required=True, help="Intervention output dir or summary.json")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    control = _load_summary(args.control)
    intervention = _load_summary(args.intervention)
    by_control = {row["instance_id"]: row for row in control.get("instances", [])}
    by_intervention = {row["instance_id"]: row for row in intervention.get("instances", [])}
    instance_ids = sorted(set(by_control) | set(by_intervention))

    rows = []
    header = f"{'instance_id':40s} {'control':>8s} {'interv':>8s} {'delta':>8s} {'ctrl_n':>6s} {'int_n':>6s}"
    print(header)
    print("-" * len(header))
    for instance_id in instance_ids:
        c = by_control.get(instance_id, {})
        i = by_intervention.get(instance_id, {})
        c_rate = c.get("success_rate")
        i_rate = i.get("success_rate")
        delta = i_rate - c_rate if c_rate is not None and i_rate is not None else None
        rows.append(
            {
                "instance_id": instance_id,
                "control_success_rate": c_rate,
                "intervention_success_rate": i_rate,
                "delta": delta,
                "control_n": c.get("n_runs", 0),
                "intervention_n": i.get("n_runs", 0),
            }
        )
        print(
            f"{instance_id:40s} "
            f"{(c_rate if c_rate is not None else float('nan')):8.3f} "
            f"{(i_rate if i_rate is not None else float('nan')):8.3f} "
            f"{(delta if delta is not None else float('nan')):+8.3f} "
            f"{int(c.get('n_runs', 0)):6d} {int(i.get('n_runs', 0)):6d}"
        )
    print("-" * len(header))
    overall_delta = intervention.get("success_rate", 0.0) - control.get("success_rate", 0.0)
    print(
        f"overall control={control.get('success_rate', 0.0):.3f} "
        f"intervention={intervention.get('success_rate', 0.0):.3f} "
        f"delta={overall_delta:+.3f}"
    )

    if args.out:
        payload = {
            "control": args.control,
            "intervention": args.intervention,
            "overall_delta": overall_delta,
            "rows": rows,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()