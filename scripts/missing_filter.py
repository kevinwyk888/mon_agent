#!/usr/bin/env python3
"""Print a `--filter` regex that selects SWE-bench instances missing from a
results CSV directory.

Usage
-----
    python scripts/missing_filter.py \
        --subset lite --split test --slice 0:300 \
        --csv-dir results/tree_run_51085914

Prints e.g.  ^(django__django-14667|django__django-14672|...)$
which can be passed as `--filter "$(...)"` to mon-agent-runner.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", default="0:300", dest="slice_spec")
    ap.add_argument("--csv-dir", required=True,
                    help="Directory containing <instance_id>.steps.csv files.")
    ap.add_argument("--print-list", action="store_true",
                    help="Print one instance_id per line instead of a regex.")
    args = ap.parse_args()

    from datasets import load_dataset
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING

    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    ds = list(load_dataset(dataset_path, split=args.split))
    if args.slice_spec:
        vals = [int(x) if x else None for x in args.slice_spec.split(":")]
        ds = ds[slice(*vals)]

    csv_dir = Path(args.csv_dir).expanduser()
    have = {p.name[: -len(".steps.csv")]
            for p in csv_dir.glob("*.steps.csv")}

    missing = [inst["instance_id"] for inst in ds
               if inst["instance_id"] not in have]

    print(f"[missing_filter] total={len(ds)} done={len(ds)-len(missing)} "
          f"missing={len(missing)}", file=sys.stderr)
    if not missing:
        print("^$")  # match nothing
        return 0
    if args.print_list:
        for m in missing:
            print(m)
    else:
        # Escape regex metacharacters in instance ids (have __ and -)
        import re as _re
        parts = "|".join(_re.escape(m) for m in missing)
        print(f"^({parts})$")
    return 0


if __name__ == "__main__":
    sys.exit(main())
