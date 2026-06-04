#!/usr/bin/env python3
"""Serially prefetch SWE-bench SIF images into a local cache.

Why: Docker Hub limits anonymous pulls to 100/6h and authenticated free
accounts to 200/6h. A parallel batch with 8 workers blows past the cap
within an hour, killing most instances mid-run with TOOMANYREQUESTS.
Pre-pulling once, serially, with a small sleep between pulls is enough
to fit comfortably inside the cap and lets the real batch run hit the
local cache (zero docker.io traffic).

Usage
-----
    source ~/.venvs/mon_agent_env/bin/activate
    module load singularity
    # optional: export DOCKERHUB_USER=... DOCKERHUB_TOKEN=...
    python scripts/prefetch_sif.py \
        --subset lite --split test --slice 0:300 \
        --sif-cache /scratch/alkontar_root/alkontar0/kevinwyk/swebench_sif \
        --sleep 10

Resumes automatically: any .sif already present and non-empty is skipped.
Failed pulls are appended to <sif-cache>/_prefetch_failures.txt for retry.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe(name: str) -> str:
    return _SAFE.sub("_", name).strip("_") or "x"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", default="0:300", dest="slice_spec")
    ap.add_argument("--sif-cache", required=True)
    ap.add_argument("--sleep", type=float, default=10.0,
                    help="Seconds to sleep between pulls (rate-limit cushion).")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Per-pull timeout (s).")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    from datasets import load_dataset
    from minisweagent.run.benchmarks.swebench import (
        DATASET_MAPPING,
        get_swebench_docker_image_name,
    )

    sif_cache = Path(args.sif_cache).expanduser()
    sif_cache.mkdir(parents=True, exist_ok=True)
    fail_log = sif_cache / "_prefetch_failures.txt"

    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    print(f"[prefetch] loading {dataset_path}/{args.split}")
    ds = list(load_dataset(dataset_path, split=args.split))
    if args.slice_spec:
        vals = [int(x) if x else None for x in args.slice_spec.split(":")]
        ds = ds[slice(*vals)]
    print(f"[prefetch] {len(ds)} instances in slice {args.slice_spec}")

    sing = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    n_skip = n_ok = n_fail = 0
    for i, inst in enumerate(ds, 1):
        iid = inst["instance_id"]
        sif_path = sif_cache / f"{safe(iid)}.sif"
        if sif_path.exists() and sif_path.stat().st_size > 0:
            n_skip += 1
            print(f"[{i:>3}/{len(ds)}] SKIP cached  {iid}")
            continue
        image = get_swebench_docker_image_name(inst)
        tmp = sif_path.with_suffix(".sif.part")
        if tmp.exists():
            tmp.unlink()
        ok = False
        for attempt in range(1, args.retries + 1):
            t0 = time.time()
            print(f"[{i:>3}/{len(ds)}] PULL ({attempt}/{args.retries}) {iid}")
            proc = subprocess.run(
                [sing, "pull", "--force", str(tmp), f"docker://{image}"],
                capture_output=True, text=True, timeout=args.timeout,
            )
            if proc.returncode == 0:
                tmp.replace(sif_path)
                ok = True
                print(f"           OK {time.time()-t0:.0f}s  -> {sif_path.name}")
                break
            err = (proc.stderr or proc.stdout or "")[-400:].strip()
            print(f"           FAIL rc={proc.returncode}: {err[:200]}")
            # Long cool-off on rate limit; short retry otherwise
            if "TOOMANYREQUESTS" in err or "rate limit" in err.lower():
                cool = 600  # 10 min
            else:
                cool = 30 * attempt
            if attempt < args.retries:
                print(f"           backing off {cool}s")
                time.sleep(cool)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            with fail_log.open("a") as fh:
                fh.write(f"{iid}\t{image}\n")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        if args.sleep > 0 and i < len(ds):
            time.sleep(args.sleep)

    print()
    print(f"[prefetch] done: cached={n_skip} new_ok={n_ok} failed={n_fail}")
    if n_fail:
        print(f"[prefetch] failures listed in {fail_log}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
