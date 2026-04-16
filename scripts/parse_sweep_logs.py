#!/usr/bin/env python3
"""Parse sweep log files into a CSV of iter time and peak memory stats.

Usage:
    python scripts/parse_sweep_logs.py [LOG_DIR] [-o OUTPUT_CSV]

Reads all *.log files in LOG_DIR (default: out/ec2_sweeps) whose names match
the sweep filename pattern:
    qwen3_9b_pp<PP>_dp<DP>__<schedule>_<timestamp>.log

Writes a CSV with columns:
    schedule, pp, dp, iter_time_mean, iter_time_stddev, peak_memory_gb_by_rank

peak_memory_gb_by_rank is the per-rank peak across all logged steps, formatted
as slash-separated values in rank order, e.g. "13.63/14.21/13.55/13.80".
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Strip ANSI escape codes
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# qwen3_9b_pp4_dp2__interleaved1f1b_20260416_024946.log
_FILENAME_RE = re.compile(r"qwen3_9b_pp(\d+)_dp(\d+)__(.+?)_\d{8}_\d{6}\.log$")

# "Final 5 iter times — avg: 6.07874 s, std: 0.04286 s"
_ITER_TIME_RE = re.compile(
    r"Final \d+ iter times.*?avg:\s*([\d.]+)\s*s,\s*std:\s*([\d.]+)\s*s"
)

# "[rank3]:... memory: 13.63GiB(...)"
_MEMORY_RE = re.compile(r"\[rank(\d+)\].*?memory:\s*([\d.]+)GiB")


def _parse_log(path: Path) -> tuple[float | None, float | None, dict[int, float]]:
    iter_avg = iter_std = None
    peak: dict[int, float] = {}

    with path.open(errors="replace") as f:
        for line in f:
            line = _ANSI_RE.sub("", line)

            m = _ITER_TIME_RE.search(line)
            if m:
                iter_avg, iter_std = float(m.group(1)), float(m.group(2))

            m = _MEMORY_RE.search(line)
            if m:
                rank, mem_gb = int(m.group(1)), float(m.group(2))
                peak[rank] = mem_gb

    return iter_avg, iter_std, peak


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse sweep logs into a CSV.")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="out/ec2_sweeps",
        help="Directory containing log files (default: out/ec2_sweeps)",
    )
    parser.add_argument(
        "-o", "--output",
        default="-",
        help="Output CSV path (default: stdout)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"Error: directory not found: {log_dir}", file=sys.stderr)
        return 1

    rows = []
    skipped = 0
    for log_path in sorted(log_dir.glob("*.log")):
        m = _FILENAME_RE.match(log_path.name)
        if not m:
            skipped += 1
            continue

        pp, dp, schedule = int(m.group(1)), int(m.group(2)), m.group(3)
        iter_avg, iter_std, peak = _parse_log(log_path)

        last_mem_str = "/".join(f"{peak[r]:.2f}" for r in sorted(peak)) if peak else ""

        rows.append({
            "schedule": schedule,
            "pp": pp,
            "dp": dp,
            "iter_time_mean": iter_avg,
            "iter_time_stddev": iter_std,
            "peak_memory_gb_by_rank": last_mem_str,
        })

    rows.sort(key=lambda r: (r["schedule"], r["pp"], r["dp"]))

    if skipped:
        print(f"Skipped {skipped} file(s) with unrecognised names.", file=sys.stderr)

    fieldnames = ["schedule", "pp", "dp", "iter_time_mean", "iter_time_stddev", "peak_memory_gb_by_rank"]

    if args.output == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        out_path = Path(args.output)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
