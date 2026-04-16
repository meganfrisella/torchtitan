#!/usr/bin/env python3
"""Run a configurable Torchtitan Qwen EC2 sweep.

Default sweep:
  SCHEDULE in {1f1b, interleaved1f1b, zerobubble, dualpipe}
  PP in {4, 8}
  DP in {1, 2, 4}

Commands look like:
  ./scripts/run-qwen-ec2.sh \
    --nnode DP --ngpu PP \
    --module qwen3 --config qwen3_9b_{SCHEDULE}_pp{PP}_dp{DP}

Each experiment writes stdout/stderr to a separate local log file.
Failures are recorded and reported at the end without stopping the sweep.
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path


DEFAULT_PP_VALUES = (4, 8)
DEFAULT_DP_VALUES = (1, 2, 4)
DEFAULT_SCHEDULES = ("1f1b", "interleaved1f1b", "zerobubble", "dualpipe")


def _build_command(
    script_path: Path,
    *,
    schedule: str,
    pp: int,
    dp: int,
    module: str,
    extra_args: list[str],
) -> list[str]:
    config = f"qwen3_9b_pp{pp}_dp{dp}_{schedule}"
    log_rank = ",".join(str(i) for i in range(pp))
    command = [
        str(script_path),
        "--nnode", str(dp),
        "--ngpu", str(pp),
        "--module", module,
        "--config", config,
        "--log-rank", log_rank,
    ]
    command.extend(extra_args)
    return command


def _default_log_dir(script_path: Path) -> Path:
    return script_path.parent.parent / "out" / "ec2_sweeps"


def _log_path(log_dir: Path, *, schedule: str, pp: int, dp: int) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return log_dir / f"qwen3_9b_pp{pp}_dp{dp}__{schedule}_{timestamp}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Torchtitan scripts/run-qwen-ec2.sh over schedule, PP, and DP values."
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=list(DEFAULT_SCHEDULES),
        choices=list(DEFAULT_SCHEDULES),
        help="Schedules to sweep. Default: 1f1b interleaved1f1b zerobubble dualpipe",
    )
    parser.add_argument(
        "--pp-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_PP_VALUES),
        help="PP values to sweep. Default: 4 8",
    )
    parser.add_argument(
        "--dp-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_DP_VALUES),
        help="DP values to sweep. Default: 1 2 4",
    )
    parser.add_argument(
        "--module",
        default="qwen3",
        help="Module passed to run-qwen-ec2.sh. Default: qwen3",
    )
    parser.add_argument(
        "--script",
        default="scripts/run-qwen-ec2.sh",
        help="Path to the underlying Torchtitan runner script",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-experiment stdout/stderr logs. Default: scripts/../out/ec2_sweeps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to run-qwen-ec2.sh. Prefix with '--'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        print(f"Runner script not found: {script_path}", file=sys.stderr)
        return 1

    log_dir = Path(args.log_dir).resolve() if args.log_dir else _default_log_dir(script_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    combinations = list(itertools.product(args.schedules, args.pp_values, args.dp_values))
    total = len(combinations)
    failures: list[dict[str, object]] = []

    for index, (schedule, pp, dp) in enumerate(combinations, start=1):
        command = _build_command(
            script_path,
            schedule=schedule,
            pp=pp,
            dp=dp,
            module=args.module,
            extra_args=extra_args,
        )
        log_path = _log_path(log_dir, schedule=schedule, pp=pp, dp=dp)

        print(f"[{index}/{total}] schedule={schedule} pp={pp} dp={dp}")
        print("  " + " ".join(shlex.quote(part) for part in command))
        print(f"  log={log_path}")

        if args.dry_run:
            continue

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode == 0:
                continue
            failures.append(
                {
                    "index": index,
                    "schedule": schedule,
                    "pp": pp,
                    "dp": dp,
                    "returncode": result.returncode,
                    "command": command,
                    "log_path": str(log_path),
                }
            )
            print(f"  command failed with exit code {result.returncode}, continuing", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nSweep interrupted by user.", file=sys.stderr)
            failures.append(
                {
                    "index": index,
                    "schedule": schedule,
                    "pp": pp,
                    "dp": dp,
                    "returncode": "interrupted",
                    "command": command,
                    "log_path": str(log_path),
                }
            )
            break
        except Exception as exc:
            failures.append(
                {
                    "index": index,
                    "schedule": schedule,
                    "pp": pp,
                    "dp": dp,
                    "returncode": "exception",
                    "command": command,
                    "log_path": str(log_path),
                    "error": repr(exc),
                }
            )
            print(f"  command raised {type(exc).__name__}: {exc}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    if failures:
        print("\nSweep completed with failures:", file=sys.stderr)
        for failure in failures:
            print(
                f"  [{failure['index']}/{total}] schedule={failure['schedule']} pp={failure['pp']} dp={failure['dp']} "
                f"status={failure['returncode']}",
                file=sys.stderr,
            )
            print(
                "    " + " ".join(shlex.quote(part) for part in failure["command"]),
                file=sys.stderr,
            )
            print(f"    log={failure['log_path']}", file=sys.stderr)
            if "error" in failure:
                print(f"    error={failure['error']}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
