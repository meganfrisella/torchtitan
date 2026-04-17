#!/usr/bin/env python3
"""Run a configurable Torchtitan Qwen EC2 sweep.

This runner now uses a small set of generic Qwen3 9B configs and passes
PP/DP/EP/schedule/sequence-length/microbatch settings as runtime overrides.

Config selection:
  - all zero levels use qwen3_9b
  - ZeRO behavior is injected entirely via runtime overrides

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
DEFAULT_PARALLEL_VALUES = (1, 2, 4)
DEFAULT_SCHEDULES = ("1f1b", "interleaved1f1b", "zerobubble", "dualpipe")
DEFAULT_ZERO_LEVELS = ("none", "zero2", "zero3")
DEFAULT_BASE_CONFIG = "qwen3_9b"
RUNTIME_SCHEDULES = {
    "1f1b": "1F1B",
    "interleaved1f1b": "Interleaved1F1B",
    "zerobubble": "InterleavedZeroBubble",
    "dualpipe": "DualPipeV",
}


def _build_config_name() -> str:
    return DEFAULT_BASE_CONFIG



def _is_valid_combination(*, schedule: str, parallel_degree: int, mode: str, zero_level: str) -> bool:
    if mode == "ep":
        return zero_level == "none"
    return (zero_level == "none") or (
        zero_level != "none" and schedule == "1f1b" and parallel_degree in (2, 4)
    )



def _local_batch_size(pp: int, mb_size: int) -> int:
    return pp * 2 * mb_size



def _global_batch_size(pp: int, parallel_degree: int, mode: str, mb_size: int) -> int:
    return _local_batch_size(pp, mb_size) * parallel_degree



def _runtime_dp_settings(*, mode: str, parallel_degree: int, zero_level: str) -> tuple[int, int, str | None, int]:
    if mode == "ep":
        return 1, -1 if parallel_degree > 1 else 1, None, parallel_degree
    if zero_level == "none":
        return parallel_degree, 1, None, parallel_degree
    if zero_level == "zero2":
        return 1, parallel_degree, "never", parallel_degree
    if zero_level == "zero3":
        return 1, parallel_degree, "always", parallel_degree
    raise ValueError(f"Unsupported zero level: {zero_level}")



def _build_command(
    script_path: Path,
    *,
    schedule: str,
    pp: int,
    parallel_degree: int,
    mode: str,
    zero_level: str,
    module: str,
    seq_len: int,
    mb_size: int,
    extra_args: list[str],
) -> list[str]:
    config = _build_config_name()
    dp_replicate, dp_shard, reshard_after_forward, nnode = _runtime_dp_settings(
        mode=mode,
        parallel_degree=parallel_degree,
        zero_level=zero_level,
    )
    ep_degree = parallel_degree if mode == "ep" else 1
    log_rank = ",".join(str(i) for i in range(pp))
    train_args = [
        "--parallelism.pipeline_parallel_degree",
        str(pp),
        "--parallelism.expert_parallel_degree",
        str(ep_degree),
        "--parallelism.pipeline_parallel_schedule",
        RUNTIME_SCHEDULES[schedule],
        "--parallelism.pipeline_parallel_microbatch_size",
        str(mb_size),
        "--parallelism.data_parallel_replicate_degree",
        str(dp_replicate),
        "--parallelism.data_parallel_shard_degree",
        str(dp_shard),
        "--training.seq_len",
        str(seq_len),
        "--training.local_batch_size",
        str(_local_batch_size(pp, mb_size)),
        "--training.global_batch_size",
        str(_global_batch_size(pp, parallel_degree, mode, mb_size)),
    ]
    if reshard_after_forward is not None:
        train_args.extend([
            "--parallelism.fsdp_reshard_after_forward",
            reshard_after_forward,
        ])
    command = [
        str(script_path),
        "--nnode",
        str(nnode),
        "--ngpu",
        str(pp),
        "--module",
        module,
        "--config",
        config,
        "--log-rank",
        log_rank,
        "--",
        *train_args,
    ]
    command.extend(extra_args)
    return command



def _default_log_dir(script_path: Path) -> Path:
    return script_path.parent.parent / "out" / "ec2_sweeps"



def _log_path(
    log_dir: Path,
    *,
    schedule: str,
    pp: int,
    parallel_degree: int,
    mode: str,
    zero_level: str,
) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    variant = mode if mode == "ep" else zero_level
    return log_dir / f"qwen3_9b_pp{pp}_{mode}{parallel_degree}__{variant}__{schedule}_{timestamp}.log"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Torchtitan scripts/run-qwen-ec2.sh over EP or DP/ZeRO runtime overrides."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dp",
        nargs="+",
        type=int,
        default=None,
        help="DP values to sweep for DP/ZeRO runs. Cannot be used with --ep.",
    )
    group.add_argument(
        "--ep",
        nargs="+",
        type=int,
        default=None,
        help="EP values to sweep for EP runs. Cannot be used with --dp.",
    )
    parser.add_argument(
        "--zero-levels",
        nargs="+",
        default=list(DEFAULT_ZERO_LEVELS),
        choices=list(DEFAULT_ZERO_LEVELS),
        help="ZeRO levels to sweep. For --ep, only 'none' is valid. Default: none zero2 zero3",
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
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length runtime override. Default: 512")
    parser.add_argument("--mb-size", type=int, default=8, help="Pipeline microbatch size runtime override. Default: 8")
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
        help="Extra arguments forwarded to torchtitan.train. Prefix with '--'.",
    )
    args = parser.parse_args()

    if args.dp is None and args.ep is None:
        args.ep = list(DEFAULT_PARALLEL_VALUES)

    return args



def main() -> int:
    args = parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        print(f"Runner script not found: {script_path}", file=sys.stderr)
        return 1

    mode = "dp" if args.dp is not None else "ep"
    parallel_values = args.dp if args.dp is not None else args.ep

    log_dir = Path(args.log_dir).resolve() if args.log_dir else _default_log_dir(script_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    raw_combinations = itertools.product(
        args.zero_levels,
        args.schedules,
        args.pp_values,
        parallel_values,
    )
    combinations = [
        (zero_level, schedule, pp, parallel_degree)
        for zero_level, schedule, pp, parallel_degree in raw_combinations
        if _is_valid_combination(
            schedule=schedule,
            parallel_degree=parallel_degree,
            mode=mode,
            zero_level=zero_level,
        )
    ]
    total = len(combinations)
    if total == 0:
        print("No valid sweep combinations generated.", file=sys.stderr)
        if mode == "ep":
            print(
                "EP mode only supports --zero-levels none. Example: --ep 1 2 4 --zero-levels none",
                file=sys.stderr,
            )
        else:
            print(
                "DP mode supports plain DP with --zero-levels none, and ZeRO configs with --zero-levels zero2 zero3 (ZeRO requires --schedules 1f1b and --dp 2 or 4).",
                file=sys.stderr,
            )
            print(
                "Example plain DP: --dp 1 2 4 --schedules 1f1b --zero-levels none",
                file=sys.stderr,
            )
        return 1
    failures: list[dict[str, object]] = []

    for index, (zero_level, schedule, pp, parallel_degree) in enumerate(combinations, start=1):
        command = _build_command(
            script_path,
            schedule=schedule,
            pp=pp,
            parallel_degree=parallel_degree,
            mode=mode,
            zero_level=zero_level,
            module=args.module,
            seq_len=args.seq_len,
            mb_size=args.mb_size,
            extra_args=extra_args,
        )
        config = _build_config_name()
        log_path = _log_path(
            log_dir,
            schedule=schedule,
            pp=pp,
            parallel_degree=parallel_degree,
            mode=mode,
            zero_level=zero_level,
        )

        print(
            f"[{index}/{total}] mode={mode} zero_level={zero_level} schedule={schedule} "
            f"pp={pp} parallel_degree={parallel_degree} config={config} mb={args.mb_size} seq_len={args.seq_len}"
        )
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
                    "mode": mode,
                    "zero_level": zero_level,
                    "schedule": schedule,
                    "pp": pp,
                    "parallel_degree": parallel_degree,
                    "config": config,
                    "returncode": result.returncode,
                    "command": command,
                    "log_path": str(log_path),
                }
            )
            print(
                f"  command failed with exit code {result.returncode}, continuing",
                file=sys.stderr,
            )
        except KeyboardInterrupt:
            print("\nSweep interrupted by user.", file=sys.stderr)
            failures.append(
                {
                    "index": index,
                    "mode": mode,
                    "zero_level": zero_level,
                    "schedule": schedule,
                    "pp": pp,
                    "parallel_degree": parallel_degree,
                    "config": config,
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
                    "mode": mode,
                    "zero_level": zero_level,
                    "schedule": schedule,
                    "pp": pp,
                    "parallel_degree": parallel_degree,
                    "config": config,
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
                f"  [{failure['index']}/{total}] mode={failure['mode']} "
                f"zero_level={failure['zero_level']} schedule={failure['schedule']} "
                f"pp={failure['pp']} parallel_degree={failure['parallel_degree']} "
                f"config={failure['config']} status={failure['returncode']}",
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
