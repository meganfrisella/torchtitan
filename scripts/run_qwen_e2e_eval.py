#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/m-coriander/coriander/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Final \d+ iter times.*?avg:\s*([\d.]+)\s*s,\s*std:\s*([\d.]+)\s*s")
MEM_RE = re.compile(r"\[rank(\d+)\].*?memory:\s*([\d.]+)GiB")
FINAL_BY_RANK_RE = re.compile(
    r"\[rank(\d+)\].*?Final \d+ iter times.*?avg:\s*([\d.]+)\s*s,\s*std:\s*([\d.]+)\s*s"
)
OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "oom",
    "std::bad_alloc",
)
DEFAULT_SCHEDULE_SWEEP = ("1f1b", "interleaved1f1b", "zerobubble")
DEFAULT_BASE_CONFIG = "qwen3_9b"


@dataclass
class Experiment:
    sweep: str
    config: str
    pp: int
    dp: int
    ep: int
    zero_level: str
    schedule: str
    mb_size: int
    seq_len: int


@dataclass
class Result:
    sweep: str
    config: str
    pp: int
    dp: int
    ep: int
    zero_level: str
    schedule: str
    mb_size: int
    seq_len: int
    local_batch_size: int
    global_batch_size: int
    nnode: int
    ngpu: int
    log_path: str
    returncode: int
    status: str
    iter_time_mean: float | None
    iter_time_stddev: float | None
    peak_memory_gb_by_rank: str
    failure_reason: str


def normalize_schedule(name: str) -> str:
    mapping = {
        "1f1b": "1f1b",
        "interleaved1f1b": "interleaved1f1b",
        "interleaved-1f1b": "interleaved1f1b",
        "zerobubble": "zerobubble",
        "zero-bubble": "zerobubble",
        "interleavedzerobubble": "zerobubble",
        "interleaved-zero-bubble": "zerobubble",
        "dualpipe": "dualpipe",
        "dualpipev": "dualpipe",
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported schedule: {name}")
    return mapping[key]


RUNTIME_SCHEDULES = {
    "1f1b": "1F1B",
    "interleaved1f1b": "Interleaved1F1B",
    "zerobubble": "InterleavedZeroBubble",
    "dualpipe": "DualPipeV",
}



def build_experiments(
    *,
    schedule_sweep: Iterable[str],
    base_config: str,
    ep_degree: int,
    seq_len: int,
    enabled_sweeps: set[str],
) -> list[Experiment]:
    experiments: list[Experiment] = []

    if "scalability" in enabled_sweeps:
        for pp in (4, 8):
            for dp in (1, 2):
                experiments.append(
                Experiment(
                    sweep="scalability",
                    config=base_config,
                    pp=pp,
                    dp=dp,
                    ep=ep_degree,
                    zero_level="none",
                    schedule="1f1b",
                    mb_size=4,
                    seq_len=seq_len,
                )
            )

    if "zero" in enabled_sweeps:
        for zero_level in ("none", "zero2", "zero3"):
            for mb_size in (4, 8, 16):
                experiments.append(
                Experiment(
                    sweep="zero",
                    config=base_config,
                    pp=8,
                    dp=2,
                    ep=ep_degree,
                    zero_level=zero_level,
                    schedule="1f1b",
                    mb_size=mb_size,
                    seq_len=seq_len,
                )
            )

    if "schedule" in enabled_sweeps:
        for schedule in schedule_sweep:
            experiments.append(
            Experiment(
                sweep="schedule",
                config=base_config,
                pp=8,
                dp=2,
                ep=ep_degree,
                zero_level="none",
                schedule=normalize_schedule(schedule),
                mb_size=4,
                seq_len=seq_len,
            )
        )

    return experiments



def local_batch_size(pp: int, mb_size: int) -> int:
    return pp * 2 * mb_size



def global_batch_size(pp: int, dp: int, mb_size: int) -> int:
    return local_batch_size(pp, mb_size) * dp



def experiment_label(exp: Experiment) -> str:
    if exp.sweep == "scalability":
        return f"pp={exp.pp}, dp={exp.dp}"
    if exp.sweep == "zero":
        return f"zero={exp.zero_level}, mb={exp.mb_size}"
    return exp.schedule



def log_filename(index: int, exp: Experiment) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        f"{index:02d}_{exp.sweep}__pp{exp.pp}_dp{exp.dp}_ep{exp.ep}__{exp.zero_level}__{exp.schedule}__mb{exp.mb_size}__sl{exp.seq_len}_{timestamp}.log"
    )



def parse_log(log_path: Path) -> tuple[float | None, float | None, str, bool, str]:
    raw = log_path.read_bytes().replace(b"\0", b"")
    text = raw.decode("utf-8", errors="replace")

    peak: dict[int, float] = {}
    iter_by_rank: dict[int, tuple[float, float]] = {}
    last_iter: tuple[float, float] | None = None
    lower = text.lower()

    for line in text.splitlines():
        line = ANSI_RE.sub("", line)
        m = FINAL_BY_RANK_RE.search(line)
        if m:
            iter_by_rank[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            last_iter = (float(m.group(2)), float(m.group(3)))
            continue
        m = ITER_RE.search(line)
        if m:
            last_iter = (float(m.group(1)), float(m.group(2)))
        m = MEM_RE.search(line)
        if m:
            rank, mem_gb = int(m.group(1)), float(m.group(2))
            peak[rank] = max(peak.get(rank, 0.0), mem_gb)

    if 0 in iter_by_rank:
        iter_avg, iter_std = iter_by_rank[0]
    elif last_iter is not None:
        iter_avg, iter_std = last_iter
    else:
        iter_avg = iter_std = None

    peak_str = "/".join(f"{peak[r]:.2f}" for r in sorted(peak)) if peak else ""
    is_oom = any(pattern in lower for pattern in OOM_PATTERNS)
    reason = "oom" if is_oom else ""
    return iter_avg, iter_std, peak_str, is_oom, reason



def dp_runtime_settings(exp: Experiment) -> tuple[int, int, str | None]:
    if exp.zero_level == "none":
        replicate_degree = exp.dp
        shard_degree = -1 if exp.ep > 1 else 1
        return replicate_degree, shard_degree, None
    if exp.zero_level == "zero2":
        shard_degree = -1 if exp.ep > 1 else exp.dp
        return 1, shard_degree, "never"
    if exp.zero_level == "zero3":
        shard_degree = -1 if exp.ep > 1 else exp.dp
        return 1, shard_degree, "always"
    raise ValueError(f"Unsupported zero level: {exp.zero_level}")



def node_count(exp: Experiment) -> int:
    return exp.dp * exp.ep



def make_command(script_path: Path, exp: Experiment) -> list[str]:
    replicate_degree, shard_degree, reshard_after_forward = dp_runtime_settings(exp)
    train_args = [
        "--parallelism.pipeline_parallel_degree",
        str(exp.pp),
        "--parallelism.expert_parallel_degree",
        str(exp.ep),
        "--parallelism.pipeline_parallel_schedule",
        RUNTIME_SCHEDULES[exp.schedule],
        "--parallelism.pipeline_parallel_microbatch_size",
        str(exp.mb_size),
        "--parallelism.data_parallel_replicate_degree",
        str(replicate_degree),
        "--parallelism.data_parallel_shard_degree",
        str(shard_degree),
        "--training.seq_len",
        str(exp.seq_len),
        "--training.local_batch_size",
        str(local_batch_size(exp.pp, exp.mb_size)),
        "--training.global_batch_size",
        str(global_batch_size(exp.pp, exp.dp, exp.mb_size)),
    ]
    if reshard_after_forward is not None:
        train_args.extend(
            [
                "--parallelism.fsdp_reshard_after_forward",
                reshard_after_forward,
            ]
        )
    if exp.sweep == "schedule":
        train_args.append("--compile.no-enable")
    return [
        str(script_path),
        "--nnode",
        str(node_count(exp)),
        "--ngpu",
        str(exp.pp),
        "--module",
        "qwen3",
        "--config",
        exp.config,
        "--log-rank",
        ",".join(str(i) for i in range(exp.pp)),
        "--",
        *train_args,
    ]



def write_csv(path: Path, rows: list[Result]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(Result.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))



def parse_peak_memory_values(peak_memory_gb_by_rank: str) -> list[float]:
    summary = peak_memory_gb_by_rank.strip()
    if not summary:
        return []

    values: list[float] = []
    for item in summary.split("/"):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def throughput_tokens_per_s(row: Result) -> float | None:
    if row.iter_time_mean is None or row.iter_time_mean <= 0:
        return None
    return (row.global_batch_size * row.seq_len) / row.iter_time_mean


def throughput_std_tokens_per_s(row: Result) -> float | None:
    if row.iter_time_mean is None or row.iter_time_stddev is None or row.iter_time_mean <= 0:
        return None
    return ((row.global_batch_size * row.seq_len) / (row.iter_time_mean ** 2)) * row.iter_time_stddev


def save_bar_plot(
    title: str,
    xlabel: str,
    rows: list[Result],
    output_path: Path,
    *,
    allow_oom_markers: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [experiment_label_from_result(row) for row in rows]
    xs = list(range(len(rows)))

    success_rows = [row for row in rows if row.status == "ok" and throughput_tokens_per_s(row) is not None]
    success_max = max((float(throughput_tokens_per_s(row)) for row in success_rows), default=0.0)
    marker_y = success_max * 0.05 if success_max > 0 else 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6))
    for index, row in enumerate(rows):
        throughput = throughput_tokens_per_s(row)
        throughput_std = throughput_std_tokens_per_s(row)
        if row.status == "ok" and throughput is not None:
            ax.bar(
                index,
                float(throughput),
                yerr=float(throughput_std or 0.0),
                color="#4C72B0",
                ecolor="#2F2F2F",
                capsize=4,
            )
        elif allow_oom_markers and row.status == "oom":
            ax.scatter(index, marker_y, marker="x", s=120, color="red", linewidths=2.5)
        else:
            ax.scatter(index, marker_y, marker="x", s=100, color="black", linewidths=2.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_memory_plot(
    title: str,
    xlabel: str,
    rows: list[Result],
    output_path: Path,
    *,
    allow_oom_markers: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [experiment_label_from_result(row) for row in rows]
    xs = list(range(len(rows)))

    success_values = [
        value
        for row in rows
        if row.status == "ok"
        for value in parse_peak_memory_values(row.peak_memory_gb_by_rank)
    ]
    marker_y = max(success_values, default=0.0) * 0.05 if success_values else 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6))
    for index, row in enumerate(rows):
        if row.status == "ok":
            values = parse_peak_memory_values(row.peak_memory_gb_by_rank)
            if values:
                ax.scatter(
                    [index] * len(values),
                    values,
                    color="#DD8452",
                    s=36,
                    alpha=0.9,
                )
        elif allow_oom_markers and row.status == "oom":
            ax.scatter(index, marker_y, marker="x", s=120, color="red", linewidths=2.5)
        else:
            ax.scatter(index, marker_y, marker="x", s=100, color="black", linewidths=2.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def experiment_label_from_result(row: Result) -> str:
    if row.sweep == "scalability":
        return f"(pp={row.pp}, dp={row.dp})"
    if row.sweep == "zero":
        return f"(zero={row.zero_level}, mb={row.mb_size})"
    return row.schedule


def write_sweep_outputs(out_dir: Path, sweep: str, rows: list[Result]) -> None:
    csv_path = out_dir / f"{sweep}.csv"
    write_csv(csv_path, rows)

    if sweep == "scalability":
        title = "Qwen3 DP Scalability"
        subtitle = f"zero=none, schedule=1f1b, ep={rows[0].ep if rows else 1}, seq_len={rows[0].seq_len if rows else 0}"
    elif sweep == "zero":
        title = "Qwen3 ZeRO Sweep"
        subtitle = f"pp=8, dp=2, schedule=1f1b, ep={rows[0].ep if rows else 1}, seq_len={rows[0].seq_len if rows else 0}"
    else:
        title = "Qwen3 Schedule Sweep"
        subtitle = f"pp=8, dp=2, zero=none, ep={rows[0].ep if rows else 1}, seq_len={rows[0].seq_len if rows else 0}"

    plots_dir = out_dir / "plots"
    save_bar_plot(
        title,
        subtitle,
        rows,
        plots_dir / f"{sweep}.png",
        allow_oom_markers=(sweep == "zero"),
    )
    save_memory_plot(
        f"{title} Memory",
        subtitle,
        rows,
        plots_dir / f"{sweep}_memory.png",
        allow_oom_markers=(sweep == "zero"),
    )


def print_progress(done: int, total: int, results: list[Result]) -> None:
    counts = {"ok": 0, "oom": 0, "failed": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(
        f"progress: finished={done}/{total} ok={counts.get('ok',0)} oom={counts.get('oom',0)} failed={counts.get('failed',0)}"
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen e2e evaluation sweeps and generate plots/CSVs.")
    parser.add_argument("--script", default="scripts/run-qwen-ec2.sh", help="EC2 runner script path")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG, help=f"Base torchtitan config to override at runtime. Default: {DEFAULT_BASE_CONFIG}")
    parser.add_argument("--ep-degree", type=int, default=1, help="Expert parallel degree for all sweeps. When > 1, data_parallel_shard_degree is forced to -1.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length override for all sweeps")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: out/e2e-eval/<timestamp>")
    parser.add_argument(
        "--schedule-sweep",
        nargs="+",
        default=list(DEFAULT_SCHEDULE_SWEEP),
        help="Schedules for the schedule sweep. Supported values: 1f1b interleaved1f1b zerobubble dualpipe",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=["scalability", "zero", "schedule"],
        default=["scalability", "zero", "schedule"],
        help="Subset of e2e sweeps to run. Default: scalability zero schedule",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan experiments and output commands without running them")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        print(f"runner script not found: {script_path}", file=sys.stderr)
        return 1

    try:
        schedule_sweep = [normalize_schedule(s) for s in args.schedule_sweep]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    base_out = Path(args.out_dir).resolve() if args.out_dir else (Path("out") / "e2e-eval" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    logs_dir = base_out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(
        schedule_sweep=schedule_sweep,
        base_config=args.base_config,
        ep_degree=args.ep_degree,
        seq_len=args.seq_len,
        enabled_sweeps=set(args.sweeps),
    )
    total = len(experiments)
    results: list[Result] = []

    plan_path = base_out / "planned_experiments.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sweep",
            "config",
            "pp",
            "dp",
            "ep",
            "zero_level",
            "schedule",
            "mb_size",
            "seq_len",
            "local_batch_size",
            "global_batch_size",
            "nnode",
            "ngpu",
            "command",
        ])
        for exp in experiments:
            cmd = make_command(script_path, exp)
            writer.writerow([
                exp.sweep,
                exp.config,
                exp.pp,
                exp.dp,
                exp.ep,
                exp.zero_level,
                exp.schedule,
                exp.mb_size,
                exp.seq_len,
                local_batch_size(exp.pp, exp.mb_size),
                global_batch_size(exp.pp, exp.dp, exp.mb_size),
                node_count(exp),
                exp.pp,
                " ".join(cmd),
            ])

    print(f"output directory: {base_out}")
    print(f"planned experiments: {total}")

    if args.dry_run:
        for i, exp in enumerate(experiments, start=1):
            print(
                f"[{i}/{total}] {exp.sweep}: {experiment_label(exp)} config={exp.config} pp={exp.pp} dp={exp.dp} ep={exp.ep} seq_len={exp.seq_len}"
            )
            print("  " + " ".join(make_command(script_path, exp)))
        return 0

    for i, exp in enumerate(experiments, start=1):
        log_path = logs_dir / log_filename(i, exp)
        cmd = make_command(script_path, exp)
        print(f"[{i}/{total}] running sweep={exp.sweep} label={experiment_label(exp)} config={exp.config}")
        print(f"  log={log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)

        iter_mean, iter_std, peak_str, is_oom, reason = parse_log(log_path)
        if result.returncode == 0 and iter_mean is not None:
            status = "ok"
        elif is_oom:
            status = "oom"
            reason = reason or "oom"
        else:
            status = "failed"
            reason = reason or f"exit_code={result.returncode}"

        row = Result(
            sweep=exp.sweep,
            config=exp.config,
            pp=exp.pp,
            dp=exp.dp,
            ep=exp.ep,
            zero_level=exp.zero_level,
            schedule=exp.schedule,
            mb_size=exp.mb_size,
            seq_len=exp.seq_len,
            local_batch_size=local_batch_size(exp.pp, exp.mb_size),
            global_batch_size=global_batch_size(exp.pp, exp.dp, exp.mb_size),
            nnode=node_count(exp),
            ngpu=exp.pp,
            log_path=str(log_path),
            returncode=result.returncode,
            status=status,
            iter_time_mean=iter_mean,
            iter_time_stddev=iter_std,
            peak_memory_gb_by_rank=peak_str,
            failure_reason=reason,
        )
        results.append(row)
        print(f"[{i}/{total}] finished status={status} iter_mean={iter_mean} iter_std={iter_std}")
        print_progress(i, total, results)

    write_csv(base_out / "all_results.csv", results)
    for sweep in ("scalability", "zero", "schedule"):
        sweep_rows = [r for r in results if r.sweep == sweep]
        write_sweep_outputs(base_out, sweep, sweep_rows)

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print("failures:")
        for row in failed:
            print(f"  sweep={row.sweep} zero={row.zero_level} schedule={row.schedule} pp={row.pp} dp={row.dp} ep={row.ep} status={row.status} reason={row.failure_reason} log={row.log_path}")

    print(f"wrote combined results to {base_out / 'all_results.csv'}")
    print(f"wrote plots and per-sweep CSVs under {base_out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
