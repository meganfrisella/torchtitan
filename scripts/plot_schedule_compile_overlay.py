#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a unified schedule TPS plot that overlays compiled results as dashed, "
            "unfilled bars on top of no-compile bars."
        )
    )
    parser.add_argument("no_compile_csv", help="Path to schedule.csv with compile_enabled=False")
    parser.add_argument("compile_csv", help="Path to schedule.csv with compile_enabled=True")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: <compile_csv_parent>/plots/schedule_compile_unified.png",
    )
    return parser.parse_args()


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if "compile_enabled" not in (reader.fieldnames or []):
            raise ValueError(f"Missing compile_enabled column in {path}")
    return rows


def _extract_schedule_tps(rows: list[dict[str, str]], *, compile_enabled: bool) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if _to_bool(row.get("compile_enabled", "false")) != compile_enabled:
            continue
        schedule = row.get("schedule", "").strip()
        if schedule not in {"1f1b", "interleaved1f1b", "dualpipe"}:
            continue
        gbs = float(row["global_batch_size"]) if row.get("global_batch_size") else None
        seq_len = float(row["seq_len"]) if row.get("seq_len") else None
        iter_mean = float(row["iter_time_mean"]) if row.get("iter_time_mean") else None
        iter_std = float(row["iter_time_stddev"]) if row.get("iter_time_stddev") else None
        if gbs is None or seq_len is None or iter_mean is None or iter_mean == 0:
            continue
        tokens_per_iter = gbs * seq_len
        tps = tokens_per_iter / iter_mean
        tps_std = tokens_per_iter * iter_std / (iter_mean**2) if iter_std is not None else 0.0
        out[schedule] = (tps, tps_std)
    return out


def main() -> int:
    args = parse_args()
    no_compile_path = Path(args.no_compile_csv).resolve()
    compile_path = Path(args.compile_csv).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else (compile_path.parent / "plots" / "schedule_compile_unified.png").resolve()
    )

    no_compile_rows = _load_rows(no_compile_path)
    compile_rows = _load_rows(compile_path)

    no_compile = _extract_schedule_tps(no_compile_rows, compile_enabled=False)
    compiled = _extract_schedule_tps(compile_rows, compile_enabled=True)

    schedule_order = ["1f1b", "interleaved1f1b", "dualpipe"]
    labels = [schedule for schedule in schedule_order if schedule in no_compile or schedule in compiled]
    if not labels:
        raise SystemExit("No valid 1f1b/interleaved1f1b/dualpipe rows found across the input CSVs.")

    x_positions = list(range(len(labels)))
    width = 0.62

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 6))

    no_compile_handle = None
    compile_handle = None
    marker_height = 1.0

    for x, label in zip(x_positions, labels):
        no_compile_entry = no_compile.get(label)
        compiled_entry = compiled.get(label)

        if no_compile_entry is not None:
            tps, tps_std = no_compile_entry
            marker_height = max(marker_height, tps * 0.05)
            bars = ax.bar(
                x,
                tps,
                width=width,
                yerr=tps_std,
                color="#4C72B0",
                ecolor="#2F2F2F",
                capsize=4,
                label="No Compile" if no_compile_handle is None else None,
                zorder=2,
            )
            if no_compile_handle is None:
                no_compile_handle = bars[0]
        else:
            ax.scatter(x, marker_height, marker="x", s=100, color="#4C72B0", linewidths=2.0, zorder=3)

        if compiled_entry is not None:
            tps, tps_std = compiled_entry
            marker_height = max(marker_height, tps * 0.05)
            bars = ax.bar(
                x,
                tps,
                width=width,
                yerr=tps_std,
                fill=False,
                edgecolor="#C44E52",
                linewidth=2.2,
                linestyle="--",
                ecolor="#C44E52",
                capsize=4,
                label="Compile Enabled" if compile_handle is None else None,
                zorder=3,
            )
            if compile_handle is None:
                compile_handle = bars[0]
        else:
            ax.scatter(x, marker_height, marker="x", s=100, color="#C44E52", linewidths=2.0, zorder=3)

    ax.set_title("Unified Schedule TPS (Compile vs No Compile)")
    ax.set_xlabel("Schedule")
    ax.set_ylabel("Tokens / Second")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
