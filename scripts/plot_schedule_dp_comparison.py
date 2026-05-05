#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SYSTEM_COLORS = {
    "torchtitan": "#4C72B0",
    "megatron": "#55A868",
    "deepspeed": "#C44E52",
    "piper": "#8172B3",
}

SYSTEM_ORDER = ("torchtitan", "megatron", "deepspeed", "piper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine schedule sweep CSVs from multiple DP runs into one 1F1B comparison plot."
    )
    parser.add_argument("csv_paths", nargs="+", help="Input combined/schedule.csv files")
    parser.add_argument("--output-dir", required=True, help="Directory to write the merged outputs")
    parser.add_argument(
        "--schedule",
        default="1f1b",
        help="Schedule value to keep from the input CSVs (default: 1f1b)",
    )
    parser.add_argument(
        "--title",
        default="Qwen3 Schedule Comparison Across DP Sizes",
        help="Figure title",
    )
    return parser.parse_args()


def parse_peak_memory_values(raw: str) -> list[float]:
    raw = (raw or "").strip()
    if not raw:
        return []
    values: list[float] = []
    for chunk in raw.replace(";", "/").split("/"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            _, chunk = chunk.split(":", 1)
        try:
            values.append(float(chunk))
        except ValueError:
            continue
    return values


def load_rows(csv_paths: list[str], schedule: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["schedule"] != schedule:
                    continue
                if row["status"] != "ok":
                    continue
                row["source_csv"] = csv_path
                rows.append(row)
    return rows


def write_filtered_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(output_path: Path, rows: list[dict[str, str]], title: str, schedule: str) -> None:
    systems = [system for system in SYSTEM_ORDER if any(row["system"] == system for row in rows)]
    dp_values = sorted({int(row["dp"]) for row in rows})
    if not systems or not dp_values:
        raise ValueError("No matching rows found for plot")

    row_map = {(row["system"], int(row["dp"])): row for row in rows}
    x_positions = list(range(len(dp_values)))

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 5.5))

    for system in systems:
        color = SYSTEM_COLORS.get(system, "#4C72B0")
        xs: list[int] = []
        ys_time: list[float] = []
        ys_time_err: list[float] = []
        ys_mem: list[float] = []
        for x, dp in zip(x_positions, dp_values):
            row = row_map.get((system, dp))
            if row is None:
                continue
            xs.append(x)
            ys_time.append(float(row["iter_time_mean"]))
            ys_time_err.append(float(row["iter_time_stddev"] or 0.0))
            ys_mem.append(max(parse_peak_memory_values(row["peak_memory_gb_by_rank"]), default=0.0))

        if not xs:
            continue

        ax_time.errorbar(
            xs,
            ys_time,
            yerr=ys_time_err,
            marker="o",
            linewidth=2,
            capsize=4,
            color=color,
            label=system,
        )
        ax_mem.plot(
            xs,
            ys_mem,
            marker="o",
            linewidth=2,
            color=color,
            label=system,
        )

    tick_labels = [f"dp={dp}" for dp in dp_values]
    subtitle = f"{schedule}, pp=8, qwen3_9b, zero1, mb=4, seq_len=512"

    ax_time.set_title("Iteration Time")
    ax_time.set_ylabel("Seconds")
    ax_time.set_xticks(x_positions)
    ax_time.set_xticklabels(tick_labels)
    ax_time.grid(axis="y", linestyle="--", alpha=0.35)
    ax_time.set_ylim(bottom=0)

    ax_mem.set_title("Max Peak Memory Per Run")
    ax_mem.set_ylabel("GB")
    ax_mem.set_xticks(x_positions)
    ax_mem.set_xticklabels(tick_labels)
    ax_mem.grid(axis="y", linestyle="--", alpha=0.35)
    ax_mem.set_ylim(bottom=0)

    fig.suptitle(title)
    fig.text(0.5, 0.02, subtitle, ha="center")
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, title="System", loc="upper center", ncol=len(systems))
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv_paths, args.schedule)
    if not rows:
        raise SystemExit(f"No rows found for schedule={args.schedule}")

    rows.sort(key=lambda row: (int(row["dp"]), SYSTEM_ORDER.index(row["system"])))

    output_dir = Path(args.output_dir)
    write_filtered_csv(output_dir / "schedule_1f1b_dp_comparison.csv", rows)
    save_plot(
        output_dir / "schedule_1f1b_dp_comparison.png",
        rows,
        title=args.title,
        schedule=args.schedule,
    )


if __name__ == "__main__":
    main()
