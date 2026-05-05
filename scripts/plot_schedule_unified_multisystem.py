#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

TITLE_FONTSIZE = 34
AXIS_LABEL_FONTSIZE = 30
XTICK_FONTSIZE = 26
YTICK_FONTSIZE = 22
LEGEND_FONTSIZE = 26

SCHEDULE_ORDER = ["1f1b", "interleaved1f1b", "dualpipe"]
SYSTEM_COLORS = {
    "megatron": "#55A868",
    "torchtitan": "#4C72B0",
    "piper": "#8172B3",
}
SYSTEM_DISPLAY_NAMES = {
    "megatron": "Megatron",
    "torchtitan": "TorchTitan",
    "piper": "Flexo",
}
SYSTEM_COMPILE_DISPLAY_NAMES = {
    "megatron": "Megatron + Inductor",
    "torchtitan": "TorchTitan + Inductor",
    "piper": "Piper + Inductor",
}
SCHEDULE_DISPLAY_NAMES = {
    "1f1b": "1F1B",
    "interleaved1f1b": "Int-1F1B",
    "dualpipe": "DualPipeV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 2-panel unified schedule TPS plot: "
            "Qwen3 1B (Megatron + TorchTitan + Piper) and "
            "Qwen3 9B (Megatron + Piper), with compile overlays."
        )
    )
    parser.add_argument("--megatron-1b", required=True, help="Qwen3 1B Megatron schedule.csv path")
    parser.add_argument("--torchtitan-no-compile-1b", required=True, help="Qwen3 1B TorchTitan no-compile schedule.csv path")
    parser.add_argument("--torchtitan-compile-1b", required=True, help="Qwen3 1B TorchTitan compile schedule.csv path")
    parser.add_argument("--piper-no-compile-1b", required=True, help="Qwen3 1B Piper no-compile schedule.csv path")
    parser.add_argument("--piper-compile-1b", required=True, help="Qwen3 1B Piper compile schedule.csv path")
    parser.add_argument("--megatron-9b", required=True, help="Qwen3 9B Megatron schedule.csv path")
    parser.add_argument("--piper-no-compile-9b", required=True, help="Qwen3 9B Piper no-compile schedule.csv path")
    parser.add_argument("--piper-compile-9b", required=False, default=None, help="Qwen3 9B Piper compile schedule.csv path")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable compile overlay bars and compile legend entries.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path. "
            "Default: <torchtitan-compile-1b-parent>/plots/schedule_unified_multisystem_compile_1b_9b.png"
        ),
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def _compute_tps(gbs: float | None, seq_len: float | None, iter_mean: float | None) -> float | None:
    if gbs is None or seq_len is None or iter_mean is None or iter_mean == 0:
        return None
    return (gbs * seq_len) / iter_mean


def _compute_tps_std(gbs: float | None, seq_len: float | None, iter_mean: float | None, iter_std: float | None) -> float | None:
    if gbs is None or seq_len is None or iter_mean is None or iter_std is None or iter_mean == 0:
        return None
    return (gbs * seq_len) * iter_std / (iter_mean**2)


def _extract_schedule_tps(rows: list[dict[str, str]], expected_system: str) -> dict[str, tuple[float, float]]:
    data: dict[str, tuple[float, float]] = {}
    for row in rows:
        if row.get("system") != expected_system:
            continue
        if row.get("status") != "ok":
            continue
        schedule = (row.get("schedule") or "").strip()
        if schedule not in SCHEDULE_ORDER:
            continue

        gbs = _to_float_or_none(row.get("global_batch_size"))
        seq_len = _to_float_or_none(row.get("seq_len"))
        iter_mean = _to_float_or_none(row.get("iter_time_mean"))
        iter_std = _to_float_or_none(row.get("iter_time_stddev"))

        tps = _compute_tps(gbs, seq_len, iter_mean)
        if tps is None:
            continue

        tps_std = _compute_tps_std(gbs, seq_len, iter_mean, iter_std)
        data[schedule] = (tps, tps_std or 0.0)
    return data


def _plot_labels_for(
    systems: list[str],
    base_by_system: dict[str, dict[str, tuple[float, float]]],
    compile_by_system: dict[str, dict[str, tuple[float, float]]],
) -> list[str]:
    return [
        schedule
        for schedule in SCHEDULE_ORDER
        if any(schedule in base_by_system.get(system, {}) for system in systems)
        or any(schedule in compile_by_system.get(system, {}) for system in systems)
    ]


def _draw_subplot(
    ax: plt.Axes,
    *,
    title: str,
    panel_label: str | None,
    systems: list[str],
    base_by_system: dict[str, dict[str, tuple[float, float]]],
    compile_by_system: dict[str, dict[str, tuple[float, float]]],
    group_tags: dict[str, str] | None = None,
) -> None:
    def _sci_fmt(y: float, _pos: int) -> str:
        if y == 0:
            return "0"
        sign = "-" if y < 0 else ""
        y_abs = abs(y)
        exponent = int(f"{y_abs:e}".split("e")[1])
        mantissa = y_abs / (10 ** exponent)
        if abs(mantissa - round(mantissa)) < 1e-9:
            mantissa_text = str(int(round(mantissa)))
        else:
            mantissa_text = f"{mantissa:.1f}".rstrip("0").rstrip(".")
        return f"{sign}{mantissa_text}e{exponent}"

    labels = _plot_labels_for(systems, base_by_system, compile_by_system)
    if not labels:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.text(0.5, 0.5, "No plottable rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    x_positions = list(range(len(labels)))
    width = 0.78 / len(systems)
    base_handles: dict[str, object] = {}
    compile_handles: dict[str, object] = {}
    max_tps = 0.0

    for x, schedule in zip(x_positions, labels):
        present_systems = [
            system
            for system in systems
            if schedule in base_by_system.get(system, {}) or schedule in compile_by_system.get(system, {})
        ]
        if not present_systems:
            continue

        for present_idx, system in enumerate(present_systems):
            offset = (present_idx - (len(present_systems) - 1) / 2) * width
            bar_x = x + offset
            color = SYSTEM_COLORS[system]
            base_entry = base_by_system.get(system, {}).get(schedule)
            if base_entry is not None:
                tps, tps_std = base_entry
                max_tps = max(max_tps, tps)
                bars = ax.bar(
                    bar_x,
                    tps,
                    width=width * 0.92,
                    yerr=tps_std,
                    color=color,
                    ecolor="#2F2F2F",
                    capsize=3,
                    label=system if system not in base_handles else None,
                    zorder=2,
                )
                if system not in base_handles:
                    base_handles[system] = bars[0]

            compile_entry = compile_by_system.get(system, {}).get(schedule)
            if compile_entry is not None and base_entry is not None:
                tps_c, tps_std_c = compile_entry
                max_tps = max(max_tps, tps_c)
                base_tps = base_entry[0]
                compile_height = tps_c - base_tps
                if compile_height <= 0:
                    continue
                bars = ax.bar(
                    bar_x,
                    compile_height,
                    width=width * 0.92,
                    yerr=tps_std_c,
                    bottom=base_tps,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.2,
                    linestyle="--",
                    ecolor=color,
                    capsize=3,
                    label=SYSTEM_COMPILE_DISPLAY_NAMES[system] if system not in compile_handles else None,
                    zorder=3,
                )
                if system not in compile_handles:
                    compile_handles[system] = bars[0]

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=40)
    if panel_label:
        ax.text(
            0.5,
            1.05,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=XTICK_FONTSIZE,
        )
    ax.set_ylabel("Tokens / Second", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_positions)
    xticklabels: list[str] = []
    for label in labels:
        schedule_display = SCHEDULE_DISPLAY_NAMES.get(label, label)
        tag = group_tags.get(label) if group_tags else None
        xticklabels.append(f"{schedule_display}\n({tag})" if tag else schedule_display)
    ax.set_xticklabels(xticklabels, fontsize=XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(_sci_fmt))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0, top=max_tps * 1.2 if max_tps > 0 else 1.0)
    _ = base_handles, compile_handles


def main() -> int:
    args = parse_args()

    megatron_1b_path = Path(args.megatron_1b).resolve()
    tt_nc_1b_path = Path(args.torchtitan_no_compile_1b).resolve()
    tt_c_1b_path = Path(args.torchtitan_compile_1b).resolve()
    piper_nc_1b_path = Path(args.piper_no_compile_1b).resolve()
    piper_c_1b_path = Path(args.piper_compile_1b).resolve()

    megatron_9b_path = Path(args.megatron_9b).resolve()
    piper_nc_9b_path = Path(args.piper_no_compile_9b).resolve()
    piper_c_9b_path = Path(args.piper_compile_9b).resolve() if args.piper_compile_9b else None

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (tt_c_1b_path.parent / "plots" / "schedule_unified_multisystem_compile_1b_9b.png").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_by_system_1b = {
        "megatron": _extract_schedule_tps(_read_rows(megatron_1b_path), "megatron"),
        "torchtitan": _extract_schedule_tps(_read_rows(tt_nc_1b_path), "torchtitan"),
        "piper": _extract_schedule_tps(_read_rows(piper_nc_1b_path), "piper"),
    }
    compile_by_system_1b = (
        {}
        if args.no_compile
        else {
            "torchtitan": _extract_schedule_tps(_read_rows(tt_c_1b_path), "torchtitan"),
            "piper": _extract_schedule_tps(_read_rows(piper_c_1b_path), "piper"),
        }
    )

    base_by_system_9b = {
        "megatron": _extract_schedule_tps(_read_rows(megatron_9b_path), "megatron"),
        "piper": _extract_schedule_tps(_read_rows(piper_nc_9b_path), "piper"),
    }
    compile_by_system_9b = (
        {}
        if args.no_compile
        else {
            "piper": _extract_schedule_tps(_read_rows(piper_c_9b_path), "piper") if piper_c_9b_path is not None else {},
        }
    )

    labels_1b = _plot_labels_for(["megatron", "torchtitan", "piper"], base_by_system_1b, compile_by_system_1b)
    labels_9b = _plot_labels_for(["megatron", "piper"], base_by_system_9b, compile_by_system_9b)
    all_tags = [
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
        "xi",
        "xii",
        "xiii",
        "xiv",
        "xv",
        "xvi",
        "xvii",
        "xviii",
        "xix",
        "xx",
        "xxi",
        "xxii",
        "xxiii",
        "xxiv",
        "xxv",
        "xxvi",
    ]
    labels_1b_tags = {label: all_tags[idx] for idx, label in enumerate(labels_1b)}
    labels_9b_tags = {label: all_tags[len(labels_1b) + idx] for idx, label in enumerate(labels_9b)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False, gridspec_kw={"width_ratios": [3, 1]})
    _draw_subplot(
        axes[0],
        title="Qwen3 1B",
        panel_label="(a)",
        systems=["megatron", "torchtitan", "piper"],
        base_by_system=base_by_system_1b,
        compile_by_system=compile_by_system_1b,
        group_tags=labels_1b_tags,
    )
    _draw_subplot(
        axes[1],
        title="Qwen3 9B",
        panel_label="(b)",
        systems=["megatron", "piper"],
        base_by_system=base_by_system_9b,
        compile_by_system=compile_by_system_9b,
        group_tags=labels_9b_tags,
    )

    legend_handles: list[Patch] = []
    legend_labels: list[str] = []
    for system in ["megatron", "torchtitan", "piper"]:
        if _plot_labels_for([system], base_by_system_1b, compile_by_system_1b):
            legend_handles.append(Patch(facecolor=SYSTEM_COLORS[system], edgecolor=SYSTEM_COLORS[system]))
            legend_labels.append(SYSTEM_DISPLAY_NAMES[system])
        has_compile_1b = bool(compile_by_system_1b.get(system))
        has_compile_9b = bool(compile_by_system_9b.get(system))
        if has_compile_1b or has_compile_9b:
            legend_handles.append(
                Patch(facecolor="none", edgecolor=SYSTEM_COLORS[system], linewidth=2.2, linestyle="--")
            )
            legend_labels.append(SYSTEM_COMPILE_DISPLAY_NAMES[system])

    fig.tight_layout()
    fig.subplots_adjust(top=0.84, bottom=0.28)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.00),
            ncol=max(1, len(legend_labels)),
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
        )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
