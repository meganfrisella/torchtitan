#!/usr/bin/env python3
from __future__ import annotations

import atexit
import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TT_ITER_RE = re.compile(r"Final \d+ iter times.*?avg:\s*([\d.]+)\s*s,\s*std:\s*([\d.]+)\s*s")
TT_MEM_RE = re.compile(r"\[rank(\d+)\].*?memory:\s*([\d.]+)GiB")
TT_FINAL_BY_RANK_RE = re.compile(
    r"\[rank(\d+)\].*?Final \d+ iter times.*?avg:\s*([\d.]+)\s*s,\s*std:\s*([\d.]+)\s*s"
)
DS_STEP_RE = re.compile(r"\[Step\s+\d+/\d+\].*?step_time=([\d.]+)s")
DS_MEM_RE = re.compile(r"\[rank(\d+)\]\s+peak_memory_allocated_gb=([\d.]+)\s+peak_memory_reserved_gb=([\d.]+)")
MG_ITER_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)")
MG_MEM_RE = re.compile(r"\[Rank\s+(\d+)\].*?max allocated:\s*([\d.]+)\s*\|")
OOM_RE = re.compile(r"(?:cuda\s+out\s+of\s+memory|out\s+of\s+memory|std::bad_alloc|\boom\b)", re.IGNORECASE)
WATCHDOG_TIMEOUT_RE = re.compile(r"Watchdog caught collective operation timeout|Process group watchdog thread terminated with exception", re.IGNORECASE)
CHILD_FAILED_RE = re.compile(r"ChildFailedError|ERROR conda\.cli\.main_run:execute", re.IGNORECASE)
DEFAULT_TORCHTITAN_PYTHONPATH = "/workspace/torchtitan"
DEFAULT_SCHEDULE_SWEEP = ("1f1b", "interleaved1f1b", "zerobubble", "dualpipe")
DEFAULT_SYSTEMS = ("torchtitan", "megatron", "deepspeed", "piper")
PIPER_RUNNER = Path("/m-coriander/coriander/mfris/piper/scripts/piper-run-qwen-ec2.sh")
SYSTEM_COLORS = {
    "torchtitan": "#4C72B0",
    "megatron": "#55A868",
    "deepspeed": "#C44E52",
    "piper": "#8172B3",
}


@dataclass
class Experiment:
    system: str
    sweep: str
    config: str
    pp: int
    dp: int
    ep: int
    zero_level: str
    schedule: str
    mb_size: int
    seq_len: int
    gradient_accumulation: bool = True
    bucket_size_mb: float | None = None
    ar_a2a_same_stream: bool = False
    overlap_chunks: bool = False


@dataclass
class Result:
    system: str
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
    metrics_path: str
    returncode: int
    status: str
    iter_time_mean: float | None
    iter_time_stddev: float | None
    peak_memory_gb_by_rank: str
    failure_reason: str


_ACTIVE_CHILD: subprocess.Popen[str] | None = None


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

PIPER_SCHEDULES = {
    "1f1b": "1f1b",
    "interleaved1f1b": "interleaved-1f1b",
    "zerobubble": "interleaved-zerobubble",
    "dualpipe": "dualpipev",
}

PIPER_ZERO_STAGES = {
    "zero0": 0,
    "zero1": 1,
    "zero2": 2,
    "zero3": 3,
}


def system_runner(system: str) -> str:
    return {
        "torchtitan": "scripts/run-qwen-torchtitan.sh",
        "megatron": "scripts/run-qwen-megatron.sh",
        "deepspeed": "scripts/run-qwen-deepspeed.sh",
        "piper": "scripts/run-qwen-piper.sh",
    }[system]


def build_experiments(
    *,
    systems: Iterable[str],
    schedule_sweep: Iterable[str],
    seq_len: int,
    enabled_sweeps: set[str],
    gradient_accumulation: bool,
    ar_a2a_same_stream: bool,
    overlap_chunks: bool,
    bucket_size_mb: float | None,
) -> list[Experiment]:
    experiments: list[Experiment] = []

    scalability_defaults = {
        "config": "qwen3_9b",
        "ep": 1,
        "zero_level": "zero0",
        "schedule": "1f1b",
        "mb_size": 4,
    }
    zero_defaults = {
        "config": "qwen3_1b",
        "pp": 8,
        "dp": 2,
        "ep": 1,
        "schedule": "1f1b",
    }
    zero_levels_by_system = {
        "torchtitan": ("zero1", "zero2", "zero3"),
        "megatron": ("zero1",),
        "deepspeed": ("zero1",),
        "piper": ("zero1", "zero2", "zero3"),
    }
    zero_sweep_values = (16, 24, 32, 34, 36, 38, 40)
    schedule_defaults = {
        "config": "qwen3_9b",
        "pp": 8,
        "dp": 2,
        "ep": 1,
        "zero_level": "zero1",
        "mb_size": 4,
    }
    local_defaults = {
        "config": "qwen3_1b",
        "pp": 1,
        "dp": 1,
        "ep": 1,
        "zero_level": "zero1",
        "schedule": "1f1b",
        "mb_size": 4,
    }
    supported_schedules_by_system = {
        "torchtitan": {"1f1b", "interleaved1f1b", "dualpipe"},
        "megatron": {"1f1b", "interleaved1f1b"},
        "deepspeed": {"1f1b"},
        "piper": {"1f1b", "interleaved1f1b", "dualpipe"},
    }

    for system in systems:
        if "scalability" in enabled_sweeps:
            defaults = dict(scalability_defaults)
            for pp in (4,8,):
                for dp in (1,2,4):
                    experiments.append(
                        Experiment(
                            system=system,
                            sweep="scalability",
                            config=str(defaults["config"]),
                            pp=pp,
                            dp=dp,
                            ep=int(defaults["ep"]),
                            zero_level=str(defaults["zero_level"]),
                            schedule=str(defaults["schedule"]),
                            mb_size=int(defaults["mb_size"]),
                            seq_len=seq_len,
                            gradient_accumulation=gradient_accumulation,
                            bucket_size_mb=bucket_size_mb,
                            ar_a2a_same_stream=ar_a2a_same_stream,
                            overlap_chunks=overlap_chunks,
                        )
                    )

        if "zero" in enabled_sweeps:
            defaults = dict(zero_defaults)
            for zero_level in zero_levels_by_system[system]:
                for sweep_value in zero_sweep_values:
                    experiments.append(
                        Experiment(
                            system=system,
                            sweep="zero",
                            config=str(defaults["config"]),
                            pp=int(defaults["pp"]),
                            dp=int(defaults["dp"]),
                            ep=int(defaults["ep"]),
                            zero_level=zero_level,
                            schedule=str(defaults["schedule"]),
                            mb_size=sweep_value,
                            seq_len=seq_len,
                            gradient_accumulation=gradient_accumulation,
                            bucket_size_mb=bucket_size_mb,
                            ar_a2a_same_stream=ar_a2a_same_stream,
                            overlap_chunks=overlap_chunks,
                        )
                    )

        if "schedule" in enabled_sweeps:
            defaults = schedule_defaults
            for schedule in schedule_sweep:
                if schedule not in supported_schedules_by_system[system]:
                    continue
                experiments.append(
                    Experiment(
                        system=system,
                        sweep="schedule",
                        config=str(defaults["config"]),
                        pp=int(defaults["pp"]),
                        dp=int(defaults["pp"]),
                        ep=int(defaults["ep"]),
                        zero_level=str(defaults["zero_level"]),
                        schedule=schedule,
                        mb_size=int(defaults["mb_size"]),
                        seq_len=seq_len,
                        gradient_accumulation=gradient_accumulation,
                        bucket_size_mb=bucket_size_mb,
                        ar_a2a_same_stream=ar_a2a_same_stream,
                        overlap_chunks=overlap_chunks,
                    )
                )

        if "local" in enabled_sweeps:
            defaults = dict(local_defaults)
            experiments.append(
                Experiment(
                    system=system,
                    sweep="local",
                    config=str(defaults["config"]),
                    pp=int(defaults["pp"]),
                    dp=int(defaults["dp"]),
                    ep=int(defaults["ep"]),
                    zero_level=str(defaults["zero_level"]),
                    schedule=str(defaults["schedule"]),
                    mb_size=int(defaults["mb_size"]),
                    seq_len=seq_len,
                    gradient_accumulation=gradient_accumulation,
                    bucket_size_mb=bucket_size_mb,
                    ar_a2a_same_stream=ar_a2a_same_stream,
                    overlap_chunks=overlap_chunks,
                )
            )

    return experiments


def local_batch_size(exp: Experiment) -> int:
    if exp.sweep == "local":
        return exp.mb_size
    return exp.pp * 2 * exp.mb_size


def global_batch_size(exp: Experiment) -> int:
    return local_batch_size(exp) * exp.dp


def node_count(exp: Experiment) -> int:
    return exp.dp


def experiment_label(exp: Experiment) -> str:
    if exp.sweep == "scalability":
        return f"pp={exp.pp}, dp={exp.dp}"
    if exp.sweep == "zero":
        return f"zero={exp.zero_level}, mb={exp.mb_size}"
    if exp.sweep == "local":
        return "local"
    return exp.schedule


def log_filename(index: int, exp: Experiment) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        f"{index:02d}_{exp.sweep}__pp{exp.pp}_dp{exp.dp}_ep{exp.ep}"
        f"__{exp.zero_level}__{exp.schedule}__mb{exp.mb_size}__sl{exp.seq_len}_{timestamp}.log"
    )


def make_command(
    exp: Experiment,
    *,
    enable_nsight: bool = False,
    fetch_dir: Path | None = None,
    piper_warmup: int = 3,
    piper_iters: int = 10,
    piper_ray_port: int = 6379,
    piper_remote_output_dir: str = "/tmp/piper/out",
) -> list[str]:
    command = [system_runner(exp.system)]
    if exp.system == "torchtitan":
        command.extend(["--nnode", str(node_count(exp)), "--ngpu", str(exp.pp)])
        if enable_nsight:
            command.append("--nsight")
        tt_args = [
            "--parallelism.pipeline_parallel_degree",
            str(exp.pp),
            "--parallelism.expert_parallel_degree",
            str(exp.ep),
            "--parallelism.pipeline_parallel_schedule",
            RUNTIME_SCHEDULES[exp.schedule],
            "--parallelism.pipeline_parallel_microbatch_size",
            str(exp.mb_size),
            "--parallelism.data_parallel_replicate_degree",
            str(exp.dp if exp.zero_level == "zero1" else 1),
            "--parallelism.data_parallel_shard_degree",
            str(1 if exp.zero_level == "zero1" else exp.dp),
            "--training.seq_len",
            str(exp.seq_len),
            "--training.local_batch_size",
            str(local_batch_size(exp)),
            "--training.global_batch_size",
            str(global_batch_size(exp)),
        ]
        hf_assets_path = torchtitan_hf_assets_path(exp.config)
        if hf_assets_path is not None:
            tt_args.extend(["--hf_assets_path", hf_assets_path])
        command.extend(
            [
                "--module",
                "qwen3",
                "--config",
                exp.config,
                "--log-rank",
                ",".join(str(i) for i in range(exp.pp)),
                "--",
                *tt_args,
            ]
        )
        if exp.zero_level == "zero2":
            command.extend(["--parallelism.fsdp_reshard_after_forward", "never"])
        elif exp.zero_level == "zero3":
            command.extend(["--parallelism.fsdp_reshard_after_forward", "always"])
        if exp.sweep == "schedule":
            command.append("--compile.no-enable")
    elif exp.system == "megatron":
        command.extend(["--nnode", str(node_count(exp)), "--ngpu", str(exp.pp)])
        if enable_nsight:
            command.append("--nsight")
        command.extend(
            [
                "--model",
                exp.config,
                "--",
                "--pp",
                str(exp.pp),
                "--dp",
                str(exp.dp),
                "--ep",
                str(exp.ep),
                "--micro-bs",
                str(exp.mb_size),
                "--global-bs",
                str(global_batch_size(exp)),
                "--seq-length",
                str(exp.seq_len),
                "--schedule",
                exp.schedule,
                "--zero-level",
                exp.zero_level,
                "--train-iters",
                "8",
            ]
        )
    elif exp.system == "deepspeed":
        command.extend(["--nnode", str(node_count(exp)), "--ngpu", str(exp.pp)])
        command.extend(
            [
                "--model",
                exp.config,
                "--",
                "--pp",
                str(exp.pp),
                "--dp",
                str(exp.dp),
                "--ep",
                str(exp.ep),
                "--micro-bs",
                str(exp.mb_size),
                "--global-bs",
                str(global_batch_size(exp)),
                "--seq-len",
                str(exp.seq_len),
                "--schedule",
                exp.schedule,
                "--zero-stage",
                {"zero1": "1", "zero2": "2", "zero3": "3"}[exp.zero_level],
                "--steps",
                "8",
            ]
        )
    else:
        if fetch_dir is None:
            raise ValueError("fetch_dir is required for piper commands")
        command.extend(
            [
                "--nnode",
                str(node_count(exp)),
                "--ngpu",
                str(exp.pp),
                "--model",
                exp.config,
                "--out-dir",
                str(fetch_dir),
                "--",
                "--schedule",
                piper_schedule_name(exp.schedule),
                "--zero-stage",
                str(PIPER_ZERO_STAGES[exp.zero_level]),
                "--batch-size",
                str(exp.mb_size),
                "--seq-len",
                str(exp.seq_len),
                "--mbs",
                str(piper_num_mbs(exp)),
                "--warmup",
                str(piper_warmup),
                "--iters",
                str(piper_iters),
                "--ray-port",
                str(piper_ray_port),
                "--remote-output-dir",
                f"{piper_remote_output_dir.rstrip('/')}/{piper_experiment_name(exp, enable_nsight)}",
                "--gradient-accumulation" if exp.gradient_accumulation else "--no-gradient-accumulation",
                "--ar-a2a-same-stream" if exp.ar_a2a_same_stream else "--no-ar-a2a-same-stream",
                "--overlap-chunks" if exp.overlap_chunks else "--no-overlap-chunks",
                "--use-inductor" if exp.sweep != "schedule" else "--no-use-inductor",
            ]
        )
        if exp.ep > 1:
            command.append("--ep")
        if enable_nsight:
            command.append("--nsight")
        if exp.bucket_size_mb is not None:
            command.extend(["--bucket-size", f"{exp.bucket_size_mb:g}"])
    return command


def torchtitan_hf_assets_path(config: str) -> str | None:
    return {
        "qwen3_1b": "/workspace/assets/hf/Qwen3-0.6B",
        "qwen3_9b": "/workspace/assets/hf/Qwen3-8B",
    }.get(config)


def inner_command(
    exp: Experiment,
    *,
    enable_nsight: bool = False,
    piper_warmup: int = 3,
    piper_iters: int = 10,
    piper_ray_port: int = 6379,
    piper_remote_output_dir: str = "/tmp/piper/out",
) -> str:
    head_private_ip = os.environ.get("HEAD_PRIVATE_IP", "<head_private_ip>")

    if exp.system == "torchtitan":
        train_script = "./run_train_nsys.sh" if enable_nsight else "./run_train.sh"
        args = [
            "--parallelism.pipeline_parallel_degree",
            str(exp.pp),
            "--parallelism.expert_parallel_degree",
            str(exp.ep),
            "--parallelism.pipeline_parallel_schedule",
            RUNTIME_SCHEDULES[exp.schedule],
            "--parallelism.pipeline_parallel_microbatch_size",
            str(exp.mb_size),
            "--parallelism.data_parallel_replicate_degree",
            str(exp.dp if exp.zero_level == "zero1" else 1),
            "--parallelism.data_parallel_shard_degree",
            str(1 if exp.zero_level == "zero1" else exp.dp),
            "--training.seq_len",
            str(exp.seq_len),
            "--training.local_batch_size",
            str(local_batch_size(exp)),
            "--training.global_batch_size",
            str(global_batch_size(exp)),
        ]
        hf_assets_path = torchtitan_hf_assets_path(exp.config)
        if hf_assets_path is not None:
            args.extend(["--hf_assets_path", hf_assets_path])
        if exp.zero_level == "zero2":
            args.extend(["--parallelism.fsdp_reshard_after_forward", "never"])
        elif exp.zero_level == "zero3":
            args.extend(["--parallelism.fsdp_reshard_after_forward", "always"])
        if exp.sweep == "schedule":
            args.append("--compile.no-enable")
        env_prefix = [
            f"PYTHONPATH={DEFAULT_TORCHTITAN_PYTHONPATH}",
            f"NNODE={node_count(exp)}",
            f"NGPU={exp.pp}",
            f"LOG_RANK={','.join(str(i) for i in range(exp.pp))}",
            "MODULE=qwen3",
            f"CONFIG={exp.config}",
            "NODE_RANK=<node_rank>",
            f"MASTER_ADDR={head_private_ip}",
            "MASTER_PORT=29500",
        ]
        return shlex.join(env_prefix + [train_script] + args)

    if exp.system == "megatron":
        args = [
            "conda",
            "run",
            "-n",
            "megatron",
            "python",
            "/workspace/torchtitan/run_megatron.py",
            "--model",
            exp.config,
            "--nnodes",
            str(node_count(exp)),
            "--nproc-per-node",
            str(exp.pp),
            "--master-addr",
            head_private_ip,
            "--master-port",
            "29500",
            "--disable-background-mode",
            "--pp",
            str(exp.pp),
            "--dp",
            str(exp.dp),
            "--ep",
            str(exp.ep),
            "--micro-bs",
            str(exp.mb_size),
            "--global-bs",
            str(global_batch_size(exp)),
            "--seq-length",
            str(exp.seq_len),
            "--schedule",
            exp.schedule,
            "--zero-level",
            exp.zero_level,
            "--train-iters",
            "8",
        ]
        return shlex.join(args)

    if exp.system == "deepspeed":
        args = [
            "conda",
            "run",
            "-n",
            "deepspeed",
            "torchrun",
            f"--nnodes={node_count(exp)}",
            f"--nproc_per_node={exp.pp}",
            "--node_rank=<node_rank>",
            f"--master_addr={head_private_ip}",
            "--master_port=29501",
            "run_deepspeed.py",
            "--model",
            exp.config,
            "--pp",
            str(exp.pp),
            "--dp",
            str(exp.dp),
            "--ep",
            str(exp.ep),
            "--micro-bs",
            str(exp.mb_size),
            "--global-bs",
            str(global_batch_size(exp)),
            "--seq-len",
            str(exp.seq_len),
            "--schedule",
            exp.schedule,
            "--zero-stage",
            {"zero1": "1", "zero2": "2", "zero3": "3"}[exp.zero_level],
            "--steps",
            "8",
        ]
        return shlex.join(args)

    args = [
        "python3",
        "-m",
        "test.test_qwen",
        "--model",
        piper_model_name(exp.config),
        "--schedule",
        piper_schedule_name(exp.schedule),
        "--pp",
        str(exp.pp),
        "--dp",
        str(exp.dp),
        "--zero-stage",
        str(PIPER_ZERO_STAGES[exp.zero_level]),
        "--batch-size",
        str(exp.mb_size),
        "--seq-len",
        str(exp.seq_len),
        "--mbs",
        str(piper_num_mbs(exp)),
        "--warmup",
        str(piper_warmup),
        "--iters",
        str(piper_iters),
        "--address",
        head_private_ip,
        "--port",
        str(piper_ray_port),
        "--output-dir",
        f"{piper_remote_output_dir.rstrip('/')}/{piper_experiment_name(exp, enable_nsight)}",
        "--temp-dir",
        "/tmp/piper/ray_tmp",
        "--gradient-accumulation" if exp.gradient_accumulation else "--no-gradient-accumulation",
        "--ar-a2a-same-stream" if exp.ar_a2a_same_stream else "--no-ar-a2a-same-stream",
        "--overlap-chunks" if exp.overlap_chunks else "--no-overlap-chunks",
        "--use-inductor" if exp.sweep != "schedule" else "--no-use-inductor",
    ]
    if exp.ep > 1:
        args.append("--ep")
    if enable_nsight:
        args.append("--nsight")
    if exp.bucket_size_mb is not None:
        args.extend(["--bucket-size", f"{exp.bucket_size_mb:g}"])
    return shlex.join(args)


def parse_log(
    system: str,
    log_path: Path,
    *,
    exp: Experiment | None = None,
    args: argparse.Namespace | None = None,
    system_out: Path | None = None,
    returncode: int = 0,
) -> tuple[float | None, float | None, str, bool, str, str]:
    raw = log_path.read_bytes().replace(b"\0", b"")
    text = raw.decode("utf-8", errors="replace")
    is_oom = OOM_RE.search(text) is not None
    reason = "oom" if is_oom else ""

    if system == "torchtitan":
        peak: dict[int, float] = {}
        iter_by_rank: dict[int, tuple[float, float]] = {}
        last_iter: tuple[float, float] | None = None
        for line in text.splitlines():
            line = ANSI_RE.sub("", line)
            m = TT_FINAL_BY_RANK_RE.search(line)
            if m:
                iter_by_rank[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                last_iter = (float(m.group(2)), float(m.group(3)))
                continue
            m = TT_ITER_RE.search(line)
            if m:
                last_iter = (float(m.group(1)), float(m.group(2)))
            m = TT_MEM_RE.search(line)
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
        return iter_avg, iter_std, peak_str, is_oom, reason, ""

    if system == "deepspeed":
        step_times = [float(m.group(1)) for m in DS_STEP_RE.finditer(text)]
        if len(step_times) > 5:
            step_times = step_times[-5:]
        peak: dict[int, float] = {}
        for m in DS_MEM_RE.finditer(text):
            peak[int(m.group(1))] = float(m.group(2))
        peak_str = "/".join(f"{peak[r]:.2f}" for r in sorted(peak)) if peak else ""
        if not step_times:
            return None, None, peak_str, is_oom, reason, ""
        mean = sum(step_times) / len(step_times)
        variance = sum((x - mean) ** 2 for x in step_times) / len(step_times)
        return mean, variance**0.5, peak_str, is_oom, reason, ""

    if system == "megatron":
        # Megatron logs include warmup and earlier steady-state samples; only score the
        # trailing five iteration times to keep the reported stddev representative.
        iter_times_ms = [float(m.group(1)) for m in MG_ITER_RE.finditer(text)][-5:]
        peak: dict[int, float] = {}
        for m in MG_MEM_RE.finditer(text):
            peak[int(m.group(1))] = float(m.group(2)) / 1024.0
        peak_str = "/".join(f"{peak[r]:.2f}" for r in sorted(peak)) if peak else ""
        if not iter_times_ms:
            return None, None, peak_str, is_oom, reason, ""
        iter_times_s = [value / 1000.0 for value in iter_times_ms]
        mean = sum(iter_times_s) / len(iter_times_s)
        variance = sum((x - mean) ** 2 for x in iter_times_s) / len(iter_times_s)
        return mean, variance**0.5, peak_str, is_oom, reason, ""

    if system == "piper":
        if exp is None or args is None or system_out is None:
            return None, None, "", is_oom, "missing_piper_context", ""
        _require_piper_env()
        metrics_dir = system_out / "metrics"
        fetch_dir = system_out / "runner-fetch"
        exp_logs_dir = log_path.parent
        metrics_dir.mkdir(parents=True, exist_ok=True)
        fetch_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{piper_experiment_name(exp, args.nsight)}.metrics"

        cluster_log, _ = _copy_latest_fetch_artifacts(exp, fetch_dir, exp_logs_dir, args.nsight)
        remote_metrics_path = _extract_remote_metrics_path(log_path)
        if remote_metrics_path is None and cluster_log is not None:
            remote_metrics_path = _extract_remote_metrics_path(cluster_log)

        metrics_found = _fetch_remote_metrics(metrics_path, remote_metrics_path)
        if not metrics_found and cluster_log is not None:
            metrics_found = _recover_metrics_from_cluster_log(exp, cluster_log, metrics_path, args.piper_iters)

        remote_output_dir = f"{args.piper_remote_output_dir.rstrip('/')}/{piper_experiment_name(exp, args.nsight)}"
        _fetch_remote_dag_order_logs(exp, remote_output_dir, exp_logs_dir)
        if args.nsight:
            _copy_experiment_nsight_profiles(log_path, exp_logs_dir)

        records = _parse_metrics_records(metrics_path) if metrics_found else []
        iter_mean, iter_std, peak_summary = _aggregate_metrics(records) if records else (None, None, "")
        if records:
            final_reason = "" if returncode == 0 else ("oom" if is_oom else f"exit_code={returncode}")
            return iter_mean, iter_std, peak_summary, is_oom, final_reason, str(metrics_path)
        if is_oom:
            return None, None, "", True, "oom", ""
        if remote_metrics_path is not None:
            # "Benchmark metrics saved to" appeared in the log — the run completed successfully even
            # if we could not fetch or parse the metrics file (e.g. SSH no longer available).
            return iter_mean, iter_std, peak_summary, is_oom, "", ""
        return None, None, "", False, ("missing_metrics" if returncode == 0 else f"exit_code={returncode}"), ""

    return None, None, "", is_oom, reason, ""


def write_csv(path: Path, rows: list[Result]) -> None:
    fieldnames = list(Result.__annotations__.keys())
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
    if ":" in summary or ";" in summary:
        for item in summary.split(";"):
            item = item.strip()
            if not item or ":" not in item:
                continue
            _, value_text = item.split(":", maxsplit=1)
            try:
                values.append(float(value_text))
            except ValueError:
                continue
        return values
    for item in summary.split("/"):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def save_bar_plot(title: str, xlabel: str, rows: list[Result], output_path: Path, *, allow_oom_markers: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [experiment_label_from_result(row) for row in rows]
    xs = list(range(len(rows)))
    success_rows = [row for row in rows if row.status == "ok" and row.iter_time_mean is not None]
    success_max = max((float(row.iter_time_mean) for row in success_rows), default=0.0)
    marker_y = success_max * 0.05 if success_max > 0 else 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), 6))
    for index, row in enumerate(rows):
        if row.status == "ok" and row.iter_time_mean is not None:
            ax.bar(index, float(row.iter_time_mean), yerr=float(row.iter_time_stddev or 0.0), color="#4C72B0", ecolor="#2F2F2F", capsize=4)
        elif allow_oom_markers and row.status == "oom":
            ax.scatter(index, marker_y, marker="x", s=120, color="red", linewidths=2.5)
        else:
            ax.scatter(index, marker_y, marker="x", s=100, color="black", linewidths=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Iteration Time (s)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_memory_plot(title: str, xlabel: str, rows: list[Result], output_path: Path, *, allow_oom_markers: bool = False) -> None:
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
                ax.scatter([index] * len(values), values, color="#DD8452", s=36, alpha=0.9)
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


def load_results_csv(path: Path) -> list[Result]:
    rows: list[Result] = []
    if not path.is_file():
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                Result(
                    system=row["system"],
                    sweep=row["sweep"],
                    config=row["config"],
                    pp=int(row["pp"]),
                    dp=int(row["dp"]),
                    ep=int(row["ep"]),
                    zero_level=row["zero_level"],
                    schedule=row["schedule"],
                    mb_size=int(row["mb_size"]),
                    seq_len=int(row["seq_len"]),
                    local_batch_size=int(row["local_batch_size"]),
                    global_batch_size=int(row["global_batch_size"]),
                    nnode=int(row["nnode"]),
                    ngpu=int(row["ngpu"]),
                    log_path=row["log_path"],
                    metrics_path=row["metrics_path"],
                    returncode=int(row["returncode"]),
                    status=row["status"],
                    iter_time_mean=(float(row["iter_time_mean"]) if row["iter_time_mean"] else None),
                    iter_time_stddev=(float(row["iter_time_stddev"]) if row["iter_time_stddev"] else None),
                    peak_memory_gb_by_rank=row["peak_memory_gb_by_rank"],
                    failure_reason=row["failure_reason"],
                )
            )
    return rows


def save_combined_bar_plot(title: str, xlabel: str, rows: list[Result], output_path: Path, *, allow_oom_markers: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(dict.fromkeys(experiment_label_from_result(row) for row in rows))
    systems = [system for system in DEFAULT_SYSTEMS if any(row.system == system for row in rows)]
    if not labels or not systems:
        return

    row_map = {(row.system, experiment_label_from_result(row)): row for row in rows}
    x_positions = list(range(len(labels)))
    width = 0.8 / max(len(systems), 1)
    success_rows = [row for row in rows if row.status == "ok" and row.iter_time_mean is not None]
    success_max = max((float(row.iter_time_mean) for row in success_rows), default=0.0)
    marker_y = success_max * 0.05 if success_max > 0 else 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 6))
    for system_index, system in enumerate(systems):
        offset = (system_index - (len(systems) - 1) / 2) * width
        for label_index, label in enumerate(labels):
            row = row_map.get((system, label))
            if row is None:
                continue
            x = x_positions[label_index] + offset
            color = SYSTEM_COLORS.get(system, "#4C72B0")
            legend_label = system if label_index == 0 else None
            if row.status == "ok" and row.iter_time_mean is not None:
                ax.bar(
                    x,
                    float(row.iter_time_mean),
                    width=width * 0.9,
                    yerr=float(row.iter_time_stddev or 0.0),
                    color=color,
                    ecolor="#2F2F2F",
                    capsize=4,
                    label=legend_label,
                )
            elif allow_oom_markers and row.status == "oom":
                ax.scatter(x, marker_y, marker="x", s=120, color="red", linewidths=2.5, label=legend_label)
            else:
                ax.scatter(x, marker_y, marker="x", s=100, color=color, linewidths=2.0, label=legend_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Iteration Time (s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    ax.legend(title="System")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_combined_memory_plot(title: str, xlabel: str, rows: list[Result], output_path: Path, *, allow_oom_markers: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(dict.fromkeys(experiment_label_from_result(row) for row in rows))
    systems = [system for system in DEFAULT_SYSTEMS if any(row.system == system for row in rows)]
    if not labels or not systems:
        return

    row_map = {(row.system, experiment_label_from_result(row)): row for row in rows}
    x_positions = list(range(len(labels)))
    width = 0.8 / max(len(systems), 1)
    success_values = [
        value
        for row in rows
        if row.status == "ok"
        for value in parse_peak_memory_values(row.peak_memory_gb_by_rank)
    ]
    marker_y = max(success_values, default=0.0) * 0.05 if success_values else 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 6))
    for system_index, system in enumerate(systems):
        offset = (system_index - (len(systems) - 1) / 2) * width
        color = SYSTEM_COLORS.get(system, "#DD8452")
        for label_index, label in enumerate(labels):
            row = row_map.get((system, label))
            if row is None:
                continue
            x = x_positions[label_index] + offset
            legend_label = system if label_index == 0 else None
            if row.status == "ok":
                values = parse_peak_memory_values(row.peak_memory_gb_by_rank)
                if values:
                    ax.scatter([x] * len(values), values, color=color, s=36, alpha=0.9, label=legend_label)
            elif allow_oom_markers and row.status == "oom":
                ax.scatter(x, marker_y, marker="x", s=120, color="red", linewidths=2.5, label=legend_label)
            else:
                ax.scatter(x, marker_y, marker="x", s=100, color=color, linewidths=2.0, label=legend_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)
    ax.legend(title="System")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def experiment_label_from_result(row: Result) -> str:
    if row.sweep == "scalability":
        return f"(pp={row.pp}, dp={row.dp})"
    if row.sweep == "zero":
        return f"(zero={row.zero_level}, mb={row.mb_size})"
    if row.sweep == "local":
        return "(local)"
    return row.schedule


def write_sweep_outputs(out_dir: Path, sweep: str, rows: list[Result]) -> None:
    if not rows:
        return
    csv_path = out_dir / f"{sweep}.csv"
    write_csv(csv_path, rows)
    if sweep == "scalability":
        title = f"{rows[0].system} Qwen3 DP Scalability"
        subtitle = f"schedule=1f1b, seq_len={rows[0].seq_len}"
    elif sweep == "zero":
        title = f"{rows[0].system} Qwen3 ZeRO Sweep"
        subtitle = f"pp={rows[0].pp}, dp={rows[0].dp}, schedule=1f1b, seq_len={rows[0].seq_len}"
    elif sweep == "local":
        title = f"{rows[0].system} Qwen3 Local Sweep"
        subtitle = f"pp={rows[0].pp}, dp={rows[0].dp}, schedule={rows[0].schedule}, seq_len={rows[0].seq_len}"
    else:
        title = f"{rows[0].system} Qwen3 Schedule Sweep"
        subtitle = f"pp={rows[0].pp}, dp={rows[0].dp}, zero={rows[0].zero_level}, seq_len={rows[0].seq_len}"
    plots_dir = out_dir / "plots"
    save_bar_plot(title, subtitle, rows, plots_dir / f"{sweep}.png", allow_oom_markers=(sweep == "zero"))
    save_memory_plot(f"{title} Memory", subtitle, rows, plots_dir / f"{sweep}_memory.png", allow_oom_markers=(sweep == "zero"))


def write_combined_sweep_outputs(base_out: Path, systems: Iterable[str]) -> None:
    combined_dir = base_out / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[Result] = []
    for system in systems:
        all_rows.extend(load_results_csv(base_out / system / "all_results.csv"))

    for sweep in ("scalability", "zero", "schedule", "local"):
        rows = [row for row in all_rows if row.sweep == sweep]
        if not rows:
            continue
        write_csv(combined_dir / f"{sweep}.csv", rows)
        if sweep == "scalability":
            title = "Qwen3 DP Scalability Across Systems"
            subtitle = f"schedule=1f1b, seq_len={rows[0].seq_len}"
        elif sweep == "zero":
            title = "Qwen3 ZeRO Sweep Across Systems"
            subtitle = f"pp={rows[0].pp}, dp={rows[0].dp}, schedule=1f1b, seq_len={rows[0].seq_len}"
        elif sweep == "local":
            title = "Qwen3 Local Sweep Across Systems"
            subtitle = f"pp={rows[0].pp}, dp={rows[0].dp}, schedule={rows[0].schedule}, seq_len={rows[0].seq_len}"
        else:
            title = "Qwen3 Schedule Sweep Across Systems"
            subtitle = f"pp={rows[0].pp}, dp={rows[0].dp}, zero={rows[0].zero_level}, seq_len={rows[0].seq_len}"
        plots_dir = combined_dir / "plots"
        save_combined_bar_plot(title, subtitle, rows, plots_dir / f"{sweep}.png", allow_oom_markers=(sweep == "zero"))
        save_combined_memory_plot(f"{title} Memory", subtitle, rows, plots_dir / f"{sweep}_memory.png", allow_oom_markers=(sweep == "zero"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen e2e evaluation sweeps across systems.")
    parser.add_argument("--systems", nargs="+", choices=list(DEFAULT_SYSTEMS), default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length override for all sweeps")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: out/e2e-eval/<timestamp>")
    parser.add_argument("--schedule-sweep", nargs="+", default=list(DEFAULT_SCHEDULE_SWEEP))
    parser.add_argument("--sweeps", nargs="+", choices=["scalability", "zero", "schedule", "local"], default=["scalability", "zero", "schedule"])
    parser.add_argument("--nsight", action="store_true", help="Enable Nsight where supported")
    parser.add_argument("--dry-run", action="store_true", help="Plan experiments and output commands without running them")
    parser.add_argument("--bucket-size-mb", type=float, default=None, help="Piper-only bucket size in MB")
    parser.add_argument("--gradient-accumulation", dest="gradient_accumulation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ar-a2a-same-stream", dest="ar_a2a_same_stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlap-chunks", dest="overlap_chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--piper-warmup", type=int, default=3)
    parser.add_argument("--piper-iters", type=int, default=10)
    parser.add_argument("--piper-ray-port", type=int, default=6379)
    parser.add_argument("--piper-remote-output-dir", default="/tmp/piper/out")
    return parser.parse_args()


def piper_model_name(config: str) -> str:
    return {"qwen3_1b": "1B", "qwen3_9b": "9B"}[config]


def piper_schedule_name(schedule: str) -> str:
    return PIPER_SCHEDULES[schedule]


def piper_num_mbs(exp: Experiment) -> int:
    if exp.sweep == "local":
        return 1
    return exp.pp * 2


def piper_experiment_name(exp: Experiment, nsight: bool) -> str:
    bucket_part = f"-bucket{exp.bucket_size_mb:g}" if exp.bucket_size_mb is not None else ""
    nsight_part = "-nsight1" if nsight else ""
    return (
        f"qwen{piper_model_name(exp.config)}-sched_{piper_schedule_name(exp.schedule)}-"
        f"pp{exp.pp}-dp{exp.dp}-ep{exp.ep}-zero{PIPER_ZERO_STAGES[exp.zero_level]}"
        f"{bucket_part}{nsight_part}-bs{exp.mb_size}-sl{exp.seq_len}-mbs{piper_num_mbs(exp)}-"
        f"ga{int(exp.gradient_accumulation)}-aras{int(exp.ar_a2a_same_stream)}-"
        f"och{int(exp.overlap_chunks)}"
    )


def _ssh_head_command(remote_command: str) -> list[str]:
    ssh_key = os.environ["SSH_KEY"]
    head_public_ip = os.environ["HEAD_PUBLIC_IP"]
    return [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        f"ubuntu@{head_public_ip}",
        remote_command,
    ]


def _ssh_worker_command(worker_ip: str, remote_command: str) -> list[str]:
    ssh_key = os.environ["SSH_KEY"]
    head_public_ip = os.environ["HEAD_PUBLIC_IP"]
    return [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        (
            "ProxyCommand="
            f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -W %h:%p ubuntu@{head_public_ip}"
        ),
        f"ubuntu@{worker_ip}",
        remote_command,
    ]


def _terminate_active_child() -> None:
    global _ACTIVE_CHILD
    child = _ACTIVE_CHILD
    if child is None:
        return
    if child.poll() is None:
        try:
            child.terminate()
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        except ProcessLookupError:
            pass
    _ACTIVE_CHILD = None


def _cleanup_remote_launchers() -> None:
    _terminate_active_child()


def _handle_exit_signal(signum: int, _frame) -> None:
    signal_name = signal.Signals(signum).name
    print(f"received {signal_name}, terminating active launcher...", file=sys.stderr)
    _terminate_active_child()
    raise SystemExit(128 + signum)


def _detect_terminal_failure_in_log(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    text = _read_text_lossy(log_path)
    if OOM_RE.search(text):
        return "oom"
    if WATCHDOG_TIMEOUT_RE.search(text):
        return "watchdog_timeout"
    if CHILD_FAILED_RE.search(text):
        return "child_failed"
    return None


def _wait_for_process_or_early_failure(log_path: Path, *, poll_interval_s: float = 5.0) -> tuple[int, str]:
    global _ACTIVE_CHILD
    child = _ACTIVE_CHILD
    if child is None:
        raise RuntimeError("no active child process to monitor")

    while True:
        returncode = child.poll()
        if returncode is not None:
            _ACTIVE_CHILD = None
            return returncode, ""

        early_reason = _detect_terminal_failure_in_log(log_path)
        if early_reason is not None:
            _terminate_active_child()
            returncode = child.returncode if child.returncode is not None else -15
            return returncode, early_reason

        time.sleep(poll_interval_s)


def _scp_from_remote(
    *,
    node_kind: str,
    remote_host: str | None,
    remote_path: str,
    destination: Path,
) -> bool:
    ssh_key = os.environ["SSH_KEY"]
    head_public_ip = os.environ["HEAD_PUBLIC_IP"]
    command = [
        "scp",
        "-r",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if node_kind == "head":
        command.extend([f"ubuntu@{head_public_ip}:{remote_path}", str(destination)])
    else:
        command.extend(
            [
                "-o",
                (
                    "ProxyCommand="
                    f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -W %h:%p ubuntu@{head_public_ip}"
                ),
                f"ubuntu@{remote_host}:{remote_path}",
                str(destination),
            ]
        )
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.returncode == 0


def _stage_remote_nsight_dir(
    *,
    node_kind: str,
    remote_host: str | None,
    remote_tmp_dir: str,
    destination: Path,
) -> bool:
    remote_command = (
        "rm -rf "
        + remote_tmp_dir
        + " && mkdir -p "
        + remote_tmp_dir
        + " && docker cp piper_ray:/tmp/piper/ray_tmp/session_latest/logs/nsight/. "
        + remote_tmp_dir
        + "/"
    )
    run_command = (
        _ssh_head_command(remote_command)
        if node_kind == "head"
        else _ssh_worker_command(str(remote_host), remote_command)
    )
    result = subprocess.run(run_command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _scp_from_remote(
        node_kind=node_kind,
        remote_host=remote_host,
        remote_path=f"{remote_tmp_dir}/.",
        destination=destination,
    )


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _read_text_lossy(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_head_private_ip(log_path: Path) -> str | None:
    for line in _read_text_lossy(log_path).splitlines():
        clean = _strip_ansi(line)
        match = re.search(r"Namespace\(.*address='([^']+)'", clean)
        if match:
            return match.group(1)
    return None


def _node_ip_map_from_env(log_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    head_private_ip = os.environ.get("HEAD_PRIVATE_IP") or _parse_head_private_ip(log_path)
    if head_private_ip:
        mapping[head_private_ip] = "head"
    for idx in range(1, 4):
        worker_ip = os.environ.get(f"WORKER{idx}_PRIVATE_IP")
        if worker_ip:
            mapping[worker_ip] = f"worker{idx}"
    return mapping


def _extract_experiment_pids_by_node(log_path: Path) -> dict[str, set[str]]:
    pid_pattern = re.compile(r"pid=(\d+)(?:, ip=([0-9.]+))?")
    ip_to_label = _node_ip_map_from_env(log_path)
    pids_by_node: dict[str, set[str]] = {}

    for line in _read_text_lossy(log_path).splitlines():
        clean = _strip_ansi(line)
        for pid, ip in pid_pattern.findall(clean):
            node_label = ip_to_label.get(ip, f"node-{ip}") if ip else "head"
            pids_by_node.setdefault(node_label, set()).add(pid)

    return pids_by_node


def _iter_remote_nodes(log_path: Path) -> list[tuple[str, str, str | None]]:
    nodes: list[tuple[str, str, str | None]] = [("head", "head", None)]
    ip_to_label = _node_ip_map_from_env(log_path)
    for ip, label in sorted(ip_to_label.items(), key=lambda item: item[1]):
        if label == "head":
            continue
        nodes.append((label, "worker", ip))
    return nodes


def _matching_nsight_paths(paths: Iterable[str], pids: set[str]) -> list[str]:
    matches: list[str] = []
    for path in paths:
        basename = Path(path).name
        if any(pid in basename or pid in path for pid in pids):
            matches.append(path)
    return sorted(set(matches))


def _copy_experiment_nsight_profiles(log_path: Path, exp_logs_dir: Path) -> list[Path]:
    pids_by_node = _extract_experiment_pids_by_node(log_path)
    if not pids_by_node:
        return []

    copied_paths: list[Path] = []
    exp_nsight_dir = exp_logs_dir / "nsight"
    stage_root = exp_nsight_dir / ".stage"
    shutil.rmtree(stage_root, ignore_errors=True)
    for node_label, node_kind, remote_host in _iter_remote_nodes(log_path):
        node_pids = pids_by_node.get(node_label, set())
        if not node_pids:
            continue
        local_stage_dir = stage_root / node_label
        remote_tmp_dir = f"/tmp/piper-nsight-{exp_logs_dir.name}-{node_label}"
        if not _stage_remote_nsight_dir(
            node_kind=node_kind,
            remote_host=remote_host,
            remote_tmp_dir=remote_tmp_dir,
            destination=local_stage_dir,
        ):
            continue
        staged_paths = [str(path.relative_to(local_stage_dir)) for path in local_stage_dir.rglob("*.nsys-rep")]
        for relative_path in _matching_nsight_paths(staged_paths, node_pids):
            source = local_stage_dir / relative_path
            destination = exp_nsight_dir / node_label / Path(relative_path).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_paths.append(destination)

    pid_manifest = {
        node_label: sorted(pids)
        for node_label, pids in sorted(pids_by_node.items())
        if pids
    }
    if pid_manifest:
        exp_nsight_dir.mkdir(parents=True, exist_ok=True)
        (exp_nsight_dir / "matched_pids.json").write_text(
            json.dumps(pid_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    shutil.rmtree(stage_root, ignore_errors=True)
    return copied_paths


def _parse_metrics_records(metrics_path: Path) -> list[dict]:
    records: list[dict] = []
    if not metrics_path.is_file():
        return records

    marker = "metrics_json="
    for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, maxsplit=1)[1].strip()
        try:
            records.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return records


def _aggregate_metrics(records: list[dict]) -> tuple[float | None, float | None, str]:
    latest_by_dp_rank: dict[int, dict] = {}
    for record in records:
        latest_by_dp_rank[int(record["dp_rank"])] = record

    selected = [latest_by_dp_rank[rank] for rank in sorted(latest_by_dp_rank)]
    weighted_samples = 0
    weighted_iter_time_mean = 0.0
    weighted_iter_time_std = 0.0
    peak_memory_by_rank_summary: list[str] = []

    for record in selected:
        samples = int(record.get("samples", len(record.get("iter_times_s", []))))
        if samples <= 0:
            continue
        iter_time_mean = record.get("iter_time_mean_s")
        if iter_time_mean is None:
            iter_times = [float(iter_time) for iter_time in record.get("iter_times_s", [])]
            if not iter_times:
                continue
            iter_time_mean = sum(iter_times) / len(iter_times)
        iter_time_mean = float(iter_time_mean)
        iter_time_std = record.get("iter_time_std_s")
        if iter_time_std is None:
            iter_times = [float(iter_time) for iter_time in record.get("iter_times_s", [])]
            if iter_times:
                variance = sum((value - iter_time_mean) ** 2 for value in iter_times) / len(iter_times)
                iter_time_std = math.sqrt(variance)
            else:
                iter_time_std = 0.0
        iter_time_std = float(iter_time_std)

        weighted_samples += samples
        weighted_iter_time_mean += samples * iter_time_mean
        weighted_iter_time_std += samples * iter_time_std

        for rank, stats in sorted(record.get("peak_memory_by_rank", {}).items(), key=lambda item: int(item[0])):
            peak_memory_by_rank_summary.append(f"{rank}:{float(stats['peak_memory_gb']):.3f}")

    if weighted_samples == 0:
        return None, None, ";".join(peak_memory_by_rank_summary)
    return (
        weighted_iter_time_mean / weighted_samples,
        weighted_iter_time_std / weighted_samples,
        ";".join(peak_memory_by_rank_summary),
    )


def _fetch_remote_metrics(
    destination: Path,
    remote_metrics_path: str | None,
) -> bool:
    if not remote_metrics_path:
        return False
    remote_command = "docker exec piper_ray bash -lc " + json.dumps(f"test -f {remote_metrics_path} && cat {remote_metrics_path}")
    result = subprocess.run(
        _ssh_head_command(remote_command),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.stdout, encoding="utf-8")
    return True


def _extract_remote_metrics_path(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    metrics_path: str | None = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_ansi(raw_line)
        for pattern in (
            r"benchmark metrics will be saved by the driver to (\S+)",
            r"Benchmark metrics saved to (\S+)",
        ):
            match = re.search(pattern, line)
            if match:
                metrics_path = match.group(1)
    return metrics_path


def _recover_metrics_from_cluster_log(exp: Experiment, cluster_log_path: Path, destination: Path, expected_iters: int) -> bool:
    if not cluster_log_path.is_file():
        return False

    records: list[dict] = []
    marker = "metrics_json="
    for raw_line in cluster_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_ansi(raw_line)
        if marker not in line:
            continue
        payload = line.split(marker, maxsplit=1)[1].strip()
        try:
            records.append(json.loads(payload))
        except json.JSONDecodeError:
            continue

    if records:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(f"rank {record['dp_rank']} metrics_json= {json.dumps(record, sort_keys=True)}\n")
        return True

    dp_rank_by_pid: dict[str, int] = {}
    iter_times_by_pid: dict[str, list[float]] = {}
    peak_memory_by_pid: dict[str, dict[str, dict[str, float | int]]] = {}

    for raw_line in cluster_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_ansi(raw_line)
        dp_rank_match = re.search(r"run_dp_rank pid=(\d+).*DP rank (\d+) done\.", line)
        if dp_rank_match:
            dp_rank_by_pid[dp_rank_match.group(1)] = int(dp_rank_match.group(2))

        step_match = re.search(r"run_dp_rank pid=(\d+).*step_time=([0-9.]+)s", line)
        if step_match:
            pid = step_match.group(1)
            iter_times_by_pid.setdefault(pid, []).append(float(step_match.group(2)))

            rank_memories = {
                rank: {
                    "peak_memory_gb": float(value),
                    "peak_memory_bytes": int(float(value) * (1024 ** 3)),
                }
                for rank, value in re.findall(r"rank(\d+)_peak_mem=([0-9.]+)GB", line)
            }
            if rank_memories:
                peak_memory_by_pid[pid] = rank_memories

    if not iter_times_by_pid:
        return False

    for index, pid in enumerate(sorted(iter_times_by_pid, key=int)):
        dp_rank_by_pid.setdefault(pid, index)

    records = []
    for pid, iter_times in sorted(iter_times_by_pid.items(), key=lambda item: dp_rank_by_pid[item[0]]):
        timed_iter_times = iter_times[-expected_iters:]
        iter_time_mean = sum(timed_iter_times) / len(timed_iter_times)
        iter_time_variance = sum((value - iter_time_mean) ** 2 for value in timed_iter_times) / len(timed_iter_times)
        iter_time_std = math.sqrt(iter_time_variance)
        records.append(
            {
                "dp_rank": dp_rank_by_pid[pid],
                "samples": len(timed_iter_times),
                "iter_time_mean_s": iter_time_mean,
                "iter_time_std_s": iter_time_std,
                "iter_times_s": timed_iter_times,
                "peak_memory_by_rank": peak_memory_by_pid.get(pid, {}),
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"rank {record['dp_rank']} metrics_json= {json.dumps(record, sort_keys=True)}\n")
    return True


def _fetch_remote_dag_order_logs(exp: Experiment, remote_output_dir: str, exp_logs_dir: Path) -> list[Path]:
    copied_paths: list[Path] = []
    for rank in range(exp.pp):
        remote_path = f"{remote_output_dir}/dag_order_rank{rank}"
        result = subprocess.run(
            _ssh_head_command("docker exec piper_ray bash -lc " + json.dumps(f"test -f {remote_path} && cat {remote_path}")),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout:
            continue
        destination = exp_logs_dir / f"dag-order-rank{rank}.log"
        destination.write_text(result.stdout, encoding="utf-8")
        copied_paths.append(destination)
    return copied_paths


def _fetch_megatron_nsight_traces(exp: Experiment, log_path: Path) -> None:
    nsight_dir = log_path.parent / (log_path.stem + "_nsight")
    nsight_dir.mkdir(parents=True, exist_ok=True)
    n_nodes = node_count(exp)
    stage_path = f"/tmp/megatron_nsight_staged_{int(time.time())}"
    nodes: list[tuple[str, str | None]] = [("head", None)]
    for i in range(1, n_nodes):
        worker_ip = os.environ.get(f"WORKER{i}_PRIVATE_IP")
        if worker_ip:
            nodes.append((f"worker{i}", worker_ip))
    for node_label, worker_ip in nodes:
        docker_cp_cmd = f"docker cp torchtitan:/tmp/megatron_nsight {stage_path}"
        ssh_cmd = _ssh_head_command(docker_cp_cmd) if worker_ip is None else _ssh_worker_command(worker_ip, docker_cp_cmd)
        result = subprocess.run(ssh_cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [nsight] no traces on {node_label} (docker cp failed)", file=sys.stderr)
            continue
        local_node_dir = nsight_dir / node_label
        local_node_dir.mkdir(parents=True, exist_ok=True)
        ok = _scp_from_remote(
            node_kind="head" if worker_ip is None else "worker",
            remote_host=worker_ip,
            remote_path=f"{stage_path}/.",
            destination=local_node_dir,
        )
        if ok:
            print(f"  [nsight] fetched traces from {node_label} → {local_node_dir}")
        else:
            print(f"  [nsight] scp failed for {node_label}", file=sys.stderr)
        cleanup_cmd = f"rm -rf {stage_path}"
        cleanup_ssh = _ssh_head_command(cleanup_cmd) if worker_ip is None else _ssh_worker_command(worker_ip, cleanup_cmd)
        subprocess.run(cleanup_ssh, check=False, capture_output=True)


def _require_piper_env() -> None:
    missing = [name for name in ("SSH_KEY", "HEAD_PUBLIC_IP") if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing required environment variables for Piper: " + ", ".join(missing))


def _copy_latest_fetch_artifacts(exp: Experiment, fetch_dir: Path, exp_logs_dir: Path, nsight: bool) -> tuple[Path | None, Path | None]:
    experiment_name = piper_experiment_name(exp, nsight)
    cluster_candidates = sorted(fetch_dir.glob(f"{experiment_name}_*.cluster.log"))
    bundle_candidates = sorted(fetch_dir.glob(f"{experiment_name}_*.ray-logs"))

    cluster_log = None
    if cluster_candidates:
        cluster_log = exp_logs_dir / "cluster.log"
        shutil.copy2(cluster_candidates[-1], cluster_log)

    ray_bundle = None
    if bundle_candidates:
        ray_bundle = exp_logs_dir / "ray-logs"
        shutil.rmtree(ray_bundle, ignore_errors=True)
        shutil.copytree(bundle_candidates[-1], ray_bundle)

    return cluster_log, ray_bundle


def main() -> int:
    signal.signal(signal.SIGINT, _handle_exit_signal)
    signal.signal(signal.SIGTERM, _handle_exit_signal)
    atexit.register(_cleanup_remote_launchers)
    args = parse_args()
    try:
        schedule_sweep = [normalize_schedule(s) for s in args.schedule_sweep]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    base_out = Path(args.out_dir).resolve() if args.out_dir else (Path("out") / "e2e-eval" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(
        systems=args.systems,
        schedule_sweep=schedule_sweep,
        seq_len=args.seq_len,
        enabled_sweeps=set(args.sweeps),
        gradient_accumulation=args.gradient_accumulation,
        ar_a2a_same_stream=args.ar_a2a_same_stream,
        overlap_chunks=args.overlap_chunks,
        bucket_size_mb=args.bucket_size_mb,
    )
    grouped: dict[str, list[Experiment]] = {system: [] for system in args.systems}
    for exp in experiments:
        grouped[exp.system].append(exp)

    print(f"output directory: {base_out}")
    print(f"planned experiments: {len(experiments)}")

    overall_failed = False
    try:
        for system in args.systems:
            system_out = base_out / system
            logs_dir = system_out / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            system_experiments = grouped[system]
            results: list[Result] = []

            plan_path = system_out / "planned_experiments.csv"
            with plan_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["system", "sweep", "config", "pp", "dp", "ep", "zero_level", "schedule", "mb_size", "seq_len", "local_batch_size", "global_batch_size", "nnode", "ngpu", "command"])
                for exp in system_experiments:
                    command = make_command(
                        exp,
                        enable_nsight=args.nsight,
                        fetch_dir=system_out / "runner-fetch",
                        piper_warmup=args.piper_warmup,
                        piper_iters=args.piper_iters,
                        piper_ray_port=args.piper_ray_port,
                        piper_remote_output_dir=args.piper_remote_output_dir,
                    )
                    writer.writerow([
                        exp.system,
                        exp.sweep,
                        exp.config,
                        exp.pp,
                        exp.dp,
                        exp.ep,
                        exp.zero_level,
                        exp.schedule,
                        exp.mb_size,
                        exp.seq_len,
                        local_batch_size(exp),
                        global_batch_size(exp),
                        node_count(exp),
                        exp.pp,
                        " ".join(command),
                    ])

            if args.dry_run:
                for i, exp in enumerate(system_experiments, start=1):
                    print(f"[{system} {i}/{len(system_experiments)}] {exp.sweep}: {experiment_label(exp)} config={exp.config}")
                    command = make_command(
                        exp,
                        enable_nsight=args.nsight,
                        fetch_dir=system_out / "runner-fetch",
                        piper_warmup=args.piper_warmup,
                        piper_iters=args.piper_iters,
                        piper_ray_port=args.piper_ray_port,
                        piper_remote_output_dir=args.piper_remote_output_dir,
                    )
                    print("  " + " ".join(command))
                    print(
                        "  inner="
                        + inner_command(
                            exp,
                            enable_nsight=args.nsight,
                            piper_warmup=args.piper_warmup,
                            piper_iters=args.piper_iters,
                            piper_ray_port=args.piper_ray_port,
                            piper_remote_output_dir=args.piper_remote_output_dir,
                        )
                    )
                continue

            for i, exp in enumerate(system_experiments, start=1):
                log_path = (logs_dir / piper_experiment_name(exp, args.nsight) / "run.log") if system == "piper" else (logs_dir / log_filename(i, exp))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = make_command(
                    exp,
                    enable_nsight=args.nsight,
                    fetch_dir=system_out / "runner-fetch",
                    piper_warmup=args.piper_warmup,
                    piper_iters=args.piper_iters,
                    piper_ray_port=args.piper_ray_port,
                    piper_remote_output_dir=args.piper_remote_output_dir,
                )
                print(f"[{system} {i}/{len(system_experiments)}] running sweep={exp.sweep} label={experiment_label(exp)} config={exp.config}")
                print(f"  log={log_path}")
                print(
                    "  inner="
                    + inner_command(
                        exp,
                        enable_nsight=args.nsight,
                        piper_warmup=args.piper_warmup,
                        piper_iters=args.piper_iters,
                        piper_ray_port=args.piper_ray_port,
                        piper_remote_output_dir=args.piper_remote_output_dir,
                    )
                )
                env = dict(os.environ)
                env.setdefault("TORCHTITAN_PYTHONPATH", DEFAULT_TORCHTITAN_PYTHONPATH)
                with log_path.open("w", encoding="utf-8") as log_file:
                    global _ACTIVE_CHILD
                    _ACTIVE_CHILD = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
                    returncode = _ACTIVE_CHILD.wait()
                    _ACTIVE_CHILD = None
                if args.nsight and system == "megatron":
                    _fetch_megatron_nsight_traces(exp, log_path)
                iter_mean, iter_std, peak_str, is_oom, reason, metrics_path = parse_log(
                    system,
                    log_path,
                    exp=exp,
                    args=args,
                    system_out=system_out,
                    returncode=returncode,
                )
                if returncode == 0 and (iter_mean is not None or system == "megatron"):
                    status = "ok"
                elif is_oom:
                    status = "oom"
                    reason = reason or "oom"
                else:
                    status = "failed"
                    reason = reason or f"exit_code={returncode}"
                run_result = Result(
                    system=system,
                    sweep=exp.sweep,
                    config=exp.config,
                    pp=exp.pp,
                    dp=exp.dp,
                    ep=exp.ep,
                    zero_level=exp.zero_level,
                    schedule=exp.schedule,
                    mb_size=exp.mb_size,
                    seq_len=exp.seq_len,
                    local_batch_size=local_batch_size(exp),
                    global_batch_size=global_batch_size(exp),
                    nnode=node_count(exp),
                    ngpu=exp.pp,
                    log_path=str(log_path),
                    metrics_path=metrics_path,
                    returncode=returncode,
                    status=status,
                    iter_time_mean=iter_mean,
                    iter_time_stddev=iter_std,
                    peak_memory_gb_by_rank=peak_str,
                    failure_reason=reason,
                )
                results.append(run_result)

            write_csv(system_out / "all_results.csv", results)
            for sweep in ("scalability", "zero", "schedule", "local"):
                write_sweep_outputs(system_out, sweep, [r for r in results if r.sweep == sweep])
            if any(r.status != "ok" for r in results):
                overall_failed = True
    finally:
        _terminate_active_child()
        _cleanup_remote_launchers()

    write_combined_sweep_outputs(base_out, args.systems)
    print(f"wrote per-system results under {base_out}")
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
