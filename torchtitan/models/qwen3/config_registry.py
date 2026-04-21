# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    CompileConfig,
    TrainingConfig,
)
from torchtitan.config.configs import CommConfig
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from . import model_registry


def qwen3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_debugmodel_flex() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_flex"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_0_6b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("0.6B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_1_7b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("1.7B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
        ),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_14b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-14B",
        model_spec=model_registry("14B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )


def qwen3_32b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-32B",
        model_spec=model_registry("32B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
    )


def qwen3_moe_debug() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_moe"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            global_batch_size=32,  # Should be divisible by pipeline_parallel_microbatch_size
            local_batch_size=16,  # Should be divisible by pipeline_parallel_microbatch_size
            seq_len=512,
            steps=8,  # Few iterations for profiling
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=2,
            expert_tensor_parallel_degree=1,
            pipeline_parallel_degree=2,
            pipeline_parallel_microbatch_size=4,  # Configurable micro batch size
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_1b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        model_spec=model_registry("1B-A0.7B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            global_batch_size=128,  # Should be divisible by pipeline_parallel_microbatch_size
            local_batch_size=64,  # Should be divisible by pipeline_parallel_microbatch_size
            seq_len=512,
            steps=8,  # Few iterations for profiling
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=2,
            expert_tensor_parallel_degree=1,
            pipeline_parallel_degree=4,
            pipeline_parallel_microbatch_size=8,  # Configurable micro batch size
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=CompileConfig(enable=True),
    )


def qwen3_9b_single() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-8B",
        model_spec=model_registry("9B-A3B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=128,  # Should be divisible by pipeline_parallel_microbatch_size
            seq_len=512,
            steps=8,  # Few iterations for profiling
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=8,
            pipeline_parallel_microbatch_size=8,  # Configurable micro batch size
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=CompileConfig(enable=True),
    )

def qwen3_9b() -> Trainer.Config:
    """Generic Qwen3 9B config for runtime PP/DP/EP/schedule/batch/ZeRO overrides."""
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-8B",
        model_spec=model_registry("9B-A3B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            global_batch_size=64,
            local_batch_size=64,
            seq_len=512,
            steps=8,
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            pipeline_parallel_degree=8,
            pipeline_parallel_schedule="1F1B",
            pipeline_parallel_microbatch_size=4,
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            fsdp_reshard_after_forward="default",
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        compile=CompileConfig(enable=True),
    )


def qwen3_30b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("30B-A3B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            global_batch_size=256,  # Should be divisible by pipeline_parallel_microbatch_size
            local_batch_size=128,  # Should be divisible by pipeline_parallel_microbatch_size
            seq_len=1024,
            steps=8,  # Few iterations for profiling
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=2,
            expert_tensor_parallel_degree=1,
            pipeline_parallel_degree=8,
            pipeline_parallel_microbatch_size=8,  # Configurable micro batch size
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        compile=CompileConfig(enable=True),
    )


def qwen3_30b_single() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("30B-A3B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=16,  # Should be divisible by pipeline_parallel_microbatch_size
            seq_len=256,
            steps=8,  # Few iterations for profiling
            dtype="bfloat16",
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
        # compile=CompileConfig(enable=True),
    )

def qwen3_30b_pp8_ep4_dualpipe() -> Trainer.Config:
    """DualPipeV PP=8 (intra-node) + EP=4 (cross-node) on Qwen3-MoE 30B-A3B.

    Requires 4 nodes × 8 GPUs = 32 ranks. Dense product PP=8, so dp_shard is
    auto-derived to 32/8 = 4 (FSDP across the batch dim).

    Mesh layout with enable_ep_outer=True (EP outermost, PP inner):
        sparse: (ep=4, dp_replicate=1, efsdp=1, pp=8, etp=1)
        dense:  (dp_replicate=1, fsdp=4, pp=8, tp=1)
      EP peers: {0,8,16,24}, {1,9,17,25}, ... → one rank per node, cross-node.
      PP stages: ranks 0-7 / 8-15 / 16-23 / 24-31 → 8 stages intra-node.

    Batch math:
        global_batch = local_batch_size × dp_world = 32 × 4 = 128
        num_microbatches = local_batch / micro_batch = 32 / 2 = 16
        DualPipeV stages = PP × 2 = 16  →  16 ≥ 16 ✓
    """
    return Trainer.Config(
        dump_folder="./e2e-benchmark/outputs/qwen3_30b_pp8_ep4",
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("30B-A3B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=1e-4,
            implementation="foreach",
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=5),
        training=TrainingConfig(
            local_batch_size=32,
            seq_len=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            gc_freq=50,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=8,
            expert_parallel_degree=4,
            pipeline_parallel_schedule="DualPipeV",
            pipeline_parallel_microbatch_size=2,
            pipeline_parallel_expert_parallel_overlap=False,
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            enable_ep_outer=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        profiling=ProfilingConfig(
            enable_profiling=True,
            save_traces_folder="e2e-benchmark/profile_traces",
            profile_freq=10,
            profiler_warmup=3,
            profiler_active=1,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        comm=CommConfig(
            init_timeout_seconds=600,
            train_timeout_seconds=600,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
    )


def qwen3_9b_pp8_ep4_dualpipe() -> Trainer.Config:
    """DualPipeV PP=8 + EP=4 on Qwen3-MoE 9B-A3B. Same shape as the 30B variant.

    9B has 8 experts, so EP=4 places 2 experts per rank.
    """
    cfg = qwen3_30b_pp8_ep4_dualpipe()
    cfg.dump_folder = "./e2e-benchmark/outputs/qwen3_9b_pp8_ep4"
    cfg.model_spec = model_registry("9B-A3B")
    return cfg


def qwen3_1b_pp8_ep4_dualpipe() -> Trainer.Config:
    """DualPipeV PP=8 + EP=4 on Qwen3-MoE 1B-A0.7B. Same shape as the 30B variant.

    1B has 4 experts and only 16 layers — one layer per virtual PP stage (PP*2=16).
    """
    cfg = qwen3_30b_pp8_ep4_dualpipe()
    cfg.dump_folder = "./e2e-benchmark/outputs/qwen3_1b_pp8_ep4"
    cfg.model_spec = model_registry("1B-A0.7B")
    return cfg


def sft_qwen3_8b_math() -> Trainer.Config:
    """Qwen3-8B SFT on GSM8K math dataset."""

    def process_sample(sample):
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]

    model_spec = model_registry("8B", attn_backend_override="varlen")
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-8B",
        model_spec=model_spec,
        optimizer=OptimizersContainer.Config(lr=2e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=15,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=2048,
            steps=180,
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="openai/gsm8k",
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )
