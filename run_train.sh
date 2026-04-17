#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh

NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-"0"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
TRAIN_ARGS_B64=${TRAIN_ARGS_B64:-""}
EXTRA_ARGS=()
if [ -n "$TRAIN_ARGS_B64" ]; then
    TRAIN_ARGS_DECODED=$(printf %s "$TRAIN_ARGS_B64" | base64 -d)
    if [ -n "$TRAIN_ARGS_DECODED" ]; then
        eval "EXTRA_ARGS=($TRAIN_ARGS_DECODED)"
    fi
fi

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@" "${EXTRA_ARGS[@]}" --comm.mode=${COMM_MODE} --training.steps 1
else
    # Normal training with torchrun
    # PYTORCH_ALLOC_CONF="expandable_segments:True" \
    # TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    # torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    # --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    # -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@" "${EXTRA_ARGS[@]}"

    # Multi-node training without nsys
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nnodes=${NNODE} --nproc_per_node=${NGPU} \
    --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@" "${EXTRA_ARGS[@]}"
fi
