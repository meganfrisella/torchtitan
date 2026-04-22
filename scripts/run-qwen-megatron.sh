#!/usr/bin/env bash
set -euo pipefail

NNODE="${NNODE:-2}"
NGPU="${NGPU:-8}"
MODEL="${MODEL:-qwen3_1b}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP="${TP:-1}"
PP="${PP:-1}"
DP="${DP:-1}"
CP="${CP:-1}"
EP="${EP:-1}"
LOG_TO_FILE=false
OUT_DIR="/m-coriander/coriander/mfris/piper-eval"
NSIGHT=false
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnode) NNODE="$2"; shift 2 ;;
        --ngpu) NGPU="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --dp) DP="$2"; shift 2 ;;
        --cp) CP="$2"; shift 2 ;;
        --ep) EP="$2"; shift 2 ;;
        --log-to-file) LOG_TO_FILE=true; shift ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --nsight) NSIGHT=true; shift ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

WORLD_SIZE=$((NNODE * NGPU))
PRODUCT=$((TP * PP * DP * CP * EP))
if (( PRODUCT != WORLD_SIZE )); then
    echo "Error: tp*pp*dp*cp*ep ($PRODUCT) != nnode*ngpu ($WORLD_SIZE)" >&2
    echo "  tp=$TP pp=$PP dp=$DP cp=$CP ep=$EP nnode=$NNODE ngpu=$NGPU" >&2
    exit 1
fi

NSIGHT_FLAG=""
if $NSIGHT; then
    NSIGHT_FLAG="--nsight"
fi

WORKERS=""
for i in $(seq 1 $((NNODE - 1))); do
    var="WORKER${i}_PRIVATE_IP"
    ip="${!var:-}"
    if [[ -z "$ip" ]]; then
        echo "Error: \$${var} is not set (needed for NNODE=$NNODE)" >&2
        exit 1
    fi
    WORKERS="$WORKERS $ip"
done
WORKERS="${WORKERS# }"

if $LOG_TO_FILE; then
    mkdir -p "$OUT_DIR"
    LOG_FILE="$OUT_DIR/${MODEL}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
PROXY="ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP"
MEGATRON_WORKSPACE="${MEGATRON_WORKSPACE:-/workspace/Megatron-LM}"
TORCHTITAN_WORKSPACE="${TORCHTITAN_WORKSPACE:-/workspace/torchtitan}"

TRAIN_ARGS_QUOTED=""
if ((${#TRAIN_ARGS[@]})); then
    TRAIN_ARGS_QUOTED=$(printf '%q ' "${TRAIN_ARGS[@]}")
fi

PARALLEL_ARGS="--tp $TP --pp $PP --dp $DP --cp $CP --ep $EP"

LOWEST_LEVEL_CMD="conda run -n megatron python $TORCHTITAN_WORKSPACE/run_megatron.py --model $MODEL --nnodes $NNODE --nproc-per-node $NGPU --master-addr $HEAD_PRIVATE_IP --master-port $MASTER_PORT --disable-background-mode $PARALLEL_ARGS $NSIGHT_FLAG ${TRAIN_ARGS_QUOTED}"
echo "Lowest-level command: $LOWEST_LEVEL_CMD"

$SSH ubuntu@$HEAD_PUBLIC_IP "docker exec torchtitan mkdir -p /m-coriander/coriander/mfris/piper-eval"
for WORKER_IP in $WORKERS; do
    $SSH -o "$PROXY" ubuntu@$WORKER_IP "docker exec torchtitan mkdir -p /m-coriander/coriander/mfris/piper-eval"
done

$SSH ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w $MEGATRON_WORKSPACE \
    -e NCCL_SOCKET_IFNAME=ens32 \
    -e GLOO_SOCKET_IFNAME=ens32 \
    torchtitan \
    bash -lc 'echo HEAD; pwd; \
      NODE_RANK=0 \
      conda run -n megatron python $TORCHTITAN_WORKSPACE/run_megatron.py \
        --model $MODEL \
        --nnodes $NNODE \
        --nproc-per-node $NGPU \
        --master-addr $HEAD_PRIVATE_IP \
        --master-port $MASTER_PORT \
        --disable-background-mode \
        $PARALLEL_ARGS \
        $NSIGHT_FLAG \
        $TRAIN_ARGS_QUOTED'" &

NODE_RANK=1
for WORKER_IP in $WORKERS; do
  $SSH -o "$PROXY" ubuntu@$WORKER_IP \
    "docker exec -w $MEGATRON_WORKSPACE \
      -e NCCL_SOCKET_IFNAME=ens32 \
      -e GLOO_SOCKET_IFNAME=ens32 \
      torchtitan \
      bash -lc 'echo WORKER; pwd; \
        NODE_RANK=$NODE_RANK \
        conda run -n megatron python $TORCHTITAN_WORKSPACE/run_megatron.py \
          --model $MODEL \
          --nnodes $NNODE \
          --nproc-per-node $NGPU \
          --master-addr $HEAD_PRIVATE_IP \
          --master-port $MASTER_PORT \
          --disable-background-mode \
          $PARALLEL_ARGS \
          $NSIGHT_FLAG \
          $TRAIN_ARGS_QUOTED'" &
  NODE_RANK=$((NODE_RANK + 1))
done

wait
