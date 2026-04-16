#!/usr/bin/env bash
# Run qwen training on EC2 cluster via SSH.
#
# Usage:
#   ./scripts/run-qwen-ec2.sh [OPTIONS]
#
# All options can also be set as environment variables.
#
# Options:
#   --nnode N         Number of nodes (default: $NNODE or 2)
#   --ngpu N          GPUs per node (default: $NGPU or 8)
#   --module NAME     torchtitan module (default: $MODULE or qwen3)
#   --config NAME     Config function name (default: $CONFIG or qwen3_9b)
#   --master-port PORT  Rendezvous port (default: $MASTER_PORT or 29500)
#   --nsight          Use run_train_nsys.sh instead of run_train.sh
#   --log-to-file     Tee all output to out/ec2/<config>_<timestamp>.log
#   --out-dir DIR     Directory for log files (default: out/ec2)
#
# Reads from environment: SSH_KEY, HEAD_PUBLIC_IP, HEAD_PRIVATE_IP,
#   WORKER1_PRIVATE_IP, WORKER2_PRIVATE_IP, ... (first NNODE-1 are used)

set -euo pipefail

NNODE="${NNODE:-2}"
NGPU="${NGPU:-8}"
MODULE="${MODULE:-qwen3}"
CONFIG="${CONFIG:-qwen3_9b}"
MASTER_PORT="${MASTER_PORT:-29500}"
NSIGHT=false
LOG_TO_FILE=false
OUT_DIR="out/ec2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnode)        NNODE="$2";        shift 2 ;;
        --ngpu)         NGPU="$2";         shift 2 ;;
        --module)       MODULE="$2";       shift 2 ;;
        --config)       CONFIG="$2";       shift 2 ;;
        --master-port)  MASTER_PORT="$2";  shift 2 ;;
        --nsight)       NSIGHT=true;       shift ;;
        --log-to-file)  LOG_TO_FILE=true;  shift ;;
        --out-dir)      OUT_DIR="$2";      shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Build worker list from WORKER1_PRIVATE_IP, WORKER2_PRIVATE_IP, ... using NNODE-1 workers
WORKERS=""
for i in $(seq 1 $((NNODE - 1))); do
    var="WORKER${i}_PRIVATE_IP"
    ip="${!var:-}"
    if [[ -z "$ip" ]]; then
        echo "Error: \$${var} is not set (needed for NNODE=$NNODE)" >&2; exit 1
    fi
    WORKERS="$WORKERS $ip"
done
WORKERS="${WORKERS# }"  # trim leading space

if $LOG_TO_FILE; then
    mkdir -p "$OUT_DIR"
    LOG_FILE="$OUT_DIR/${CONFIG}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
PROXY="ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP"
TRAIN_SCRIPT="$($NSIGHT && echo './run_train_nsys.sh' || echo './run_train.sh')"

# Create output dirs
$SSH ubuntu@$HEAD_PUBLIC_IP \
  "docker exec torchtitan mkdir -p /workspace/torchtitan/out"

for WORKER_IP in $WORKERS; do
  $SSH -o "$PROXY" ubuntu@$WORKER_IP \
    "docker exec torchtitan mkdir -p /workspace/torchtitan/out"
done

# Launch head
$SSH ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=ens32 \
   -e GLOO_SOCKET_IFNAME=ens32 \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'echo HEAD; pwd; ls -l $TRAIN_SCRIPT; \
     NNODE=$NNODE NGPU=$NGPU LOG_RANK=0 MODULE=$MODULE CONFIG=$CONFIG \
     NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=$MASTER_PORT \
     $TRAIN_SCRIPT'" &

# Launch workers
NODE_RANK=1
for WORKER_IP in $WORKERS; do
  $SSH -o "$PROXY" ubuntu@$WORKER_IP \
    "docker exec -w /workspace/torchtitan \
     -e NCCL_SOCKET_IFNAME=ens32 \
     -e GLOO_SOCKET_IFNAME=ens32 \
     -e NCCL_PROTO=simple \
     torchtitan \
     bash -c 'echo WORKER; pwd; ls -l $TRAIN_SCRIPT; \
       NNODE=$NNODE NGPU=$NGPU LOG_RANK=0 MODULE=$MODULE CONFIG=$CONFIG \
       NODE_RANK=$NODE_RANK MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=$MASTER_PORT \
       $TRAIN_SCRIPT'" &
  NODE_RANK=$((NODE_RANK + 1))
done

wait
