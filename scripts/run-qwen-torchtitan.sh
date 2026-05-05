#!/usr/bin/env bash
# Run TorchTitan Qwen training on EC2 cluster via SSH.
#
# Usage:
#   ./scripts/run-qwen-torchtitan.sh [OPTIONS]
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
#   --log-rank RANKS  LOG_RANK value for run_train.sh (default: $LOG_RANK or 0)
#   --log-to-file     Tee all output to out/ec2/<config>_<timestamp>.log
#   --out-dir DIR     Directory for log files (default: out/ec2)
#   --local           Run on local machine without SSH (for testing)
#   --                Forward remaining args to torchtitan.train inside the container
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
LOCAL=false
LOG_RANK="${LOG_RANK:-0}"
LOG_TO_FILE=false
OUT_DIR="/m-coriander/coriander/mfris/piper-eval"
TORCHTITAN_PYTHONPATH="${TORCHTITAN_PYTHONPATH:-/workspace/torchtitan}"
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnode)        NNODE="$2";        shift 2 ;;
        --ngpu)         NGPU="$2";         shift 2 ;;
        --module)       MODULE="$2";       shift 2 ;;
        --config)       CONFIG="$2";       shift 2 ;;
        --master-port)  MASTER_PORT="$2";  shift 2 ;;
        --nsight)       NSIGHT=true;       shift ;;
        --local)        LOCAL=true;        shift ;;
        --log-rank)     LOG_RANK="$2";     shift 2 ;;
        --log-to-file)  LOG_TO_FILE=false;  shift ;;
        --out-dir)      OUT_DIR="$2";      shift 2 ;;
        --)             shift; TRAIN_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if $LOG_TO_FILE; then
    mkdir -p "$OUT_DIR"
    LOG_FILE="$OUT_DIR/${CONFIG}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

TRAIN_SCRIPT="$($NSIGHT && echo './run_train_nsys.sh' || echo './run_train.sh')"

TRAIN_ARGS_QUOTED=""
if ((${#TRAIN_ARGS[@]})); then
    TRAIN_ARGS_QUOTED=$(printf '%q ' "${TRAIN_ARGS[@]}")
fi

MASTER_ADDR_VALUE="${HEAD_PRIVATE_IP:-127.0.0.1}"
LOWEST_LEVEL_CMD="conda run -n torchtitan env PYTHONPATH=$TORCHTITAN_PYTHONPATH NNODE=$NNODE NGPU=$NGPU LOG_RANK=$LOG_RANK MODULE=$MODULE CONFIG=$CONFIG NODE_RANK=<node_rank> MASTER_ADDR=$MASTER_ADDR_VALUE MASTER_PORT=$MASTER_PORT $TRAIN_SCRIPT ${TRAIN_ARGS_QUOTED}"
echo "Lowest-level command: $LOWEST_LEVEL_CMD"

if $LOCAL; then
  if [[ "$NNODE" != "1" ]]; then
    echo "Error: --local mode only supports --nnode 1 (got $NNODE)" >&2
    exit 1
  fi

  LOCAL_CMD=(
    conda run -n torchtitan env
    "PYTHONPATH=$TORCHTITAN_PYTHONPATH"
    "NNODE=$NNODE"
    "NGPU=$NGPU"
    "LOG_RANK=$LOG_RANK"
    "MODULE=$MODULE"
    "CONFIG=$CONFIG"
    "NODE_RANK=0"
    "$TRAIN_SCRIPT"
  )
  if ((${#TRAIN_ARGS[@]})); then
    LOCAL_CMD+=("${TRAIN_ARGS[@]}")
  fi

  printf 'Running local command: '
  printf '%q ' "${LOCAL_CMD[@]}"
  printf '\n'
  "${LOCAL_CMD[@]}"
  exit 0
fi

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

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
PROXY="ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP"

# Create output dirs
$SSH ubuntu@$HEAD_PUBLIC_IP \
  "docker exec torchtitan mkdir -p /m-coriander/coriander/mfris/piper-eval"

for WORKER_IP in $WORKERS; do
  $SSH -o "$PROXY" ubuntu@$WORKER_IP \
    "docker exec torchtitan mkdir -p /m-coriander/coriander/mfris/piper-eval"
done

# Launch head
$SSH ubuntu@$HEAD_PUBLIC_IP   "docker exec -w /workspace/torchtitan    -e NCCL_SOCKET_IFNAME=ens32    -e GLOO_SOCKET_IFNAME=ens32    -e PYTHONPATH=$TORCHTITAN_PYTHONPATH    torchtitan    bash -lc 'echo HEAD; pwd; ls -l $TRAIN_SCRIPT;      conda run -n torchtitan env PYTHONPATH=$TORCHTITAN_PYTHONPATH      NNODE=$NNODE NGPU=$NGPU LOG_RANK=$LOG_RANK MODULE=$MODULE CONFIG=$CONFIG      NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=$MASTER_PORT      $TRAIN_SCRIPT $TRAIN_ARGS_QUOTED'" &
# Launch workers
NODE_RANK=1
for WORKER_IP in $WORKERS; do
  $SSH -o "$PROXY" ubuntu@$WORKER_IP     "docker exec -w /workspace/torchtitan      -e NCCL_SOCKET_IFNAME=ens32      -e GLOO_SOCKET_IFNAME=ens32      -e PYTHONPATH=$TORCHTITAN_PYTHONPATH      torchtitan      bash -lc 'echo WORKER; pwd; ls -l $TRAIN_SCRIPT;        conda run -n torchtitan env PYTHONPATH=$TORCHTITAN_PYTHONPATH        NNODE=$NNODE NGPU=$NGPU LOG_RANK=$LOG_RANK MODULE=$MODULE CONFIG=$CONFIG        NODE_RANK=$NODE_RANK MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=$MASTER_PORT        $TRAIN_SCRIPT $TRAIN_ARGS_QUOTED'" &
  NODE_RANK=$((NODE_RANK + 1))
done

wait
