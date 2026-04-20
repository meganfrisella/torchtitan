#!/usr/bin/env bash
set -euo pipefail

NNODE="${NNODE:-2}"
NGPU="${NGPU:-8}"
MODEL="${MODEL:-qwen3_1b}"
LOG_TO_FILE=false
OUT_DIR="/m-coriander/coriander/mfris/piper-eval"
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnode) NNODE="$2"; shift 2 ;;
        --ngpu) NGPU="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --log-to-file) LOG_TO_FILE=true; shift ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if $LOG_TO_FILE; then
    mkdir -p "$OUT_DIR"
    LOG_FILE="$OUT_DIR/${MODEL}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

PIPER_MODEL="$MODEL"
case "$MODEL" in
    qwen3_1b) PIPER_MODEL="1B" ;;
    qwen3_9b) PIPER_MODEL="9B" ;;
esac

ARGS=(--model "$PIPER_MODEL" --pp "$NGPU" --dp "$NNODE")
if ((${#TRAIN_ARGS[@]})); then
    ARGS+=("${TRAIN_ARGS[@]}")
fi

exec /m-coriander/coriander/mfris/piper/scripts/piper-run-qwen-ec2.sh "${ARGS[@]}"
