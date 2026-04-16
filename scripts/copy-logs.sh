#!/usr/bin/env bash
# Copy training logs (and optionally Nsight profiles) from the EC2 cluster.
# Requires: SSH_KEY, HEAD_PUBLIC_IP, WORKERS (space-separated worker private IPs)

set -euo pipefail

NSIGHT=false
OUT_DIR="out/ec2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsight)  NSIGHT=true; shift ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

# Copy logs from head (out/ is volume-mounted, directly accessible on host)
rsync -av \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/out/ \
  "$OUT_DIR/"

if $NSIGHT; then
    # Head nsight profiles (already in OUT_DIR, also collect into nsight-head/ for clarity)
    mkdir -p "$OUT_DIR/nsight-head"
    rsync -av \
      -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
      ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/out/ \
      "$OUT_DIR/nsight-head/"

    # Worker nsight profiles
    WORKER_IDX=1
    for WORKER_IP in $WORKERS; do
        mkdir -p "$OUT_DIR/nsight-worker${WORKER_IDX}"
        rsync -av \
          -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
          ubuntu@$WORKER_IP:/home/ubuntu/torchtitan/out/ \
          "$OUT_DIR/nsight-worker${WORKER_IDX}/"
        WORKER_IDX=$((WORKER_IDX + 1))
    done
fi
