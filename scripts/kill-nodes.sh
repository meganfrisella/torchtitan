#!/usr/bin/env bash
# Kill python3.10 processes inside the torchtitan container on all cluster nodes
# and free GPU memory. pkill returning non-zero (no matching processes) is fine —
# a clean node is still a clean node.
# Requires: SSH_KEY, HEAD_PUBLIC_IP, WORKER1_PRIVATE_IP, WORKER2_PRIVATE_IP, WORKER3_PRIVATE_IP

set -uo pipefail

: "${SSH_KEY:?}"
: "${HEAD_PUBLIC_IP:?}"

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
PROXY="ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP"

REMOTE_CMD='
docker exec -u 0 torchtitan bash -c "pkill -9 python3.10" || true
sleep 3
echo "=== $(hostname) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -8
'

$SSH ubuntu@"$HEAD_PUBLIC_IP" "$REMOTE_CMD"

for WORKER in "${WORKER1_PRIVATE_IP:-}" "${WORKER2_PRIVATE_IP:-}" "${WORKER3_PRIVATE_IP:-}"; do
  [ -z "$WORKER" ] && continue
  $SSH -o "$PROXY" ubuntu@"$WORKER" "$REMOTE_CMD"
done

REMOTE_CMD='
docker exec -u 0 piper_ray bash -c "pkill -9 python3" || true
sleep 3
echo "=== $(hostname) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -8
'

REMOTE_CMD='
docker exec -u 0 piper_ray bash -c "pkill -9 python"
'

$SSH ubuntu@"$HEAD_PUBLIC_IP" "$REMOTE_CMD"

for WORKER in "${WORKER1_PRIVATE_IP:-}" "${WORKER2_PRIVATE_IP:-}" "${WORKER3_PRIVATE_IP:-}"; do
  [ -z "$WORKER" ] && continue
  $SSH -o "$PROXY" ubuntu@"$WORKER" "$REMOTE_CMD"
done

