#!/usr/bin/env bash
# Resync local torchtitan source to all cluster nodes.
# Requires: SSH_KEY, HEAD_PUBLIC_IP, WORKER1_PRIVATE_IP, WORKER2_PRIVATE_IP, WORKER3_PRIVATE_IP
#           PATH_TO_TORCHTITAN (defaults to repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATH_TO_TORCHTITAN="${PATH_TO_TORCHTITAN:-"$(dirname "$SCRIPT_DIR")"}"

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
RSYNC_EXCLUDES='--exclude=.git --exclude=__pycache__ --exclude=.venv --exclude=out --exclude=ec2-out'

# Head node
$SSH ubuntu@$HEAD_PUBLIC_IP \
  "sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

rsync -av --delete $RSYNC_EXCLUDES \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "$PATH_TO_TORCHTITAN/" ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/

# Worker nodes
for WORKER_IP in ${WORKER1_PRIVATE_IP:-} ${WORKER2_PRIVATE_IP:-} ${WORKER3_PRIVATE_IP:-}; do
  [[ -z "$WORKER_IP" ]] && continue

  $SSH \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP \
    "sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

  rsync -av --delete $RSYNC_EXCLUDES \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
    "$PATH_TO_TORCHTITAN/" ubuntu@$WORKER_IP:/home/ubuntu/torchtitan/
done

# Create output dirs in containers
$SSH ubuntu@$HEAD_PUBLIC_IP \
  "docker exec torchtitan mkdir -p /workspace/torchtitan/out"

for WORKER_IP in ${WORKER1_PRIVATE_IP:-} ${WORKER2_PRIVATE_IP:-} ${WORKER3_PRIVATE_IP:-}; do
  [[ -z "$WORKER_IP" ]] && continue

  $SSH \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP \
    "docker exec torchtitan mkdir -p /workspace/torchtitan/out"
done
