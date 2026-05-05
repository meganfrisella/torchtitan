  for i in 0 1 2 3; do
    if [ "$i" -eq 0 ]; then
      HOST="$HEAD_PUBLIC_IP"
      SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no)
    else
      HOST_VAR="WORKER${i}_PRIVATE_IP"
      HOST="${!HOST_VAR}"
      SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP")
    fi
    echo "===== node$i ($HOST) ====="
    ssh "${SSH_OPTS[@]}" ubuntu@"$HOST" "nvidia-smi" || echo "node$i failed"
  done


  for i in 0 1 2 3; do
    if [ "$i" -eq 0 ]; then
      HOST="$HEAD_PUBLIC_IP"
      SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no)
    else
      HOST_VAR="WORKER${i}_PRIVATE_IP"
      HOST="${!HOST_VAR}"
      SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP")
    fi
    echo "===== node$i ($HOST) ====="
    ssh "${SSH_OPTS[@]}" ubuntu@"$HOST" "pkill -9 -u ubuntu || true; docker ps -q | xargs -r -n1 docker kill || true"
  done





# EC2 Cluster Commands

## Copy Nsight traces from container to local experiment directory

```bash
LOCAL_NSIGHT_DIR="out/e2e-eval/20260421_052204/megatron/logs/01_local__pp1_dp1_ep1__zero1__1f1b__mb4__sl512_20260421_052204_nsight"

# Head node
mkdir -p "$LOCAL_NSIGHT_DIR/head"
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker cp torchtitan:/tmp/megatron_nsight /tmp/megatron_nsight_staged" && \
scp -r -i $SSH_KEY -o StrictHostKeyChecking=no \
  ubuntu@$HEAD_PUBLIC_IP:/tmp/megatron_nsight_staged/. \
  "$LOCAL_NSIGHT_DIR/head/" && \
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "rm -rf /tmp/megatron_nsight_staged"

# Worker nodes (repeat for i in 1 2 3 4)
for i in 1 2 3 4; do
  WORKER_IP_VAR="WORKER${i}_PRIVATE_IP"
  mkdir -p "$LOCAL_NSIGHT_DIR/worker${i}"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} \
    "docker cp torchtitan:/tmp/megatron_nsight /tmp/megatron_nsight_staged" && \
  scp -r -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR}:/tmp/megatron_nsight_staged/. \
    "$LOCAL_NSIGHT_DIR/worker${i}/" && \
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} \
    "rm -rf /tmp/megatron_nsight_staged"
done
```

## List running processes on all nodes

```bash
LIST_CMD="echo '=== '\$(hostname)' ==='; docker exec torchtitan ps aux --no-headers | grep -v '^root.*ps aux'"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "$LIST_CMD" &
for i in 1 2 3 4; do
  WORKER_IP_VAR="WORKER${i}_PRIVATE_IP"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} "$LIST_CMD" &
done
wait
```

## Install Apex FusedAdam on all nodes

```bash
APEX_CMD="docker exec torchtitan conda run -n megatron pip install -v \
  --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings '--build-option=--cpp_ext' \
  --config-settings '--build-option=--cuda_ext' \
  git+https://github.com/NVIDIA/apex.git"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "$APEX_CMD" &
for i in 1 2 3 4; do
  WORKER_IP_VAR="WORKER${i}_PRIVATE_IP"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} "$APEX_CMD" &
done
wait
```

Verify:

```bash
VERIFY_CMD="docker exec torchtitan conda run -n megatron python -c 'from apex.optimizers import FusedAdam; print(\"FusedAdam ok\")'"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "$VERIFY_CMD" &
for i in 1 2 3 4; do
  WORKER_IP_VAR="WORKER${i}_PRIVATE_IP"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} "$VERIFY_CMD" &
done
wait
```

## Kill hung Megatron processes on all nodes

```bash
KILL_CMD="docker exec torchtitan pkill -9 -f 'pretrain_gpt.py|run_megatron.py'"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "$KILL_CMD" &
for i in 1 2 3 4; do
  WORKER_IP_VAR="WORKER${i}_PRIVATE_IP"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} "$KILL_CMD" &
done
wait
```

## Free port 29500 on all nodes

```bash
FREE_PORT_CMD="docker exec torchtitan sh -c 'lsof -ti:29500 | xargs -r kill -9'"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "$FREE_PORT_CMD" &
for i in 1 2 3 4; do
  WORKER_IP_VAR="WORKER${i}_PRIVATE_IP"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} "$FREE_PORT_CMD" &
done
wait
```

Check port is free (no output = free):

```bash
CHECK_PORT_CMD="docker exec torchtitan grep -i '7394' /proc/net/tcp"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "$CHECK_PORT_CMD" &
for i in 1 2 3 4; do
  WORKER_IP_VAR="WORKER${i}_PRIVATE_IP"
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@${!WORKER_IP_VAR} "$CHECK_PORT_CMD" &
done
wait
```
