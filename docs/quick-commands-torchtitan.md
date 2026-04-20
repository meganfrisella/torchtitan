
```bash
export PATH_TO_TORCHTITAN=/m-coriander/coriander/mfris/torchtitan
export IMAGE=mfris/torchtitan-70b:latest
export SSH_KEY=~/.ssh/ray-autoscaler_us-east-2.pem
export HEAD_PUBLIC_IP=xxx
export HEAD_PRIVATE_IP=xxx
export WORKER1_PRIVATE_IP=xxx
export WORKER2_PRIVATE_IP=xxx
export WORKER3_PRIVATE_IP=xxx
```

# Re-sync code

```bash
./scripts/resync.sh
```


# Run qwen

```bash
./scripts/run-qwen-ec2.sh \
  --nnode 2 --ngpu 8 \
  --module qwen3 --config qwen3_9b -- \
  --parallelism.pipeline_parallel_degree 8 \
  --parallelism.expert_parallel_degree 1 \
  --parallelism.pipeline_parallel_schedule DualPipeV \
  --parallelism.pipeline_parallel_microbatch_size 8 \
  --parallelism.data_parallel_replicate_degree 2 \
  --parallelism.data_parallel_shard_degree 1 \
  --training.seq_len 512 \
  --training.local_batch_size 128 \
  --training.global_batch_size 256
```

Add `--log-to-file` to tee output to `out/ec2/<config>_<timestamp>.log`.
Add `--nsight` to enable Nsight profiling via `run_train_nsys.sh`.

# Kill hung torchrun

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec torchtitan pkill -9 -f nsys; docker exec torchtitan pkill -9 -f torchrun; docker exec torchtitan pkill -9 -f run_train.sh" 2>/dev/null

for WORKER_IP in $WORKERS; do
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP \
    "docker exec torchtitan pkill -9 -f nsys; docker exec torchtitan pkill -9 -f torchrun; docker exec torchtitan pkill -9 -f run_train.sh" 2>/dev/null &
done
wait
```

# Copy outputs to local

```bash
./scripts/copy-logs.sh
```

Add `--nsight` to also copy Nsight profiles into `out/ec2/nsight-head` and `out/ec2/nsight-workerX`.


# Kill stuff

```bash
 for HOST in "$HEAD_PUBLIC_IP" "$WORKER1_PRIVATE_IP" "$WORKER2_PRIVATE_IP" "$WORKER3_PRIVATE_IP"; do
    [ -z "$HOST" ] && continue

    if [ "$HOST" = "$HEAD_PUBLIC_IP" ]; then
      SSH_CMD=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@"$HOST")
    else
      SSH_CMD=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP"
  \
        ubuntu@"$HOST")
    fi

    "${SSH_CMD[@]}" '
      echo "=== HOST $(hostname) ==="

      for port in 29500 29501 29600; do
        pids=$(sudo ss -ltnp "( sport = :$port )" | sed -n "s/.*pid=\([0-9]\+\).*/\1/p" | sort -u)
        for pid in $pids; do
          sudo kill -9 "$pid" || true
        done
      done

      sudo pkill -9 -f pt_elastic || true
      sudo pkill -9 -f torchrun || true
      sudo pkill -9 -f run_megatron.py || true
      sudo pkill -9 -f pretrain_gpt.py || true
      sudo pkill -9 -f megatron || true
      sudo pkill -9 -f run_deepspeed.py || true
      sudo pkill -9 -f torchtitan.train || true
      sudo pkill -9 -f run_train.sh || true
      sudo pkill -9 -f run_train_nsys.sh || true

      gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk
  "NF" | sort -u)
      for pid in $gpu_pids; do
        sudo kill -9 "$pid" || true
      done

      docker exec torchtitan bash -lc "
        pkill -9 -f pt_elastic || true
        pkill -9 -f torchrun || true
        pkill -9 -f run_megatron.py || true
        pkill -9 -f pretrain_gpt.py || true
      echo
      nvidia-smi
    '
  done
```