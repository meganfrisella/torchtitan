
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
  --nnode 4 --ngpu 2 \
  --module qwen3 --config qwen3_9b_scalability_pp2_dp4
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
