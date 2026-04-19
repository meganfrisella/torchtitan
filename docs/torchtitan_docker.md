# Variables

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

# Start docker

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out' --exclude 'ec2-out' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $PATH_TO_TORCHTITAN/ ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "
  docker stop torchtitan 2>/dev/null || true
  docker rm   torchtitan 2>/dev/null || true
  docker run -d --name torchtitan \
    --gpus all --ipc=host --shm-size=64g \
    --ulimit nofile=65536:65536 --ulimit memlock=-1:-1 --privileged \
    --device /dev/infiniband --network host \
    -v /home/ubuntu/torchtitan:/workspace/torchtitan \
    -v /opt/amazon/efa:/opt/amazon/efa:ro \
    -v /opt/amazon/ofi-nccl:/opt/aws-ofi-nccl:ro \
    -v /usr/lib/x86_64-linux-gnu:/opt/host-lib:ro \
    -v /home/ubuntu/torchtitan/assets/hf/:/workspace/assets/hf/:ro \
    -v /home/ubuntu/torchtitan/tests/assets:/workspace/tests/assets \
    -e LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:/opt/host-lib \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    -e RDMAV_FORK_SAFE=1 \
    -e FI_EFA_FORK_SAFE=1 \
    -e NCCL_SOCKET_IFNAME=ens32 \
    -e GLOO_SOCKET_IFNAME=ens32 \
    $IMAGE sleep infinity
"
for WORKER_IP in $WORKER1_PRIVATE_IP $WORKER2_PRIVATE_IP $WORKER3_PRIVATE_IP; do
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP \
    "sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

  rsync -av --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out' --exclude 'ec2-out' --exclude='.claude' --exclude='.github' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
    $PATH_TO_TORCHTITAN/ ubuntu@$WORKER_IP:/home/ubuntu/torchtitan/

  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP "
    docker stop torchtitan 2>/dev/null || true
    docker rm   torchtitan 2>/dev/null || true
    docker run -d --name torchtitan \
      --gpus all --ipc=host --shm-size=64g \
      --ulimit nofile=65536:65536 --ulimit memlock=-1:-1 --privileged \
      --device /dev/infiniband --network host \
      -v /home/ubuntu/torchtitan:/workspace/torchtitan \
      -v /opt/amazon/efa:/opt/amazon/efa:ro \
      -v /opt/amazon/ofi-nccl:/opt/aws-ofi-nccl:ro \
      -v /usr/lib/x86_64-linux-gnu:/opt/host-lib:ro \
      -v /home/ubuntu/torchtitan/assets/hf/:/workspace/assets/hf/:ro \
      -v /home/ubuntu/torchtitan/tests/assets:/workspace/tests/assets \
      -e LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:/opt/host-lib \
      -e FI_PROVIDER=efa \
      -e FI_EFA_USE_DEVICE_RDMA=1 \
      -e RDMAV_FORK_SAFE=1 \
      -e FI_EFA_FORK_SAFE=1 \
      -e NCCL_SOCKET_IFNAME=ens32 \
      -e GLOO_SOCKET_IFNAME=ens32 \
      $IMAGE sleep infinity
  "
done
```

# Patch fsspec version incompatibility in torchtitan containers

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec torchtitan pip install --upgrade fsspec"

for WORKER_IP in $WORKER1_PRIVATE_IP $WORKER2_PRIVATE_IP $WORKER3_PRIVATE_IP; do
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP \
    "docker exec torchtitan pip install --upgrade fsspec" &
done
wait
```