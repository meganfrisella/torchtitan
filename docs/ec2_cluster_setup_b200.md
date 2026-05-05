# EC2 Cluster Setup for TorchTitan (p6-b200.48xlarge)

Launches a 2-node torchtitan cluster on **p6-b200.48xlarge** (8x NVIDIA B200 GPUs,
3.2 Tbps EFA). Adapted from `ec2_cluster_setup.md`; see that doc for p4d.24xlarge
(A100) instructions.

The image contains only the Python environment. The torchtitan source is synced
separately and mounted at runtime, so code changes only require an rsync — not
a rebuild or repull.

## Key differences from p4d.24xlarge

| | p4d.24xlarge | p6-b200.48xlarge |
|---|---|---|
| GPUs | 8x A100 (40 GB) | 8x B200 (179 GB) |
| Network cards | 4 | 8 |
| EFA bandwidth | 400 Gbps | 3,200 Gbps |
| Interface type | `efa` (ENA+EFA combined) | `efa-only` (pure EFA, no IP) |
| Interfaces per node | 4 EFA | 1 ENA (primary) + 8 EFA-only |

**Placement group still required:** Cross-node EFA RDMA on p6-b200.48xlarge hangs
without a cluster placement group, same as p4d.

**Verify your primary interface name** before running Docker (Step 6/8b). The
primary ENA interface name in the OS depends on the AMI. On the piper AMI
`ami-03ab193c1b65d3fc7`, p4d uses `ens32`; p6-b200 may differ. Check with:
```bash
ssh -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP "ip -br link show | grep -v 'lo\|docker\|veth\|br-'"
```
Update `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to match the name of the
primary ENA interface (the one that has the private IP assigned to it).


## Variables

Set these before running any commands:

```bash
export PATH_TO_TORCHTITAN=/m-coriander/coriander/mfris/torchtitan
export REGION=us-east-2
export AMI=ami-02cfc6fe57ae221af #ami-035924cffff7b3476 #ami-03ab193c1b65d3fc7
export INSTANCE_TYPE=p6-b200.48xlarge
export SUBNET=subnet-0e27513d97f7aa13f  # public subnet (all NICs, both nodes)
export SG=sg-0687f77bfa22e1791
export KEY=ray-autoscaler_us-east-2
export CR_ID=cr-05530f9da9f836618  # set by Step 0, or hardcode: cr-XXXXXXXXXXXXXXXXX
export PG_NAME=piper-cluster-pg  # create once: aws ec2 create-placement-group --group-name $PG_NAME --strategy cluster --region $REGION
export SSH_KEY=~/.ssh/ray-autoscaler_us-east-2.pem
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REGISTRY=${ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com
export IMAGE=mfris/torchtitan-70b:latest
```

---

## Rebuild and Push

Only needed when dependencies change (i.e. `pyproject.toml` changes). Code changes
don't require a rebuild — torchtitan source is mounted at runtime.

```bash
docker build -t mfris/torchtitan-70b:latest $PATH_TO_TORCHTITAN
docker push mfris/torchtitan-70b:latest
```

---

## Test Locally

Before deploying to the cluster, verify the image and config work on a single node.

### Llama 3 debug model (requires 4x GPU)

```bash
docker run --rm --gpus all \
  -v $PATH_TO_TORCHTITAN:/workspace/torchtitan \
  -v $PATH_TO_TORCHTITAN/assets/hf/:/workspace/assets/hf/ \
  -v $PATH_TO_TORCHTITAN/tests/assets:/workspace/tests/assets \
  mfris/torchtitan-70b:latest \
  torchrun --nproc_per_node=4 \
  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train --module llama3 --config llama3_debugmodel
```

### Qwen 3 debug model (requires 4x GPU)

```bash
docker run --rm --gpus all \
  -v $PATH_TO_TORCHTITAN:/workspace/torchtitan \
  -v $PATH_TO_TORCHTITAN/assets/hf/:/workspace/assets/hf/ \
  -v $PATH_TO_TORCHTITAN/tests/assets:/workspace/tests/assets \
  mfris/torchtitan-70b:latest \
  torchrun --nproc_per_node=4 \
  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train --module qwen3 --config qwen3_moe_debug
```

---

### One time: Download the Llama 3.1 70B tokenizer

Requires a Hugging Face account with access granted to
[meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B).

```bash
pip install huggingface_hub

huggingface-cli download meta-llama/Llama-3.1-70B \
  --include "tokenizer*.json" "special_tokens_map.json" \
  --local-dir $PATH_TO_TORCHTITAN/assets/hf/Llama-3.1-70B
```

## Step 0: Create Capacity Reservation

```bash
CR_ID=$(aws ec2 create-capacity-reservation \
  --region us-east-2 \
  --instance-type p6-b200.48xlarge \
  --instance-platform Linux/UNIX \
  --availability-zone us-east-2a \
  --instance-count 2 \
  --instance-match-criteria targeted \
  --query 'CapacityReservation.CapacityReservationId' \
  --output text)

echo "Capacity reservation: $CR_ID"
```

## Release Stale EIPs (relaunch only)

If relaunching over a previous cluster, old EIPs become unassociated after the
instances are terminated. Find and release them before they accumulate:

```bash
# List all allocated EIPs — review before releasing
aws ec2 describe-addresses \
  --region $REGION \
  --query 'Addresses[].[AllocationId,PublicIp,AssociationId,InstanceId]' \
  --output table

# Release all unassociated EIPs
for alloc in $(aws ec2 describe-addresses \
  --region $REGION \
  --query 'Addresses[?AssociationId==null].AllocationId' \
  --output text); do
  aws ec2 release-address --allocation-id $alloc --region $REGION
  echo "Released $alloc"
done
```

---

```bash
  CONTROL_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --instance-type "$INSTANCE_TYPE" \
    --image-id "$AMI" \
    --key-name "$KEY" \
    --network-interfaces "[{\"DeviceIndex\":0,\"NetworkCardIndex\":0,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"interface\","DeleteOnTermination\":true}]" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":3000,"Throughput":500}}]' \
    --placement "GroupName=$PG_NAME" \
    --instance-market-options MarketType=capacity-block \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR_ID}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=torchtitan-control-no-efa}]" \
    --query 'Instances[0].InstanceId' \
    --output text)
```

## Step 1: Launch Head Node

p6-b200.48xlarge has 8 network cards. The minimum-IP configuration attaches
one primary ENA interface (card 0, gets the instance IP) and 8 EFA-only interfaces
(one per card for pure RDMA, no IP consumed). AWS prohibits `AssociatePublicIpAddress`
with multiple NICs, so we associate an EIP after launch for SSH access.

```bash
HEAD_ID=$(aws ec2 run-instances \
  --region $REGION \
  --instance-type $INSTANCE_TYPE \
  --image-id $AMI \
  --key-name $KEY \
  --network-interfaces "[
    {\"DeviceIndex\":0,\"NetworkCardIndex\":0,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"interface\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":1,\"NetworkCardIndex\":0,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":1,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":2,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":3,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":4,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":5,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":6,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":7,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true}
  ]" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":3000,"Throughput":500}}]' \
  --placement "GroupName=$PG_NAME" \
  --instance-market-options MarketType=capacity-block \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR_ID}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=torchtitan-head}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Head instance: $HEAD_ID"
aws ec2 wait instance-running --instance-ids $HEAD_ID --region $REGION

HEAD_PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$HEAD_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

HEAD_PRIVATE_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$HEAD_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text)

# Allocate an Elastic IP and associate it with the primary NIC for SSH access
EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --region $REGION \
  --query 'AllocationId' --output text)
HEAD_ENI=$(aws ec2 describe-instances --instance-ids $HEAD_ID --region $REGION \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.NetworkCardIndex==`0` && Attachment.DeviceIndex==`0`].NetworkInterfaceId' \
  --output text)
aws ec2 associate-address --allocation-id $EIP_ALLOC \
  --network-interface-id $HEAD_ENI --region $REGION

HEAD_PUBLIC_IP=$(aws ec2 describe-addresses --allocation-ids $EIP_ALLOC --region $REGION \
  --query 'Addresses[0].PublicIp' --output text)
HEAD_PRIVATE_IP=$(aws ec2 describe-instances --instance-ids $HEAD_ID --region $REGION \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)

echo "Head public IP:  $HEAD_PUBLIC_IP"
echo "Head private IP: $HEAD_PRIVATE_IP"
echo "EIP allocation:  $EIP_ALLOC  (release this on teardown)"
```

## Step 2: Launch Worker Node

```bash
WORKER_ID=$(aws ec2 run-instances \
  --region $REGION \
  --instance-type $INSTANCE_TYPE \
  --image-id $AMI \
  --key-name $KEY \
  # --subnet-id "$SUBNET" \
  # --security-group-ids "$SG" \
  # --associate-public-ip-address \
  --network-interfaces "[
    {\"DeviceIndex\":0,\"NetworkCardIndex\":0,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"interface\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":1,\"NetworkCardIndex\":0,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":1,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":2,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":3,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":4,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":5,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":6,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":0,\"NetworkCardIndex\":7,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa-only\",\"DeleteOnTermination\":true}
  ]" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":3000,"Throughput":500}}]' \
  --placement "GroupName=$PG_NAME" \
  --instance-market-options MarketType=capacity-block \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR_ID}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=torchtitan-worker}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Worker instance: $WORKER_ID"
aws ec2 wait instance-running --instance-ids $WORKER_ID --region $REGION

WORKER_PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$WORKER_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

WORKER_PRIVATE_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$WORKER_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text)

# Allocate an EIP for outbound internet access (ECR pull, HF download, etc.)
WORKER_EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --region $REGION \
  --query 'AllocationId' --output text)
WORKER_ENI=$(aws ec2 describe-instances --instance-ids $WORKER_ID --region $REGION \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.NetworkCardIndex==`0` && Attachment.DeviceIndex==`0`].NetworkInterfaceId' \
  --output text)
aws ec2 associate-address --allocation-id $WORKER_EIP_ALLOC \
  --network-interface-id $WORKER_ENI --region $REGION

WORKER_PUBLIC_IP=$(aws ec2 describe-addresses --allocation-ids $WORKER_EIP_ALLOC --region $REGION \
  --query 'Addresses[0].PublicIp' --output text)
WORKER_PRIVATE_IP=$(aws ec2 describe-instances --instance-ids $WORKER_ID --region $REGION \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)

echo "Worker private IP:  $WORKER_PRIVATE_IP"
echo "Worker EIP alloc:   $WORKER_EIP_ALLOC  (release this on teardown)"
```

## Step 3: Wait for SSH on Head

```bash
until ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$HEAD_PUBLIC_IP "echo ok" 2>/dev/null; do
  echo "Waiting for SSH..."; sleep 10
done
```

## Step 3b: Identify the Primary Interface Name

EFA-only interfaces have no IP address and don't appear as usable network
interfaces for socket-based communication. Only the primary ENA interface
(NetworkCardIndex=0, DeviceIndex=0) has an IP. Identify its name in the OS:

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "ip -br addr show | grep -v '127.0.0.1\|docker\|veth\|br-'"
```

Look for the interface that has the private IP (e.g. `10.x.x.x`). Set it:

```bash
export PRIMARY_IFACE=<name>  # e.g. ens5, ens32, or similar
```

Use `$PRIMARY_IFACE` in place of `ens32` everywhere below.

## Step 4: Pull Image and Sync Code to Head

Choose one of the two options below to pull the image on the head node.


```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "
  docker pull mfris/torchtitan-70b:latest
"
export IMAGE=mfris/torchtitan-70b:latest
```


## Step 6: Start Docker on Head

The `efa-only` interfaces surface inside the container through `/dev/infiniband`
(same as p4d). The EFA library bind mount (`/opt/amazon/efa`) and the
`FI_PROVIDER=efa` env var tell libfabric to use them for RDMA. Only
`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` need to name the primary ENA
interface (the one with the IP address).

Replace `ens32` with `$PRIMARY_IFACE` identified in Step 3b.

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out' \
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
    -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e NCCL_PROTO=simple \
    $IMAGE sleep infinity
"
```

## Step 7: Wait for SSH on Worker

The worker has no public IP — SSH via the head as a jump host.

```bash
until ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP "echo ok" 2>/dev/null; do
  echo "Waiting for worker SSH..."; sleep 10
done
```

## Step 8: Pull Image, Sync Code on Worker

Choose one of the two options below to pull the image on the worker node.

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP "
  docker pull mfris/torchtitan-70b:latest
"
```

Sync the local source tree to the worker before starting Docker. Docker bind
mounts this host directory at `/workspace/torchtitan`; if the directory is empty,
the container will also see an empty source tree.

## Step 8b: Start Docker on Worker

Replace `ens32` with `$PRIMARY_IFACE` identified in Step 3b.

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
  $PATH_TO_TORCHTITAN/ ubuntu@$WORKER_PRIVATE_IP:/home/ubuntu/torchtitan/

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP "
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
    -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e NCCL_PROTO=simple \
    $IMAGE sleep infinity
"
```

## Step 8c: Fix /etc/hosts for Cross-Node torchrun

c10d rendezvous tries to resolve peer hostnames via DNS. AWS EC2 internal
hostnames (`ip-A-B-C-D`) are not resolvable inside Docker containers, causing
`torchrun` to hang silently. Add both nodes to each container's `/etc/hosts`:

```bash
HEAD_HOSTNAME="ip-$(echo $HEAD_PRIVATE_IP | tr . -)"
WORKER_HOSTNAME="ip-$(echo $WORKER_PRIVATE_IP | tr . -)"

# Head container: add worker entry
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -u root torchtitan bash -c 'echo \"$WORKER_PRIVATE_IP $WORKER_HOSTNAME\" >> /etc/hosts'"

# Worker container: add head entry
ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -u root torchtitan bash -c 'echo \"$HEAD_PRIVATE_IP $HEAD_HOSTNAME\" >> /etc/hosts'"
```

## Step 8d: Fix NCCL Topology File Path

NCCL looks for the topology XML at `/opt/amazon/ofi-nccl/share/...` but the
mount point inside the container is `/opt/aws-ofi-nccl/`. Without this symlink
the topology is silently skipped and NCCL may hang at the first collective.

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -u root torchtitan bash -c 'mkdir -p /opt/amazon && ln -sfn /opt/aws-ofi-nccl /opt/amazon/ofi-nccl'"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -u root torchtitan bash -c 'mkdir -p /opt/amazon && ln -sfn /opt/aws-ofi-nccl /opt/amazon/ofi-nccl'"
```

## Step 8e: Verify Source Mounts

Confirm both containers can see the synced source tree before launching a
training job:

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w /workspace/torchtitan torchtitan test -f ./run_train.sh"

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -w /workspace/torchtitan torchtitan test -f ./run_train.sh"
```

## Step 9: Verify EFA

### 9a. Host-level check

Verify EFA cross-node RDMA works at the host level.
Start server on head, then run client on worker — should print latency/bandwidth numbers.

```bash
ssh -n -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP \
  "nohup /opt/amazon/efa/bin/fi_pingpong -p efa > /tmp/fi_ping_server.log 2>&1" &
sleep 5
ssh -i $SSH_KEY \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "/opt/amazon/efa/bin/fi_pingpong -p efa $HEAD_PRIVATE_IP"
ssh -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP "cat /tmp/fi_ping_server.log"
```

### 9b. Container-level check

Verify the EFA bind mounts (`--device /dev/infiniband`, `/opt/amazon/efa`) are
accessible from inside the containers. This must pass before running training.

```bash
ssh -n -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP \
  "nohup docker exec torchtitan /opt/amazon/efa/bin/fi_pingpong -p efa \
   > /tmp/fi_ping_container_server.log 2>&1" &
sleep 5
ssh -i $SSH_KEY \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec torchtitan /opt/amazon/efa/bin/fi_pingpong -p efa $HEAD_PRIVATE_IP"
ssh -i $SSH_KEY ubuntu@$HEAD_PUBLIC_IP "cat /tmp/fi_ping_container_server.log"
```

## Step 10: Run Tests

Run these in order. Each step validates a prerequisite for the next.

### 10a. EFA connectivity

Run the container-level fi_pingpong from Step 9b. Both nodes must pass before
proceeding.

### 10b. Debug model with nsys tracing

Smoke test cross-node training and verify nsys tracing works. Uses static
rendezvous (`--node_rank`) to avoid c10d hostname resolution issues. Both nodes
can start simultaneously.

To confirm EFA is active, add `-e NCCL_DEBUG=INFO` and look for
`"NET/OFI Initializing aws-ofi-nccl"` and `"Using EFA"` in the output.

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w /workspace \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   torchrun --nnodes=2 --nproc_per_node=8 \
   --node_rank=0 \
   --master_addr=$HEAD_PRIVATE_IP --master_port=29500 \
   --local-ranks-filter 0 --role rank --tee 3 \
   torchtitan/run.py torchtitan.train --module llama3 --config llama3_debugmodel" &

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -w /workspace \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   torchrun --nnodes=2 --nproc_per_node=8 \
   --node_rank=1 \
   --master_addr=$HEAD_PRIVATE_IP --master_port=29500 \
   --local-ranks-filter 0 --role rank --tee 3 \
   torchtitan/run.py torchtitan.train --module llama3 --config llama3_debugmodel"
wait
```

### 10c. Llama 70B run

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'NNODE=2 NGPU=8 LOG_RANK=0 CONFIG=llama3_70b NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'" &

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'NNODE=2 NGPU=8 LOG_RANK=0 CONFIG=llama3_70b NODE_RANK=1 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'"
wait
```

Force-terminate if hung:

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec torchtitan pkill -9 -f torchtitan.train; pkill -9 -f torchrun" &
ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec torchtitan pkill -9 -f torchtitan.train; pkill -9 -f torchrun"
wait
```

### 10d. Qwen3-30B-A3B with EP=2, PP=4

Runs `qwen3_30b` (MoE, 30B-A3B) across 2 nodes. The config has `expert_parallel_degree=2`,
`pipeline_parallel_degree=4`, and `data_parallel_shard_degree=-1` (resolves to 4 on 16 GPUs),
using the Qwen3-1.7B tokenizer at `./assets/hf/Qwen3-1.7B`.

**Topology note:** `parallel_dims.py` builds the sparse mesh with DP/FSDP before PP in
rank-major order. On 2 nodes with contiguous `torchrun` ranks per node, this keeps PP
stages within a node and places expert-parallel communication across nodes.

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'NNODE=2 NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_30b NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'" &

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'NNODE=2 NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_30b NODE_RANK=1 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'"
wait
```

### 10e. Qwen3-1B

Runs `qwen3_1b` across 2 nodes x 4 GPUs with `pipeline_parallel_degree=4`,
`expert_parallel_degree=2`, and `data_parallel_shard_degree=-1` (resolves to 2).
With the DP/FSDP-first sparse mesh in `parallel_dims.py`, each 4-rank PP group
stays within a node while each EP group synchronizes across nodes.

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'echo HEAD; pwd; ls -l ./run_train.sh; NNODE=2 NGPU=4 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_1b NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'" &

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'echo WORKER; pwd; ls -l ./run_train.sh; NNODE=2 NGPU=4 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_1b NODE_RANK=1 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'"
wait
```

Resync code on head and worker

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $PATH_TO_TORCHTITAN/ ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
-o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
ubuntu@$WORKER_PRIVATE_IP \
"sudo mkdir -p /home/ubuntu/torchtitan && sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan"

rsync -av --delete \
--exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out' \
-e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
$PATH_TO_TORCHTITAN/ ubuntu@$WORKER_PRIVATE_IP:/home/ubuntu/torchtitan/
```

### 10f. Qwen3-9B

Runs `qwen3_9b` across 2 nodes x 8 GPUs with `pipeline_parallel_degree=4`,
`expert_parallel_degree=2`, and `data_parallel_shard_degree=-1` (resolves to 2).
With the DP/FSDP-first sparse mesh in `parallel_dims.py`, each 4-rank PP group
stays within a node while each EP group synchronizes across nodes.

```bash
# Create out/ dir on both nodes (nsys needs it to exist before writing traces)
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec torchtitan mkdir -p /workspace/torchtitan/out"
ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec torchtitan mkdir -p /workspace/torchtitan/out"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@"$HEAD_PUBLIC_IP" \
  "docker exec -w /workspace/torchtitan \
    -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e NCCL_NET=Socket \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_PROTO=simple \
    torchtitan \
    bash -lc 'unset FI_PROVIDER FI_EFA_USE_DEVICE_RDMA FI_EFA_FORK_SAFE RDMAV_FORK_SAFE; echo HEAD; pwd; ls -l ./run_train.sh; NNODE=2 NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_30b NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'" &

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@"$WORKER_PRIVATE_IP" \
  "docker exec -w /workspace/torchtitan \
    -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
    -e NCCL_NET=Socket \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_PROTO=simple \
    torchtitan \
    bash -lc 'unset FI_PROVIDER FI_EFA_USE_DEVICE_RDMA FI_EFA_FORK_SAFE RDMAV_FORK_SAFE; echo WORKER; pwd; ls -l ./run_train.sh; NNODE=2 NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_30b NODE_RANK=1 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'"
wait

# ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
#   "docker exec -w /workspace/torchtitan \
#    -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
#    -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
#    -e NCCL_PROTO=simple \
#    torchtitan \
#    bash -c 'echo HEAD; pwd; ls -l ./run_train.sh; NNODE=2 NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_9b NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'" &

# ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
#   -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
#   ubuntu@$WORKER_PRIVATE_IP \
#   "docker exec -w /workspace/torchtitan \
#    -e NCCL_SOCKET_IFNAME=$PRIMARY_IFACE \
#    -e GLOO_SOCKET_IFNAME=$PRIMARY_IFACE \
#    -e NCCL_PROTO=simple \
#    torchtitan \
#    bash -c 'echo WORKER; pwd; ls -l ./run_train.sh; NNODE=2 NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_9b NODE_RANK=1 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'"
# wait
```

---

## Downloading Traces

Nsys traces are written to `/workspace/torchtitan/out/`, which is bind-mounted from
`/home/ubuntu/torchtitan/out/` on each host — no `docker cp` needed.

```bash
# From head node
rsync -av \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/out/ \
  $PATH_TO_TORCHTITAN/ec2-out/head-qwen30b/

# From worker node (via head as jump host)
rsync -av \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
  ubuntu@$WORKER_PRIVATE_IP:/home/ubuntu/torchtitan/out/ \
  $PATH_TO_TORCHTITAN/ec2-out/worker-qwen30b/
```

---

## Updating Code

Since the source is mounted from the host, code changes only require an rsync
to each node — no rebuild or repull needed:

```bash
RSYNC_OPTS=(
  --delete
  --delete-excluded
  --exclude='.git'
  --exclude='__pycache__'
  --exclude='.venv'
  --exclude='venv'
  --exclude='out/***'
  --exclude='assets/hf/***'
)

# Head
rsync -av "${RSYNC_OPTS[@]}" \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $PATH_TO_TORCHTITAN/ ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/

# Worker (via head as jump host)
rsync -av "${RSYNC_OPTS[@]}" \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
  $PATH_TO_TORCHTITAN/ ubuntu@$WORKER_PRIVATE_IP:/home/ubuntu/torchtitan/
```

Only rebuild and repush the image when `pyproject.toml` dependencies change.

---

## Step 11: Teardown

```bash
aws ec2 terminate-instances --instance-ids $HEAD_ID $WORKER_ID --region $REGION
aws ec2 wait instance-terminated --instance-ids $HEAD_ID $WORKER_ID --region $REGION
aws ec2 release-address --allocation-id $EIP_ALLOC --region $REGION
aws ec2 release-address --allocation-id $WORKER_EIP_ALLOC --region $REGION
aws ec2 cancel-capacity-reservation --capacity-reservation-id $CR_ID --region $REGION
```
