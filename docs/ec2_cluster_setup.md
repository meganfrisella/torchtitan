# EC2 Cluster Setup for TorchTitan

Launches a 2-node torchtitan cluster using AWS CLI directly. Uses the same AMI
and networking setup as the piper cluster (`ami-03ab193c1b65d3fc7`), which has
Docker, EFA drivers, and aws-ofi-nccl pre-installed.

The image contains only the Python environment. The torchtitan source is synced
separately and mounted at runtime, so code changes only require an rsync — not
a rebuild or repull.

**Important:** A cluster placement group is required for cross-node EFA RDMA
to work on p4d.24xlarge. Without it, `fi_pingpong` and NCCL collectives hang
at the data-transfer stage even though NCCL init succeeds.


## Variables

Set these before running any commands:

```bash
export PATH_TO_TORCHTITAN=/m-coriander/coriander/mfris/torchtitan
export REGION=us-east-2
export AMI=ami-03ab193c1b65d3fc7
export INSTANCE_TYPE=p4d.24xlarge
export SUBNET=subnet-0572b36ed0e0551f2  # public subnet (all NICs, both nodes)
export SG=sg-0687f77bfa22e1791
export KEY=ray-autoscaler_us-east-2
export CR_ID=$CR_ID  # set by Step 0, or hardcode: cr-XXXXXXXXXXXXXXXXX
export PG_NAME=piper-cluster-pg  # create once: aws ec2 create-placement-group --group-name $PG_NAME --strategy cluster --region $REGION
export SSH_KEY=~/.ssh/ray-autoscaler_us-east-2.pem
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REGISTRY=${ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com
export IMAGE=${ECR_REGISTRY}/torchtitan-70b:latest  # Option A (ECR); for Option B use: export IMAGE=mfris/torchtitan-70b:latest
```

---

## Test Locally (requires 4x GPU)

Before deploying to the cluster, verify the image and config work on a single node:

```bash
docker run --rm --gpus all \
  -v $PATH_TO_TORCHTITAN:/workspace/torchtitan \
  -v $PATH_TO_TORCHTITAN/assets/hf/:/workspace/assets/hf/ \
  -v $PATH_TO_TORCHTITAN/tests/assets:/workspace/tests/assets \
  torchtitan-70b:latest \
  torchrun --nproc_per_node=4 \
  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train --module llama3 --config llama3_debugmodel
```

---

## Step 0: Create Capacity Reservation

```bash
CR_ID=$(aws ec2 create-capacity-reservation \
  --region us-east-2 \
  --instance-type p4d.24xlarge \
  --instance-platform Linux/UNIX \
  --availability-zone us-east-2a \
  --instance-count 2 \
  --instance-match-criteria targeted \
  --query 'CapacityReservation.CapacityReservationId' \
  --output text)

echo "Capacity reservation: $CR_ID"
```

---

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

## Step 1: Launch Head Node

AWS prohibits `AssociatePublicIpAddress` with multiple NICs, so we associate
an EIP after launch for SSH access to the head.

```bash
HEAD_ID=$(aws ec2 run-instances \
  --region $REGION \
  --instance-type $INSTANCE_TYPE \
  --image-id $AMI \
  --key-name $KEY \
  --network-interfaces "[
    {\"DeviceIndex\":0,\"NetworkCardIndex\":0,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":1,\"NetworkCardIndex\":1,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":2,\"NetworkCardIndex\":2,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":3,\"NetworkCardIndex\":3,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true}
  ]" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":3000,"Throughput":500}}]' \
  --placement "GroupName=$PG_NAME" \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR_ID}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=torchtitan-head}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Head instance: $HEAD_ID"
aws ec2 wait instance-running --instance-ids $HEAD_ID --region $REGION

# Allocate an Elastic IP and associate it with the primary NIC for SSH access
EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --region $REGION \
  --query 'AllocationId' --output text)
HEAD_ENI=$(aws ec2 describe-instances --instance-ids $HEAD_ID --region $REGION \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].NetworkInterfaceId' \
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
  --network-interfaces "[
    {\"DeviceIndex\":0,\"NetworkCardIndex\":0,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":1,\"NetworkCardIndex\":1,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":2,\"NetworkCardIndex\":2,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true},
    {\"DeviceIndex\":3,\"NetworkCardIndex\":3,\"SubnetId\":\"$SUBNET\",\"Groups\":[\"$SG\"],\"InterfaceType\":\"efa\",\"DeleteOnTermination\":true}
  ]" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":3000,"Throughput":500}}]' \
  --placement "GroupName=$PG_NAME" \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR_ID}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=torchtitan-worker}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Worker instance: $WORKER_ID"
aws ec2 wait instance-running --instance-ids $WORKER_ID --region $REGION

# Allocate an EIP for outbound internet access (ECR pull, HF download, etc.)
WORKER_EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --region $REGION \
  --query 'AllocationId' --output text)
WORKER_ENI=$(aws ec2 describe-instances --instance-ids $WORKER_ID --region $REGION \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].NetworkInterfaceId' \
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

## Step 4: Pull Image and Sync Code to Head

Choose one of the two options below to pull the image on the head node.

**Option A: Pull from ECR** (requires `AmazonEC2ContainerRegistryReadOnly` on the instance IAM role)

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "
  aws ecr get-login-password --region $REGION \
    | docker login --username AWS --password-stdin $ECR_REGISTRY
  docker pull $IMAGE
"
```

**Option B: Pull from DockerHub**

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP "
  docker pull mfris/torchtitan-70b:latest
"
export IMAGE=mfris/torchtitan-70b:latest
```

```bash
rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $PATH_TO_TORCHTITAN/ ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/
```

## Step 6: Start Docker on Head

```bash
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

## Step 8: Pull Image, Sync Code, and Start Docker on Worker

Choose one of the two options below to pull the image on the worker node.

**Option A: Pull from ECR** (requires `AmazonEC2ContainerRegistryReadOnly` on the instance IAM role)

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP "
  aws ecr get-login-password --region $REGION \
    | docker login --username AWS --password-stdin $ECR_REGISTRY
  docker pull $IMAGE
"
```

**Option B: Pull from DockerHub**

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP "
  docker pull mfris/torchtitan-70b:latest
"
```

```bash
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
    -e NCCL_SOCKET_IFNAME=ens32 \
    -e GLOO_SOCKET_IFNAME=ens32 \
    -e NCCL_PROTO=simple \
    $IMAGE sleep infinity
"
```

## Step 8b: Fix /etc/hosts for Cross-Node torchrun

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

## Step 8c: Fix NCCL Topology File Path

NCCL looks for the p4d topology XML at `/opt/amazon/ofi-nccl/share/...` but the
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
   -e NCCL_SOCKET_IFNAME=ens32 \
   -e GLOO_SOCKET_IFNAME=ens32 \
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
   -e NCCL_SOCKET_IFNAME=ens32 \
   -e GLOO_SOCKET_IFNAME=ens32 \
   -e NCCL_PROTO=simple \
   torchtitan \
   torchrun --nnodes=2 --nproc_per_node=8 \
   --node_rank=1 \
   --master_addr=$HEAD_PRIVATE_IP --master_port=29500 \
   --local-ranks-filter 0 --role rank --tee 3 \
   torchtitan/run.py torchtitan.train --module llama3 --config llama3_debugmodel"
wait
```

### 10c. Full 70B run

```bash
ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=ens32 \
   -e GLOO_SOCKET_IFNAME=ens32 \
   -e NCCL_PROTO=simple \
   torchtitan \
   bash -c 'NNODE=2 NGPU=8 LOG_RANK=0 CONFIG=llama3_70b NODE_RANK=0 MASTER_ADDR=$HEAD_PRIVATE_IP MASTER_PORT=29500 ./run_train.sh'" &

ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
  -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
  ubuntu@$WORKER_PRIVATE_IP \
  "docker exec -w /workspace/torchtitan \
   -e NCCL_SOCKET_IFNAME=ens32 \
   -e GLOO_SOCKET_IFNAME=ens32 \
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

---

## Downloading Traces

Nsys traces are written to `/workspace/torchtitan/out/`, which is bind-mounted from
`/home/ubuntu/torchtitan/out/` on each host — no `docker cp` needed.

```bash
# From head node
rsync -av \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/out/ \
  $PATH_TO_TORCHTITAN/ec2-out/

# From worker node (via head as jump host)
rsync -av \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
  ubuntu@$WORKER_PRIVATE_IP:/home/ubuntu/torchtitan/out/ \
  $PATH_TO_TORCHTITAN/ec2-out/
```

---

## Updating Code

Since the source is mounted from the host, code changes only require an rsync
to each node — no rebuild or repull needed:

```bash
RSYNC_OPTS="--delete --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='out'"

# Head
rsync -av $RSYNC_OPTS \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  $PATH_TO_TORCHTITAN/ ubuntu@$HEAD_PUBLIC_IP:/home/ubuntu/torchtitan/

# Worker (via head as jump host)
rsync -av $RSYNC_OPTS \
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
