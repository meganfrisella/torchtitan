# Running torchtitan on EC2 with Docker

This guide covers building the Docker image, pushing to AWS ECR, testing locally,
and running on EC2 with the `llama3_debugmodel` config.

The image contains only the Python environment (PyTorch, CUDA, dependencies).
The torchtitan source is **mounted at runtime**, so code changes never require
a rebuild or repush — only dependency changes do.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed locally
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-clang.html) installed and configured (`aws configure`)
- An AWS account with permissions to create ECR repositories and EC2 instances
- Llama 3.1 70B tokenizer downloaded to `assets/hf/Llama-3.1-70B/` (see below)

### Download the Llama 3.1 70B tokenizer

Requires a Hugging Face account with access granted to
[meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B).

```bash
pip install huggingface_hub

huggingface-cli download meta-llama/Llama-3.1-70B \
  --include "tokenizer*.json" "special_tokens_map.json" \
  --local-dir $PATH_TO_TORCHTITAN/assets/hf/Llama-3.1-70B
```

---

## Variables

Set these before running any commands:

```bash
export PATH_TO_TORCHTITAN=/m-coriander/coriander/mfris/torchtitan
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REGISTRY=${ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com
```

---

## 1. Build the Docker Image

From the torchtitan repo root (the Docker build context):

```bash
docker build -t torchtitan-70b:latest $PATH_TO_TORCHTITAN
```

Only rebuild when dependencies change (i.e. `pyproject.toml` changes).

---

## 2. Test Locally

### 2a. Test local run (requires 4x GPU)

The tokenizer lives in the repo at `assets/hf/` and is mounted separately
because `hf_assets_path` resolves relative to the container's working directory
(`/workspace`), not the torchtitan source root.

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

### 2b. Test nsys tracing (requires 4x GPU)

Traces are written to `/workspace/out/` inside the container.

```bash
docker run --rm --gpus all \
  -v $PATH_TO_TORCHTITAN:/workspace/torchtitan \
  -v $PATH_TO_TORCHTITAN/assets/hf/:/workspace/assets/hf/ \
  -v $PATH_TO_TORCHTITAN/tests/assets:/workspace/tests/assets \
  -w /workspace \
  torchtitan-70b:latest \
  torchrun --nproc_per_node=4 \
  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  torchtitan/run.py torchtitan.train --module llama3 --config llama3_debugmodel
```

Traces will appear at `/workspace/out/torchtitan-compile-rank{0..3}.nsys-rep`.

---

## 3. Push Image

Choose either ECR (private, same-region pulls are free) or DockerHub (public, no IAM setup needed).

---

### Option A: ECR

#### 3a. Create the ECR repository (one-time)

```bash
aws ecr create-repository --repository-name torchtitan-70b --region us-east-2
```

Note the `repositoryUri` from the output:
`${ECR_REGISTRY}/torchtitan-70b`

#### 3a-ii. Create the ECRReadOnly instance profile (one-time)

EC2 instances use this profile to pull from ECR without manual credential setup.

```bash
aws iam create-role --role-name ECRReadOnly \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

aws iam attach-role-policy --role-name ECRReadOnly \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

aws iam create-instance-profile --instance-profile-name ECRReadOnly
aws iam add-role-to-instance-profile \
  --instance-profile-name ECRReadOnly \
  --role-name ECRReadOnly
```

#### 3b. Authenticate and push

```bash
aws ecr get-login-password --region us-east-2 \
  | docker login --username AWS --password-stdin $ECR_REGISTRY

docker tag torchtitan-70b:latest \
  ${ECR_REGISTRY}/torchtitan-70b:latest

docker push \
  ${ECR_REGISTRY}/torchtitan-70b:latest
```

---

### Option B: DockerHub

```bash
docker login

docker tag torchtitan-70b:latest <username>/torchtitan-70b:latest

docker push <username>/torchtitan-70b:latest
```

---

## 4. Run on EC2

### 4a. Instance setup

- Use a GPU instance with enough VRAM for 70B (e.g. `p4de.24xlarge` — 8x A100 80GB,
  or `p5.48xlarge` — 8x H100 80GB)
- Attach an IAM role with the `AmazonEC2ContainerRegistryReadOnly` policy
- Use the **same AWS region** as your ECR repository for free, fast pulls

### 4b. Install Docker and the NVIDIA Container Toolkit on the instance

```bash
# Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # log out and back in after this

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4c. Sync code to instance

The tokenizer lives in the repo at `assets/hf/` and is included in the rsync.

```bash
rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  $PATH_TO_TORCHTITAN/ ubuntu@<instance-ip>:/home/ubuntu/torchtitan/
```

### 4d. Pull and run

```bash
aws ecr get-login-password --region us-east-2 \
  | docker login --username AWS --password-stdin \
    ${ECR_REGISTRY}

docker pull \
  ${ECR_REGISTRY}/torchtitan-70b:latest

docker run --rm --gpus all \
  -v /home/ubuntu/torchtitan:/workspace/torchtitan \
  -v /home/ubuntu/torchtitan/assets/hf/:/workspace/assets/hf/ \
  -v /home/ubuntu/torchtitan/tests/assets:/workspace/tests/assets \
  ${ECR_REGISTRY}/torchtitan-70b:latest \
  torchrun --nproc_per_node=8 \
  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train --module llama3 --config llama3_debugmodel
```

### 4e. Iterating on code

For code changes, just rsync and rerun — no repull needed:

```bash
rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  $PATH_TO_TORCHTITAN/ ubuntu@<instance-ip>:/home/ubuntu/torchtitan/
```
