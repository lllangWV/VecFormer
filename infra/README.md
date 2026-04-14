# EC2 Simple Infrastructure

GPU-based ML service deployment on EC2 using Terraform. The infrastructure is split into two tiers with shared modules.

## Directory Structure

```
infra/ec2-simple/
├── modules/        # Shared Terraform modules
│   └── ec2-serve/  # Reusable EC2 + GPU + container module
├── platforms/      # Tier 1: Permanent shared infrastructure
│   ├── ecr.tf      # ECR repositories
│   ├── ssh.tf      # SSH key pairs
│   └── outputs.tf  # VPC/subnet discovery
└── services/       # Tier 2: Deployable service instances
    ├── sam3/        # SAM3 segmentation
    ├── glm-ocr/     # GLM-OCR via vLLM
    ├── pp-doclayout/# PaddlePaddle document layout
    ├── combined/    # SAM3 + GLM-OCR on one instance
    ├── ocr-layout/  # OCR + layout detection
    └── ykkvision/   # Full stack: FastAPI + GLM-OCR + Qwen3 (docker compose)
```

### `modules/` -- Shared Code

Reusable Terraform modules called by services. `ec2-serve` handles the common pattern: find a Deep Learning AMI, create IAM roles (ECR pull, S3 model access, CloudWatch, SSM), set up a security group, and launch a GPU EC2 instance with a user-data bootstrap script.

### `platforms/` -- Tier 1 (Permanent)

Long-lived shared infrastructure that services depend on. Deployed once and rarely changed. Contains ECR repositories (with lifecycle policies), optional SSH key pairs, and data sources that discover VPC subnets. Destroying platform resources affects all services.

### `services/` -- Tier 2 (Temporary)

Individual service deployments. Each is an independent Terraform root module that can be created and destroyed without affecting other services. Most services use the `ec2-serve` module directly; `ykkvision` uses a custom docker-compose setup for multi-container orchestration.

## Services

| Service | Instance | Description |
|---------|----------|-------------|
| `sam3` | g5.xlarge | SAM3 segmentation model |
| `glm-ocr` | g5.xlarge | GLM-OCR document OCR via vLLM |
| `pp-doclayout` | g5.xlarge | PaddlePaddle document layout detection |
| `combined` | g6.xlarge | SAM3 + GLM-OCR on a single instance |
| `ocr-layout` | g7e.2xlarge | OCR + layout on Blackwell GPU (96GB VRAM) |
| `ykkvision` | g7e.2xlarge | Full stack: FastAPI app + GLM-OCR vLLM + Qwen3 vLLM (FP8) via docker compose |

## Prerequisites

- Terraform installed
- AWS CLI configured with the `ykk` profile (see below)
- Platform infrastructure deployed (for ECR repos)
- Docker images pushed to ECR (see root `justfile`)

## AWS Profile Setup

All commands use `AWS_PROFILE=ykk` to target the correct account (`064561338865`). Configure it with the AWS CLI:

```bash
aws configure --profile ykk
```

You will be prompted for:

```
AWS Access Key ID: <your-access-key>
AWS Secret Access Key: <your-secret-key>
Default region name: us-east-1
Default output format: json
```

Verify the profile is working:

```bash
AWS_PROFILE=ykk aws sts get-caller-identity
```

You should see account `064561338865` in the output. To avoid passing `AWS_PROFILE` on every command, you can export it for your session:

```bash
export AWS_PROFILE=ykk
```

## Deploying a Service

Each service lives in its own directory under `services/` and is an independent Terraform root module.

```bash
cd infra/ec2-simple/services/<service-name>

# Initialize Terraform (first time or after provider changes)
terraform init

# Preview what will be created
AWS_PROFILE=ykk terraform plan -var-file=ykk.tfvars

# Deploy
AWS_PROFILE=ykk terraform apply -var-file=ykk.tfvars
```

Terraform outputs the instance's public IP and service URL after apply.

## Updating a Service

For **code/image updates** (new Docker image pushed to ECR), SSH into the instance and restart:

```bash
ssh ubuntu@<public-ip>
cd /opt/ec2-serve
docker compose pull && docker compose up -d --remove-orphans
```

For **infrastructure changes** (instance type, environment variables, security groups, etc.), re-run apply:

```bash
cd infra/ec2-simple/services/<service-name>
AWS_PROFILE=ykk terraform apply -var-file=ykk.tfvars
```

Note: changes to `user_data` will not take effect on a running instance. To force recreation, taint the instance first:

```bash
AWS_PROFILE=ykk terraform taint aws_instance.this
AWS_PROFILE=ykk terraform apply -var-file=ykk.tfvars
```

## Destroying a Service

```bash
cd infra/ec2-simple/services/<service-name>
AWS_PROFILE=ykk terraform destroy -var-file=ykk.tfvars
```

This tears down the EC2 instance, security group, and IAM resources for that service only. Other services and platform infrastructure are unaffected.

## Deploying Platform Infrastructure

Platform resources (ECR repos, SSH keys) should be deployed before any services:

```bash
cd infra/ec2-simple/platforms
terraform init
AWS_PROFILE=ykk terraform apply -var-file=ykk.tfvars
```

ECR repositories have `prevent_destroy` lifecycle rules to avoid accidental deletion.
