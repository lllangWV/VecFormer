# VecFormer Infrastructure

Terraform infrastructure for VecFormer development and training on AWS GPU instances.

## Directory Structure

```
infra/
├── modules/                    # Reusable Terraform modules
│   ├── s3-training-bucket/     # S3 bucket for training artifacts
│   ├── persistent-ebs/         # Persistent EBS volume
│   └── sns-notifications/      # SNS email notifications
│
├── services/                   # Deployable services
│   ├── embd/                   # Simple SSH-accessible GPU instance
│   ├── training-s3/            # S3-centric training (Option 1)
│   └── training-ebs/           # EBS-centric training (Option 3)
│
└── envs/                       # Environment-specific configurations
    └── us-east-2/
        ├── embd.tfvars         # Dev/exploration instance config
        ├── training-s3.tfvars  # S3 training config
        └── training-ebs.tfvars # EBS training config
```

## Services Overview

| Service | Purpose | Instance | Spot Price |
|---------|---------|----------|------------|
| `embd` | Development/exploration | g7e.2xlarge (1× L40S) | ~$0.90/hr |
| `training-s3` | Production training (S3 storage) | p5en.48xlarge (8× H200) | ~$11/hr |
| `training-ebs` | Production training (EBS storage) | p5en.48xlarge (8× H200) | ~$11/hr |

## Prerequisites

### 1. AWS CLI with `ykk` Profile

```bash
aws configure --profile ykk
```

Verify:
```bash
AWS_PROFILE=ykk aws sts get-caller-identity
```

### 2. Terraform >= 1.0

```bash
brew install terraform  # macOS
```

### 3. Just Command Runner

```bash
brew install just  # macOS
```

### 4. SSH Key

Ensure you have an SSH key at `~/.ssh/id_ed25519` (or update the tfvars).

---

## Service 1: Development Instance (`embd`)

Simple GPU instance for exploration, testing, and development.

### Quick Start

```bash
# Initialize (first time)
just tf-init

# Deploy
just tf-apply

# Add to SSH config
just ssh-config-add

# SSH in
ssh vecformer

# Stop when not using (saves cost)
just stop-instance

# Start again
just start-instance

# Destroy when done
just tf-destroy
```

### Commands

| Command | Description |
|---------|-------------|
| `just tf-init` | Initialize Terraform |
| `just tf-plan` | Preview changes |
| `just tf-apply` | Deploy instance |
| `just tf-destroy` | Destroy instance |
| `just ssh-instance` | SSH via EC2 Instance Connect |
| `just ssh-config-add` | Add `Host vecformer` to ~/.ssh/config |
| `just ssh-config-remove` | Remove from SSH config |
| `just start-instance` | Start stopped instance |
| `just stop-instance` | Stop instance |
| `just instance-status` | Check instance state |
| `just nvidia-smi` | Run nvidia-smi remotely |

---

## Service 2: S3-Centric Training (`training-s3`)

Ephemeral instance with S3 for persistent storage. Training starts automatically, syncs checkpoints to S3, and self-terminates on completion.

### Features

- ✅ Auto-starts training on boot
- ✅ Syncs checkpoints to S3 every 5 minutes
- ✅ Email notification on completion/failure
- ✅ Self-terminates when done
- ✅ Handles spot interruption (saves checkpoint)
- ⚠️ Manual restart required after interruption

### Quick Start

```bash
# 1. Configure (REQUIRED: set subnet_id and run_id)
vim infra/envs/us-east-2/training-s3.tfvars

# 2. Initialize
just train-s3-init

# 3. Start training
just train-s3-start

# 4. Monitor
just train-s3-logs
just train-s3-gpu

# 5. Download results (after completion email)
just train-s3-results my-run-id

# 6. Cleanup
just train-s3-stop
```

### Commands

| Command | Description |
|---------|-------------|
| `just train-s3-init` | Initialize Terraform |
| `just train-s3-plan` | Preview changes |
| `just train-s3-start` | Start training |
| `just train-s3-stop` | Destroy instance |
| `just train-s3-ssh` | SSH into instance |
| `just train-s3-logs` | Tail training logs |
| `just train-s3-gpu` | Check GPU status |
| `just train-s3-output` | Show Terraform outputs |
| `just train-s3-results <run_id>` | Download results from S3 |

---

## Service 3: EBS-Centric Training (`training-ebs`)

Persistent EBS volume with automatic spot recovery. Training auto-resumes after interruption.

### Features

- ✅ Auto-starts training on boot
- ✅ Persistent EBS volume survives termination
- ✅ **Automatic recovery** from spot interruption
- ✅ Email notification on completion/failure
- ✅ Self-terminates when done
- ✅ No data loss on interruption

### Quick Start

```bash
# 1. Configure (REQUIRED: set subnet_id, availability_zone, run_id)
vim infra/envs/us-east-2/training-ebs.tfvars

# 2. Initialize
just train-ebs-init

# 3. Start training
just train-ebs-start

# 4. Monitor
just train-ebs-logs
just train-ebs-state
just train-ebs-gpu

# 5. Download results (after completion email)
just train-ebs-results my-run-id

# 6. Cleanup (WARNING: deletes all data!)
just train-ebs-destroy
```

### Commands

| Command | Description |
|---------|-------------|
| `just train-ebs-init` | Initialize Terraform |
| `just train-ebs-plan` | Preview changes |
| `just train-ebs-start` | Start training |
| `just train-ebs-stop` | Stop instance (EBS persists) |
| `just train-ebs-resume` | Restart stopped instance |
| `just train-ebs-destroy` | Destroy everything |
| `just train-ebs-ssh` | SSH into instance |
| `just train-ebs-logs` | Tail training logs |
| `just train-ebs-gpu` | Check GPU status |
| `just train-ebs-state` | Check training state |
| `just train-ebs-output` | Show Terraform outputs |
| `just train-ebs-results <run_id>` | SCP results from instance |

---

## Configuration

### Finding Your Subnet ID

```bash
AWS_PROFILE=ykk aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=vpc-097a3bf6bd387ae0b" \
    --query "Subnets[*].{SubnetId:SubnetId,AZ:AvailabilityZone,CIDR:CidrBlock}" \
    --output table
```

### Required tfvars Updates

**Before first run**, update these values:

```hcl
# training-s3.tfvars
subnet_id = "subnet-xxxxxxxxx"      # Your subnet ID
run_id    = "my-experiment-1"       # Unique run identifier

# training-ebs.tfvars (additional)
availability_zone = "us-east-2a"    # Must match subnet's AZ
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instance_type` | p5en.48xlarge | 8× H200 GPUs |
| `num_epochs` | 500 | Total training epochs |
| `checkpoint_epochs` | 10 | Save checkpoint every N epochs |
| `batch_size_per_gpu` | 10 | Batch size per GPU |
| `num_gpus` | 8 | Number of GPUs to use |
| `notification_email` | (required) | Email for alerts |

---

## Modules

### `s3-training-bucket`

S3 bucket with:
- Versioning enabled
- Lifecycle rules (checkpoint cleanup after 7 days, results archived after 30 days)
- Server-side encryption (AES256)
- Public access blocked
- IAM policy for instance access

### `persistent-ebs`

Persistent EBS volume:
- gp3 type with configurable IOPS/throughput
- Survives instance termination
- IAM policy for attach/detach operations

### `sns-notifications`

SNS topic for notifications:
- Email subscription (confirmation required)
- IAM policy for publishing
- Used for: training started, completed, failed, interrupted

---

## Training Comparison

| Aspect | S3-Centric | EBS-Centric |
|--------|------------|-------------|
| **Storage** | S3 (multi-region) | EBS (single AZ) |
| **Spot Recovery** | Manual restart | Automatic |
| **Resume Speed** | ~5-10 min (S3 download) | ~1-2 min (EBS attach) |
| **Data Durability** | 99.999999999% (S3) | 99.999% (EBS) |
| **Monthly Storage Cost** | ~$1.15 (50GB) | ~$16 (200GB) |
| **Complexity** | Medium | Lower |

**Recommendation:**
- Use **EBS** for production training (automatic recovery)
- Use **S3** for experiments or cross-region needs

---

## Cost Estimates

### Compute (Spot)

| Instance | GPUs | Spot Price | 500 Epochs | Total |
|----------|------|------------|------------|-------|
| g7e.2xlarge | 1× L40S | ~$0.90/hr | ~60 hrs | ~$54 |
| p5en.48xlarge | 8× H200 | ~$11/hr | ~1.5 hrs | ~$17 |

### Storage

| Resource | Size | Monthly Cost |
|----------|------|--------------|
| S3 (checkpoints + results) | ~50 GB | ~$1.15 |
| EBS gp3 volume | 200 GB | ~$16 |

---

## Troubleshooting

### Spot Request Not Fulfilled

```bash
# Check spot request status
AWS_PROFILE=ykk aws ec2 describe-spot-instance-requests \
    --query "SpotInstanceRequests[*].{ID:SpotInstanceRequestId,State:State,Status:Status.Message}" \
    --output table
```

**Common issues:**
- Insufficient capacity in AZ → Try different AZ
- Price too low → Increase `spot_max_price`

### Training Not Starting

```bash
# SSH and check logs
just train-s3-ssh  # or train-ebs-ssh

# On instance:
sudo cat /var/log/user-data.log
sudo systemctl status vecformer-training
sudo journalctl -u vecformer-training -f
```

### EBS Volume Not Attaching

```bash
# Check volume status
AWS_PROFILE=ykk aws ec2 describe-volumes \
    --volume-ids vol-xxxxxxxxx \
    --query "Volumes[0].{State:State,AZ:AvailabilityZone}"
```

**Ensure:** Instance and volume are in the **same Availability Zone**!

### Out of Memory

Reduce batch size in tfvars:
```hcl
batch_size_per_gpu = 8  # Down from 10
```

---

## Cleanup

### Development Instance

```bash
just tf-destroy
just ssh-config-remove
```

### S3 Training

```bash
# Destroy instance (S3 data persists)
just train-s3-stop

# Optional: Delete S3 data
bucket=$(cd infra/services/training-s3 && AWS_PROFILE=ykk terraform output -raw s3_bucket)
AWS_PROFILE=ykk aws s3 rm "s3://$bucket" --recursive
```

### EBS Training

```bash
# Stop instance only (EBS persists)
just train-ebs-stop

# Destroy everything (DELETES ALL DATA!)
just train-ebs-destroy
```

---

## Architecture Diagrams

### S3-Centric

```
┌─────────────────────────────────────────┐
│               S3 Bucket                  │
│  ├── datasets/cached.tar.gz             │
│  ├── checkpoints/{run}/                 │
│  └── results/{run}/                     │
└────────────────────▲────────────────────┘
                     │ sync
┌────────────────────┴────────────────────┐
│         Ephemeral Spot Instance          │
│  Boot → Setup → Train → Complete → Term  │
└─────────────────────────────────────────┘
```

### EBS-Centric

```
┌─────────────────────────────────────────┐
│         Persistent EBS Volume            │
│  ├── .training-state                    │
│  ├── repo/                              │
│  ├── datasets/                          │
│  └── results/                           │
└────────────────────▲────────────────────┘
                     │ attach
┌────────────────────┴────────────────────┐
│      Persistent Spot Instance            │
│  Boot → Attach → Check State → Resume    │
│         ↑                                │
│         └── Auto-restart on interruption │
└─────────────────────────────────────────┘
```
