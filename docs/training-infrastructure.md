# VecFormer Training Infrastructure

This document describes how to use the automated training infrastructure for VecFormer.

## Overview

Two training options are available:

| Option | Description | Best For |
|--------|-------------|----------|
| **S3-Centric** | Ephemeral instance, S3 for persistence | Simplicity, cross-region flexibility |
| **EBS-Centric** | Persistent EBS volume, auto-resume | Automatic spot recovery, faster resume |

Both options:
- Use p5en.48xlarge (8× H200 GPUs) by default
- Auto-start training on boot
- Send email notifications on completion/failure
- Support checkpointing every 10 epochs
- Self-terminate when training completes

---

## Quick Start

### Option 1: S3-Centric Training

```bash
# 1. Initialize (first time only)
just train-s3-init

# 2. Update tfvars (set subnet_id, run_id)
vim infra/envs/us-east-2/training-s3.tfvars

# 3. Start training
just train-s3-start

# 4. Monitor
just train-s3-logs
just train-s3-gpu

# 5. Download results (after completion email)
just train-s3-results vecformer-500ep-h200

# 6. Cleanup
just train-s3-stop
```

### Option 3: EBS-Centric Training

```bash
# 1. Initialize (first time only)
just train-ebs-init

# 2. Update tfvars (set subnet_id, availability_zone, run_id)
vim infra/envs/us-east-2/training-ebs.tfvars

# 3. Start training
just train-ebs-start

# 4. Monitor
just train-ebs-logs
just train-ebs-gpu
just train-ebs-state

# 5. Download results (after completion email)
just train-ebs-results vecformer-500ep-h200-ebs

# 6. Cleanup (WARNING: deletes all data!)
just train-ebs-destroy
```

---

## Configuration

### Required Updates Before First Run

Edit the tfvars file for your chosen option:

```hcl
# infra/envs/us-east-2/training-s3.tfvars (or training-ebs.tfvars)

# REQUIRED: Set your subnet
subnet_id = "subnet-xxxxxxxxx"  # Choose a subnet in your VPC

# For EBS option only:
availability_zone = "us-east-2a"  # Must match subnet's AZ

# REQUIRED: Unique run identifier
run_id = "vecformer-experiment-1"  # Change for each run
```

### Finding Your Subnet ID

```bash
# List subnets in your VPC
AWS_PROFILE=ykk aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=vpc-097a3bf6bd387ae0b" \
    --query "Subnets[*].{SubnetId:SubnetId,AZ:AvailabilityZone,CIDR:CidrBlock}" \
    --output table
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instance_type` | p5en.48xlarge | 8× H200 GPUs |
| `num_epochs` | 500 | Total training epochs |
| `checkpoint_epochs` | 10 | Save every N epochs |
| `batch_size_per_gpu` | 10 | Batch size per GPU |
| `num_gpus` | 8 | GPUs to use |

---

## Monitoring

### Check Training Logs

```bash
# Last 100 lines of training log
just train-s3-logs   # or train-ebs-logs

# SSH for interactive monitoring
just train-s3-ssh    # or train-ebs-ssh

# On instance:
tail -f /var/log/vecformer-training.log
nvtop  # GPU monitoring
```

### Check Training State (EBS only)

```bash
just train-ebs-state
# Returns: init | running | complete
```

### GPU Status

```bash
just train-s3-gpu  # or train-ebs-gpu
```

---

## Spot Interruption Handling

### S3 Option

1. Spot interruption notice received (2-min warning)
2. Emergency checkpoint saved
3. Checkpoint synced to S3
4. Email notification sent
5. Instance terminated

**To resume:** Run `just train-s3-start` again. Training resumes from last checkpoint.

### EBS Option

1. Spot interruption → Instance stopped (not terminated)
2. Persistent spot request automatically restarts instance
3. On boot: EBS volume re-attached, training auto-resumes
4. Email notification sent

**No manual intervention needed** - training resumes automatically!

---

## Cost Estimates

| Instance | Spot Price | 500 Epochs | Est. Cost |
|----------|------------|------------|-----------|
| p5en.48xlarge | ~$11/hr | ~1-1.5 hrs | ~$12-17 |

### Cost Breakdown

- **Compute:** ~$11-15 (spot instance)
- **S3 Storage:** ~$0.50/month (checkpoints + results)
- **EBS Volume:** ~$16/month for 200GB gp3 (if kept)
- **SNS:** Free tier (first 1M publishes)

---

## Troubleshooting

### Training Not Starting

```bash
# Check user-data log
just train-s3-ssh  # or train-ebs-ssh
sudo cat /var/log/user-data.log

# Check systemd service
sudo systemctl status vecformer-training
sudo journalctl -u vecformer-training -f
```

### EBS Volume Not Attaching

```bash
# Check volume state
AWS_PROFILE=ykk aws ec2 describe-volumes \
    --volume-ids vol-xxxxxxxxx \
    --query "Volumes[0].{State:State,Attachments:Attachments}"
```

### Spot Request Not Fulfilled

```bash
# Check spot request status
just train-s3-output  # or train-ebs-output
# Look at spot_request_id, then:

AWS_PROFILE=ykk aws ec2 describe-spot-instance-requests \
    --spot-instance-request-ids sir-xxxxxxxxx
```

### Out of Memory

Reduce batch size in tfvars:
```hcl
batch_size_per_gpu = 8  # Down from 10
```

---

## Architecture Diagrams

### S3-Centric (Option 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                         S3 Bucket                                │
│  ├── datasets/FloorPlanCAD-cached.tar.gz                        │
│  ├── checkpoints/{run-id}/checkpoint-*/                         │
│  └── results/{run-id}/                                          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ sync every 5 min
                              │
┌─────────────────────────────────────────────────────────────────┐
│              Ephemeral Spot Instance                             │
│  - Downloads dataset from S3 on first run (cached after)       │
│  - Downloads checkpoint if resuming                             │
│  - Syncs checkpoints to S3 periodically                        │
│  - Self-terminates when complete                                │
└─────────────────────────────────────────────────────────────────┘
```

### EBS-Centric (Option 3)

```
┌─────────────────────────────────────────────────────────────────┐
│              Persistent EBS Volume (200GB)                       │
│  ├── .training-state          (init|running|complete)           │
│  ├── repo/VecFormer/          (cloned once)                     │
│  ├── datasets/                (cached once)                     │
│  ├── checkpoints/             (training checkpoints)            │
│  └── results/{run-id}/        (final outputs)                   │
└─────────────────────────────────────────────────────────────────┘
         │ attach/detach
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Persistent Spot Instance                            │
│  - Attaches EBS on every boot                                   │
│  - Checks .training-state                                       │
│  - Resumes training automatically                               │
│  - Auto-restarts after spot interruption                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
infra/
├── modules/
│   ├── s3-training-bucket/    # S3 bucket for artifacts
│   ├── persistent-ebs/        # Persistent EBS volume
│   └── sns-notifications/     # Email notifications
│
├── services/
│   ├── training-s3/           # Option 1 implementation
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── scripts/
│   │       └── user-data.sh
│   │
│   └── training-ebs/          # Option 3 implementation
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       └── scripts/
│           └── user-data.sh
│
└── envs/
    └── us-east-2/
        ├── training-s3.tfvars
        └── training-ebs.tfvars
```

---

## Cleanup

### S3 Option

```bash
# Destroy instance (results stay in S3)
just train-s3-stop

# Optional: Delete S3 bucket contents
AWS_PROFILE=ykk aws s3 rm s3://vecformer-training-ACCOUNT_ID/ --recursive
```

### EBS Option

```bash
# Stop instance (EBS persists, can resume later)
just train-ebs-stop

# Or destroy everything (DELETES ALL DATA!)
just train-ebs-destroy
```
