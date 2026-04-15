# AWS EC2 Instance Options for VecFormer Training

**Date:** April 15, 2026  
**Region:** us-east-2  
**Purpose:** Compare GPU instance options for full 500-epoch VecFormer training

---

## Executive Summary

| Recommendation | Instance | GPUs | Est. Time | Est. Cost | Why |
|----------------|----------|------|-----------|-----------|-----|
| **Best Value** | p5en.48xlarge | 8× H200 | ~1-1.5 hrs | **~$12-17** | Cheapest spot + huge VRAM |
| **Budget Single-GPU** | g7e.4xlarge | 1× L40S | ~1.5 days | ~$35 | Low hourly, flexible |
| **Fastest** | p6-b200.48xlarge | 8× B200 | ~30-45 min | ~$15-23 | Cutting-edge speed |

**Key Insight:** The p5en (8× H200) at $11/hr spot is cheaper than p5 (8× H100) at $13/hr, while having 76% more VRAM and 43% more memory bandwidth. This is the clear winner for VecFormer training.

---

## 1. Spot Pricing (us-east-2, April 2026)

| Instance | GPUs | VRAM/GPU | Total VRAM | On-Demand | **Spot Price** |
|----------|------|----------|------------|-----------|----------------|
| g7e.2xlarge | 1× L40S | 48 GB | 48 GB | $2.66/hr | ~$0.80-1.00/hr |
| g7e.4xlarge | 1× L40S | 48 GB | 48 GB | $3.99/hr | ~$1.20-1.60/hr |
| g7e.8xlarge | 1× L40S | 48 GB | 48 GB | $5.27/hr | ~$1.60-2.10/hr |
| p4d.24xlarge | 8× A100 | 40 GB | 320 GB | $32.77/hr | ~$10/hr |
| p5.48xlarge | 8× H100 | 80 GB | 640 GB | $55.04/hr | ~$13/hr |
| **p5en.48xlarge** | 8× H200 | 141 GB | 1,128 GB | $63.30/hr | **~$11/hr** |
| p6-b200.48xlarge | 8× B200 | 180 GB | 1,440 GB | $113.93/hr | ~$30/hr |

**Notable:** p5en is cheaper than p5 on spot despite having better specs!

---

## 2. GPU Specifications Comparison

| Spec | L40S | A100 40GB | H100 80GB | H200 141GB | B200 180GB |
|------|------|-----------|-----------|------------|------------|
| Architecture | Ada Lovelace | Ampere | Hopper | Hopper | Blackwell |
| VRAM | 48 GB | 40 GB | 80 GB | 141 GB | 180 GB |
| Memory Type | GDDR6 | HBM2e | HBM3 | HBM3e | HBM3e |
| Memory BW | 864 GB/s | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8.0 TB/s |
| FP16 TFLOPS | 362 | 312 | 990 | 990 | ~1,800 |
| FP8 TFLOPS | 733 | N/A | 1,979 | 1,979 | ~4,000 |
| TDP | 350W | 400W | 700W | 700W | 1000W |
| Year | 2023 | 2020 | 2023 | 2024 | 2025 |

---

## 3. Why the Paper Used Batch Size 2 per GPU

The original VecFormer paper trained on **8× A100 40GB** with batch size 2 per GPU.

**This makes sense now:**
- A100 has only 40 GB VRAM
- VecFormer processes variable-length point clouds (376 to 63,344 points)
- Worst-case samples can spike memory significantly
- Batch 2 provides safety margin for outliers

**With more VRAM, we can use larger batches:**

| GPU | VRAM | Safe Batch/GPU | Effective Batch (8 GPUs) |
|-----|------|----------------|--------------------------|
| A100 40GB | 40 GB | 2 | 16 |
| H100 80GB | 80 GB | 4-6 | 32-48 |
| H200 141GB | 141 GB | 8-12 | 64-96 |
| B200 180GB | 180 GB | 12-16 | 96-128 |

**Larger batch = fewer steps per epoch = faster training**

---

## 4. Combined Speedup Analysis

Training speed depends on three factors:
1. **Compute speed** (TFLOPS)
2. **Memory bandwidth** (for attention operations)
3. **Batch size** (fewer steps with larger batches)

### Speedup Breakdown

| GPU | Compute vs A100 | Memory BW vs A100 | Batch Boost | **Combined Speedup** |
|-----|-----------------|-------------------|-------------|----------------------|
| A100 | 1× | 1× (2.0 TB/s) | 1× (batch 2) | **1×** (baseline) |
| H100 | ~2-3× | 1.7× (3.35 TB/s) | 2-3× (batch 4-6) | **~4-6×** |
| H200 | ~2-3× | 2.4× (4.8 TB/s) | 4-6× (batch 8-12) | **~8-12×** |
| B200 | ~4-5× | 4× (8.0 TB/s) | 6-8× (batch 12-16) | **~15-25×** |

### Steps per Epoch by Configuration

Dataset: 6,960 training samples

| Config | Batch/GPU | Effective Batch | Steps/Epoch |
|--------|-----------|-----------------|-------------|
| Paper (8× A100) | 2 | 16 | 435 |
| 8× H100 | 5 | 40 | 174 |
| 8× H200 | 10 | 80 | 87 |
| 8× B200 | 14 | 112 | 62 |

---

## 5. Training Time & Cost Estimates (500 Epochs)

### Single-GPU Options

| Instance | Spot Price | Batch | Steps/Epoch | Time | **Cost** |
|----------|------------|-------|-------------|------|----------|
| g7e.2xlarge | $0.90/hr | 16 | 435 | ~2.5 days | ~$54 |
| g7e.2xlarge | $0.90/hr | 64 | 109 | ~1.5 days | ~$32 |
| g7e.4xlarge* | $1.40/hr | 64 | 109 | ~1.2 days | ~$40 |
| g7e.4xlarge* | $1.40/hr | 128 | 55 | ~1 day | ~$34 |

*With data loading optimizations (16 workers, in-memory caching)

### Multi-GPU Options

| Instance | Spot | Batch/GPU | Eff. Batch | Est. Time | **Est. Cost** |
|----------|------|-----------|------------|-----------|---------------|
| 8× A100 (p4d) | $10/hr | 2 | 16 | ~8-10 hrs | ~$80-100 |
| 8× H100 (p5) | $13/hr | 5 | 40 | ~2-3 hrs | ~$26-39 |
| **8× H200 (p5en)** | **$11/hr** | 10 | 80 | **~1-1.5 hrs** | **~$12-17** |
| 8× B200 (p6) | $30/hr | 14 | 112 | ~30-45 min | ~$15-23 |

### Visual Cost Comparison

```
500-Epoch Training Cost (lower is better)
─────────────────────────────────────────
p5en (H200)  ████ $12-17         ← BEST VALUE
p6-b200      █████ $15-23        ← Fastest  
p5 (H100)    ████████ $26-39
g7e.4xlarge  ██████████ $34-40   ← Best single-GPU
g7e.2xlarge  ████████████████ $54
p4d (A100)   ██████████████████████████ $80-100
```

---

## 6. Detailed Recommendations

### Best Overall: p5en.48xlarge (8× H200)

**Why p5en is the clear winner:**

| Advantage | Details |
|-----------|---------|
| Cheapest spot | $11/hr (less than H100!) |
| Huge VRAM | 141 GB/GPU = batch 10+ per GPU |
| Fast memory | 4.8 TB/s (43% faster than H100) |
| Quick training | ~1-1.5 hours for 500 epochs |
| Low total cost | ~$12-17 total |

**Comparison to current setup (g7e.2xlarge):**
- 10× faster
- 3-4× cheaper total cost
- More reliable (shorter spot window = less interruption risk)

**Training configuration:**
```bash
torchrun --nproc_per_node=8 launch.py \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 12 \
    --bf16 true \
    --save_strategy steps \
    --save_steps 500
```

### Budget Single-GPU: g7e.4xlarge

**When to choose:**
- Running multiple experiments over days/weeks
- Need flexibility to start/stop
- Lower hourly commitment

**Upgrade from g7e.2xlarge because:**
- 16 vCPUs (vs 8) = 2× dataloader workers
- 128 GB RAM = full dataset in memory
- Only ~$0.50/hr more

### Fastest: p6-b200.48xlarge (8× B200)

**When to choose:**
- Time is critical
- Willing to pay premium
- Want cutting-edge hardware

**Caveats:**
- $30/hr (3× more than p5en)
- Only ~30-40% faster than p5en
- Newer = potentially less spot availability
- Currently limited to us-west-2

---

## 7. Implementation Guide

### Option A: Upgrade to p5en.48xlarge

**Step 1: Create new Terraform configuration**

Create `infra/services/embd-p5en/main.tf`:
```hcl
# Key differences from g7e:
instance_type  = "p5en.48xlarge"
spot_max_price = "20.00"  # Well above $11 spot for reliability

# Larger root volume for 8-GPU instance
root_volume_size = 500

# EFA networking for multi-GPU
# (requires additional security group rules)
```

**Step 2: Update training script for distributed**

```bash
#!/bin/bash
cd ~/Projects/VecFormer

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500

pixi run bash -c '
    export PYTHONPATH=$(pwd):$PYTHONPATH
    torchrun \
        --nproc_per_node=8 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        launch.py \
        --launch_mode train \
        --config_path configs/vecformer.yaml \
        --model_args_path configs/model/vecformer.yaml \
        --data_args_path configs/data/floorplancad.yaml \
        --run_name full-training-8gpu \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 10 \
        --num_train_epochs 500 \
        --eval_strategy epoch \
        --save_strategy steps \
        --save_steps 500 \
        --logging_steps 10 \
        --dataloader_num_workers 12 \
        --bf16 true \
        --output_dir outputs/full-training-8gpu
'
```

**Step 3: Set up checkpointing to S3**

For spot instance protection:
```bash
# In training script, sync checkpoints to S3
aws s3 sync outputs/full-training-8gpu s3://your-bucket/checkpoints/ \
    --exclude "*" --include "checkpoint-*"
```

### Option B: Stay with g7e.4xlarge (Single GPU)

**Step 1: Update tfvars**
```hcl
instance_type  = "g7e.4xlarge"
spot_max_price = "2.50"
```

**Step 2: Apply data loading optimizations**
```bash
# On instance: pre-cache dataset
pixi run python scripts/precache_dataset.py

# Training with optimizations
--per_device_train_batch_size 64 \
--dataloader_num_workers 16 \
--dataloader_pin_memory true
```

---

## 8. Risk Considerations

### Spot Interruption

| Instance | Interruption Risk | Mitigation |
|----------|-------------------|------------|
| g7e.* | Low (~2%) | Checkpoint every 1000 steps |
| p5 (H100) | Medium (~4%) | Checkpoint every 500 steps |
| p5en (H200) | Medium (~4%) | Checkpoint every 500 steps |
| p6-b200 | Higher (new hardware) | Use Capacity Blocks if available |

**Key insight:** Shorter training time = lower total interruption probability

- 2.5 days on g7e: ~5-10% chance of at least one interruption
- 1.5 hours on p5en: ~0.1% chance of interruption

### Batch Size vs Convergence

Larger batches may affect training dynamics:
- May need learning rate scaling (linear scaling rule)
- Monitor loss curves for first few epochs
- Paper used batch 16 effectively; batch 80 should be fine with LR adjustment

---

## 9. Quick Decision Flowchart

```
┌─────────────────────────────────────────┐
│ What's your priority?                    │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Lowest  │ │ Best    │ │ Fastest │
   │ Hourly  │ │ Total   │ │ Time    │
   │ Rate    │ │ Cost    │ │         │
   └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │g7e.4xl  │ │ p5en    │ │p6-b200  │
   │$1.40/hr │ │ $11/hr  │ │ $30/hr  │
   │~1.5 days│ │~1.5 hrs │ │~30 min  │
   │ ~$35    │ │ ~$15    │ │ ~$20    │
   └─────────┘ └─────────┘ └─────────┘
```

---

## 10. Summary Table

| Metric | g7e.2xl (current) | g7e.4xl | p5 (H100) | p5en (H200) | p6-b200 |
|--------|-------------------|---------|-----------|-------------|---------|
| GPUs | 1× L40S | 1× L40S | 8× H100 | 8× H200 | 8× B200 |
| VRAM | 48 GB | 48 GB | 640 GB | 1,128 GB | 1,440 GB |
| Spot Price | $0.90/hr | $1.40/hr | $13/hr | **$11/hr** | $30/hr |
| Batch Size | 16-64 | 16-128 | 40 | 80 | 112 |
| Time (500ep) | 2.5 days | 1.5 days | 2-3 hrs | **1-1.5 hrs** | 30-45 min |
| Total Cost | $54 | $35-40 | $26-39 | **$12-17** | $15-23 |
| Best For | Experiments | Budget | Production | **Best Value** | Speed |

---

## 11. Conclusion

**The p5en.48xlarge (8× H200) is the optimal choice for VecFormer training:**

1. **Cheapest total cost** (~$12-17 vs $54 current)
2. **Fastest practical option** (~1-1.5 hours)
3. **Lower spot price than H100** ($11/hr vs $13/hr)
4. **Massive VRAM** (141 GB/GPU) enables large batches
5. **Lower interruption risk** (short training window)

The counterintuitive spot pricing (H200 cheaper than H100) makes this a clear winner. The only reason to choose differently:
- **g7e.4xlarge**: If you need to run many short experiments over time
- **p6-b200**: If you need absolute fastest and have budget

---

## Sources

- [NVIDIA L40S Specifications](https://www.nvidia.com/en-us/data-center/l40s/)
- [NVIDIA H100 vs H200 Comparison](https://www.spheron.network/blog/nvidia-h100-vs-h200/)
- [NVIDIA B200 Specifications](https://www.runpod.io/articles/guides/nvidia-b200)
- [AWS EC2 P5 Instances](https://aws.amazon.com/ec2/instance-types/p5/)
- [AWS EC2 P5en Instances](https://instances.vantage.sh/aws/ec2/p5en.48xlarge)
- [AWS EC2 P6-B200 Instances](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6-b200-instances-powered-by-nvidia-blackwell-gpus-to-accelerate-ai-innovations/)
- [AWS EC2 Spot Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [H100 Rental Price Comparison](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [AWS GPU Price Cuts 2025](https://www.datacenterdynamics.com/en/news/aws-cuts-costs-for-h100-h200-and-a100-instances-by-up-to-45/)
