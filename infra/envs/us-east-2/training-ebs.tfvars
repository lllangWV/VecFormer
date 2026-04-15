# Training infrastructure - EBS-centric option
# Persistent EBS volume with auto-resume on spot interruption

# ── AWS Configuration ──────────────────────────────────────────
aws_region        = "us-east-2"
vpc_id            = "vpc-097a3bf6bd387ae0b"
subnet_id         = "subnet-0123456789abcdef0"  # UPDATE: Choose a subnet in your VPC
availability_zone = "us-east-2a"                 # UPDATE: Must match subnet AZ

# ── Instance Configuration ─────────────────────────────────────
instance_type    = "p5en.48xlarge"  # 8x H200, $11/hr spot
spot_max_price   = "20.00"
root_volume_size = 100
ebs_volume_size  = 200  # Persistent data volume

# ── Access Configuration ───────────────────────────────────────
allowed_cidrs  = ["167.77.192.18/32", "71.182.199.107/32"]
ssh_public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKw0vPOhBiiGJiLomlfzCDNo3SEM0KCsktGY+4ccva+W lllangWV@gmail.com"
git_user_name  = "Logan Lang"
git_user_email = "lllangWV@gmail.com"

# ── Notifications ──────────────────────────────────────────────
notification_email = "logan.lang@xtechlab.ykkap.com"

# ── Training Configuration ─────────────────────────────────────
run_id             = "vecformer-500ep-h200-ebs"  # UPDATE: Unique run identifier
git_repo           = "https://github.com/lllangWV/VecFormer.git"
git_branch         = "main"
num_epochs         = 500
checkpoint_epochs  = 10
batch_size_per_gpu = 10   # H200 141GB VRAM allows larger batches
num_gpus           = 8
