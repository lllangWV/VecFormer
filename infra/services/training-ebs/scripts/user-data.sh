#!/bin/bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
# VecFormer Training - EBS-Centric Setup
# Runs on EVERY boot (including after spot interruption recovery)
# ════════════════════════════════════════════════════════════════

exec > >(tee -a /var/log/user-data.log) 2>&1
echo "=========================================="
echo "Starting user-data script at $(date)"
echo "=========================================="

# ── Variables from Terraform ───────────────────────────────────
EBS_VOLUME_ID="${ebs_volume_id}"
SNS_TOPIC_ARN="${sns_topic_arn}"
REGION="${region}"
RUN_ID="${run_id}"
GIT_REPO="${git_repo}"
GIT_BRANCH="${git_branch}"
SSH_PUBLIC_KEY="${ssh_public_key}"
GIT_USER_NAME="${git_user_name}"
GIT_USER_EMAIL="${git_user_email}"
CHECKPOINT_EPOCHS="${checkpoint_epochs}"
NUM_EPOCHS="${num_epochs}"
BATCH_SIZE_PER_GPU="${batch_size_per_gpu}"
NUM_GPUS="${num_gpus}"

export HOME=/home/ubuntu
export USER=ubuntu
DATA_MOUNT="/data"
STATE_FILE="$DATA_MOUNT/.training-state"
REPO_DIR="$DATA_MOUNT/repo/VecFormer"
RESULTS_DIR="$DATA_MOUNT/results/$RUN_ID"
CHECKPOINT_DIR="$DATA_MOUNT/checkpoints"

# ── Helper Functions ───────────────────────────────────────────

notify() {
    local subject="$1"
    local message="$2"
    aws sns publish \
        --region "$REGION" \
        --topic-arn "$SNS_TOPIC_ARN" \
        --subject "$subject" \
        --message "$message" || true
}

get_instance_id() {
    curl -s http://169.254.169.254/latest/meta-data/instance-id
}

self_terminate() {
    echo "Training complete. Self-terminating instance..."
    echo "complete" > "$STATE_FILE"
    INSTANCE_ID=$(get_instance_id)

    # Cancel the persistent spot request first
    SPOT_REQUEST_ID=$(aws ec2 describe-spot-instance-requests \
        --region "$REGION" \
        --filters "Name=instance-id,Values=$INSTANCE_ID" \
        --query "SpotInstanceRequests[0].SpotInstanceRequestId" \
        --output text)

    if [ "$SPOT_REQUEST_ID" != "None" ] && [ -n "$SPOT_REQUEST_ID" ]; then
        aws ec2 cancel-spot-instance-requests \
            --region "$REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" || true
    fi

    # Terminate instance
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
}

# ── SSH Setup (always run) ─────────────────────────────────────

if [ -n "$SSH_PUBLIC_KEY" ]; then
    mkdir -p /home/ubuntu/.ssh
    grep -qxF "$SSH_PUBLIC_KEY" /home/ubuntu/.ssh/authorized_keys 2>/dev/null || \
        echo "$SSH_PUBLIC_KEY" >> /home/ubuntu/.ssh/authorized_keys
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh
    chmod 700 /home/ubuntu/.ssh
    chmod 600 /home/ubuntu/.ssh/authorized_keys
fi

# ── Git Configuration ──────────────────────────────────────────

if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    sudo -u ubuntu git config --global user.name "$GIT_USER_NAME"
    sudo -u ubuntu git config --global user.email "$GIT_USER_EMAIL"
fi

# ── Attach and Mount EBS Volume ────────────────────────────────

echo "Attaching EBS volume $EBS_VOLUME_ID..."
INSTANCE_ID=$(get_instance_id)

# Check if already attached
ATTACHED=$(aws ec2 describe-volumes \
    --region "$REGION" \
    --volume-ids "$EBS_VOLUME_ID" \
    --query "Volumes[0].Attachments[?InstanceId=='$INSTANCE_ID'].State" \
    --output text)

if [ "$ATTACHED" != "attached" ]; then
    # Wait for volume to be available
    aws ec2 wait volume-available --region "$REGION" --volume-ids "$EBS_VOLUME_ID" || true

    # Attach volume
    aws ec2 attach-volume \
        --region "$REGION" \
        --volume-id "$EBS_VOLUME_ID" \
        --instance-id "$INSTANCE_ID" \
        --device /dev/xvdf

    # Wait for attachment
    sleep 10
    aws ec2 wait volume-in-use --region "$REGION" --volume-ids "$EBS_VOLUME_ID"
    sleep 5
fi

# Find the actual device name (NVMe devices get renamed)
DEVICE=""
for dev in /dev/nvme1n1 /dev/xvdf /dev/sdf; do
    if [ -b "$dev" ]; then
        DEVICE="$dev"
        break
    fi
done

if [ -z "$DEVICE" ]; then
    echo "ERROR: Could not find attached EBS device"
    exit 1
fi

echo "EBS device: $DEVICE"

# Check if filesystem exists
if ! blkid "$DEVICE" | grep -q ext4; then
    echo "Creating ext4 filesystem on $DEVICE..."
    mkfs.ext4 "$DEVICE"
fi

# Mount
mkdir -p "$DATA_MOUNT"
if ! mountpoint -q "$DATA_MOUNT"; then
    mount "$DEVICE" "$DATA_MOUNT"
fi

# Add to fstab for persistence across reboots
grep -qxF "$DEVICE $DATA_MOUNT ext4 defaults,nofail 0 2" /etc/fstab || \
    echo "$DEVICE $DATA_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab

echo "EBS volume mounted at $DATA_MOUNT"

# ── Check Training State ───────────────────────────────────────

STATE=$(cat "$STATE_FILE" 2>/dev/null || echo "init")
echo "Current training state: $STATE"

case "$STATE" in
    "complete")
        echo "Training already complete. Keeping instance running for result retrieval."
        notify "VecFormer Instance Ready: $RUN_ID" \
            "Instance is running with completed training results.\n\nSSH: ssh ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)\nResults: $RESULTS_DIR"
        exit 0
        ;;
    "running")
        echo "Resuming training from checkpoint..."
        # Will continue to training section below
        ;;
    "init"|*)
        echo "Initializing fresh training run..."
        # Will continue to initialization section below
        ;;
esac

# ── First-Time Initialization ──────────────────────────────────

if [ "$STATE" = "init" ] || [ ! -d "$REPO_DIR" ]; then
    echo "Performing first-time setup..."

    # Install system packages
    apt-get update -qq
    apt-get install -y nvtop jq

    # Install Pixi
    if [ ! -f /home/ubuntu/.pixi/bin/pixi ]; then
        sudo -u ubuntu bash -c 'curl -fsSL https://pixi.sh/install.sh | bash'
    fi

    # Create directories
    mkdir -p "$DATA_MOUNT/repo" "$DATA_MOUNT/datasets" "$CHECKPOINT_DIR" "$DATA_MOUNT/results"
    chown -R ubuntu:ubuntu "$DATA_MOUNT"

    # Clone repository
    if [ ! -d "$REPO_DIR" ]; then
        sudo -u ubuntu git clone --branch "$GIT_BRANCH" "$GIT_REPO" "$REPO_DIR"
    fi

    # Install dependencies
    cd "$REPO_DIR"
    sudo -u ubuntu /home/ubuntu/.pixi/bin/pixi install

    # Download and preprocess dataset
    if [ ! -d "$DATA_MOUNT/datasets/FloorPlanCAD-sampled-as-line-jsons" ]; then
        echo "Downloading and preprocessing dataset..."
        cd "$REPO_DIR"
        sudo -u ubuntu /home/ubuntu/.pixi/bin/pixi run python scripts/download_data.py
        sudo -u ubuntu /home/ubuntu/.pixi/bin/pixi run python scripts/preprocess_floorplancad.py

        # Move to data volume
        mv "$REPO_DIR/datasets/"* "$DATA_MOUNT/datasets/" || true

        # Create symlink
        ln -sf "$DATA_MOUNT/datasets" "$REPO_DIR/datasets"
    fi

    # Create results directory
    mkdir -p "$RESULTS_DIR"
    chown -R ubuntu:ubuntu "$RESULTS_DIR"
fi

# Ensure symlinks exist
ln -sf "$DATA_MOUNT/datasets" "$REPO_DIR/datasets" 2>/dev/null || true

# ── Create Training Script ─────────────────────────────────────

cat > "$DATA_MOUNT/run-training.sh" << 'TRAINING_SCRIPT'
#!/bin/bash
set -euo pipefail

export PATH="/home/ubuntu/.pixi/bin:$PATH"

RUN_ID="__RUN_ID__"
REPO_DIR="__REPO_DIR__"
RESULTS_DIR="__RESULTS_DIR__"
STATE_FILE="__STATE_FILE__"
CHECKPOINT_EPOCHS="__CHECKPOINT_EPOCHS__"
NUM_EPOCHS="__NUM_EPOCHS__"
BATCH_SIZE_PER_GPU="__BATCH_SIZE_PER_GPU__"
NUM_GPUS="__NUM_GPUS__"
SNS_TOPIC_ARN="__SNS_TOPIC_ARN__"
REGION="__REGION__"

cd "$REPO_DIR"

# Calculate steps
DATASET_SIZE=6960
EFFECTIVE_BATCH=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH))
SAVE_STEPS=$((STEPS_PER_EPOCH * CHECKPOINT_EPOCHS))

echo "Training configuration:"
echo "  Run ID: $RUN_ID"
echo "  Batch per GPU: $BATCH_SIZE_PER_GPU"
echo "  Effective batch: $EFFECTIVE_BATCH"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Save every: $SAVE_STEPS steps ($CHECKPOINT_EPOCHS epochs)"
echo "  Total epochs: $NUM_EPOCHS"

# Check for existing checkpoint
RESUME_ARG=""
LATEST_CHECKPOINT=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
fi

# Set state to running
echo "running" > "$STATE_FILE"

# Run training
echo "Starting training at $(date)"
pixi run bash -c "export PYTHONPATH=\$(pwd):\$PYTHONPATH && torchrun --nproc_per_node=$NUM_GPUS launch.py \
    --launch_mode train \
    --config_path configs/vecformer.yaml \
    --model_args_path configs/model/vecformer.yaml \
    --data_args_path configs/data/floorplancad.yaml \
    --run_name $RUN_ID \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --num_train_epochs $NUM_EPOCHS \
    --eval_strategy epoch \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --logging_steps 10 \
    --dataloader_num_workers 12 \
    --bf16 true \
    --output_dir $RESULTS_DIR \
    $RESUME_ARG"

TRAIN_EXIT_CODE=$?

echo "Training finished at $(date) with exit code $TRAIN_EXIT_CODE"

# Notify completion
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "complete" > "$STATE_FILE"
    aws sns publish \
        --region "$REGION" \
        --topic-arn "$SNS_TOPIC_ARN" \
        --subject "VecFormer Training Complete: $RUN_ID" \
        --message "Training completed successfully on $(hostname).\n\nResults location: $RESULTS_DIR\n\nSSH to retrieve: ssh ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
else
    aws sns publish \
        --region "$REGION" \
        --topic-arn "$SNS_TOPIC_ARN" \
        --subject "VecFormer Training FAILED: $RUN_ID" \
        --message "Training failed with exit code $TRAIN_EXIT_CODE.\n\nCheck logs: /var/log/vecformer-training.log"
fi
TRAINING_SCRIPT

# Replace placeholders
sed -i "s|__RUN_ID__|$RUN_ID|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__REPO_DIR__|$REPO_DIR|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__RESULTS_DIR__|$RESULTS_DIR|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__STATE_FILE__|$STATE_FILE|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__CHECKPOINT_EPOCHS__|$CHECKPOINT_EPOCHS|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__NUM_EPOCHS__|$NUM_EPOCHS|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__BATCH_SIZE_PER_GPU__|$BATCH_SIZE_PER_GPU|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__NUM_GPUS__|$NUM_GPUS|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__SNS_TOPIC_ARN__|$SNS_TOPIC_ARN|g" "$DATA_MOUNT/run-training.sh"
sed -i "s|__REGION__|$REGION|g" "$DATA_MOUNT/run-training.sh"

chmod +x "$DATA_MOUNT/run-training.sh"
chown ubuntu:ubuntu "$DATA_MOUNT/run-training.sh"

# ── Create Systemd Service ─────────────────────────────────────

cat > /etc/systemd/system/vecformer-training.service << EOF
[Unit]
Description=VecFormer Training
After=network.target local-fs.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$DATA_MOUNT
ExecStart=$DATA_MOUNT/run-training.sh
StandardOutput=append:/var/log/vecformer-training.log
StandardError=append:/var/log/vecformer-training.log
Restart=no

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vecformer-training

# ── Start Training ─────────────────────────────────────────────

if [ "$STATE" = "running" ]; then
    notify "VecFormer Training Resumed: $RUN_ID" \
        "Training resumed after spot interruption on $(hostname) at $(date).\n\nInstance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
else
    notify "VecFormer Training Started: $RUN_ID" \
        "Training started on $(hostname) at $(date).\n\nInstance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)\nGPUs: $NUM_GPUS\nBatch/GPU: $BATCH_SIZE_PER_GPU\nEpochs: $NUM_EPOCHS"
fi

# Mark as running before starting
echo "running" > "$STATE_FILE"

echo "Starting training service..."
systemctl start vecformer-training

echo "User-data script completed at $(date)"
