#!/bin/bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
# VecFormer Training - S3-Centric Setup
# ════════════════════════════════════════════════════════════════

exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting user-data script at $(date)"

# ── Variables from Terraform ───────────────────────────────────
S3_BUCKET="${s3_bucket}"
SNS_TOPIC_ARN="${sns_topic_arn}"
REGION="${region}"
RUN_ID="${run_id}"
GIT_REPO="${git_repo}"
GIT_BRANCH="${git_branch}"
SSH_PUBLIC_KEY="${ssh_public_key}"
GIT_USER_NAME="${git_user_name}"
GIT_USER_EMAIL="${git_user_email}"
TRAINING_CONFIG="${training_config}"
CHECKPOINT_EPOCHS="${checkpoint_epochs}"
NUM_EPOCHS="${num_epochs}"
BATCH_SIZE_PER_GPU="${batch_size_per_gpu}"
NUM_GPUS="${num_gpus}"

export HOME=/home/ubuntu
export USER=ubuntu
WORK_DIR="/home/ubuntu/training"
REPO_DIR="$WORK_DIR/VecFormer"
DATA_DIR="$WORK_DIR/data"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
RESULTS_DIR="$WORK_DIR/results"

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

self_terminate() {
    echo "Self-terminating instance..."
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
}

# ── SSH Setup ──────────────────────────────────────────────────

if [ -n "$SSH_PUBLIC_KEY" ]; then
    mkdir -p /home/ubuntu/.ssh
    echo "$SSH_PUBLIC_KEY" >> /home/ubuntu/.ssh/authorized_keys
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh
    chmod 700 /home/ubuntu/.ssh
    chmod 600 /home/ubuntu/.ssh/authorized_keys
    echo "SSH public key installed"
fi

# ── Git Configuration ──────────────────────────────────────────

if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    sudo -u ubuntu git config --global user.name "$GIT_USER_NAME"
    sudo -u ubuntu git config --global user.email "$GIT_USER_EMAIL"
    sudo -u ubuntu git config --global init.defaultBranch main
    echo "Git configured for $GIT_USER_NAME <$GIT_USER_EMAIL>"
fi

# ── System Setup ───────────────────────────────────────────────

echo "Installing system packages..."
apt-get update -qq
apt-get install -y nvtop jq

# ── Pixi Installation ──────────────────────────────────────────

echo "Installing Pixi..."
sudo -u ubuntu bash -c 'curl -fsSL https://pixi.sh/install.sh | bash'
export PATH="/home/ubuntu/.pixi/bin:$PATH"

# ── Directory Setup ────────────────────────────────────────────

mkdir -p "$WORK_DIR" "$DATA_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"
chown -R ubuntu:ubuntu "$WORK_DIR"

# ── Clone Repository ───────────────────────────────────────────

echo "Cloning repository..."
sudo -u ubuntu git clone --branch "$GIT_BRANCH" "$GIT_REPO" "$REPO_DIR"

# ── Install Dependencies ───────────────────────────────────────

echo "Installing dependencies with Pixi..."
cd "$REPO_DIR"
sudo -u ubuntu /home/ubuntu/.pixi/bin/pixi install

# ── Download/Cache Dataset ─────────────────────────────────────

echo "Checking for cached dataset in S3..."
if aws s3 ls "s3://$S3_BUCKET/datasets/FloorPlanCAD-cached.tar.gz" --region "$REGION" 2>/dev/null; then
    echo "Downloading cached dataset from S3..."
    aws s3 cp "s3://$S3_BUCKET/datasets/FloorPlanCAD-cached.tar.gz" "$DATA_DIR/" --region "$REGION"
    cd "$DATA_DIR"
    tar -xzf FloorPlanCAD-cached.tar.gz
    rm FloorPlanCAD-cached.tar.gz
    echo "Dataset downloaded and extracted"
else
    echo "No cached dataset found, downloading and preprocessing..."
    cd "$REPO_DIR"

    # Download raw data
    sudo -u ubuntu /home/ubuntu/.pixi/bin/pixi run python scripts/download_data.py

    # Preprocess
    sudo -u ubuntu /home/ubuntu/.pixi/bin/pixi run python scripts/preprocess_floorplancad.py

    # Cache to S3 for future runs
    echo "Uploading cached dataset to S3..."
    cd "$REPO_DIR/datasets"
    tar -czf FloorPlanCAD-cached.tar.gz FloorPlanCAD-sampled-as-line-jsons/
    aws s3 cp FloorPlanCAD-cached.tar.gz "s3://$S3_BUCKET/datasets/" --region "$REGION"
    rm FloorPlanCAD-cached.tar.gz
    echo "Dataset cached to S3"
fi

# ── Check for Existing Checkpoint ──────────────────────────────

RESUME_CHECKPOINT=""
LATEST_CHECKPOINT=$(aws s3 ls "s3://$S3_BUCKET/checkpoints/$RUN_ID/" --region "$REGION" 2>/dev/null | grep "checkpoint-" | sort -V | tail -1 | awk '{print $4}' || echo "")

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT, downloading..."
    aws s3 sync "s3://$S3_BUCKET/checkpoints/$RUN_ID/$LATEST_CHECKPOINT" "$CHECKPOINT_DIR/$LATEST_CHECKPOINT" --region "$REGION"
    RESUME_CHECKPOINT="$CHECKPOINT_DIR/$LATEST_CHECKPOINT"
    echo "Checkpoint downloaded, will resume from $RESUME_CHECKPOINT"
fi

# ── Create Training Script ─────────────────────────────────────

cat > "$WORK_DIR/run-training.sh" << 'TRAINING_SCRIPT'
#!/bin/bash
set -euo pipefail

export PATH="/home/ubuntu/.pixi/bin:$PATH"
cd /home/ubuntu/training/VecFormer

S3_BUCKET="__S3_BUCKET__"
SNS_TOPIC_ARN="__SNS_TOPIC_ARN__"
REGION="__REGION__"
RUN_ID="__RUN_ID__"
CHECKPOINT_DIR="__CHECKPOINT_DIR__"
RESULTS_DIR="__RESULTS_DIR__"
RESUME_CHECKPOINT="__RESUME_CHECKPOINT__"
CHECKPOINT_EPOCHS="__CHECKPOINT_EPOCHS__"
NUM_EPOCHS="__NUM_EPOCHS__"
BATCH_SIZE_PER_GPU="__BATCH_SIZE_PER_GPU__"
NUM_GPUS="__NUM_GPUS__"

# Calculate steps
DATASET_SIZE=6960
EFFECTIVE_BATCH=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH))
SAVE_STEPS=$((STEPS_PER_EPOCH * CHECKPOINT_EPOCHS))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))

echo "Training configuration:"
echo "  Batch per GPU: $BATCH_SIZE_PER_GPU"
echo "  Effective batch: $EFFECTIVE_BATCH"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Save every: $SAVE_STEPS steps ($CHECKPOINT_EPOCHS epochs)"
echo "  Total steps: $TOTAL_STEPS"

# Build training command
TRAIN_CMD="pixi run bash -c 'export PYTHONPATH=\$(pwd):\$PYTHONPATH && torchrun --nproc_per_node=$NUM_GPUS launch.py \
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
    --output_dir $RESULTS_DIR'"

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
fi

# Start checkpoint sync in background
(
    while true; do
        sleep 300  # Sync every 5 minutes
        if [ -d "$RESULTS_DIR" ]; then
            aws s3 sync "$RESULTS_DIR" "s3://$S3_BUCKET/checkpoints/$RUN_ID/" --region "$REGION" --exclude "*" --include "checkpoint-*" 2>/dev/null || true
        fi
    done
) &
SYNC_PID=$!

# Run training
echo "Starting training at $(date)"
eval $TRAIN_CMD
TRAIN_EXIT_CODE=$?

# Stop sync process
kill $SYNC_PID 2>/dev/null || true

# Final sync
echo "Final sync of results to S3..."
aws s3 sync "$RESULTS_DIR" "s3://$S3_BUCKET/results/$RUN_ID/" --region "$REGION"

# Notify completion
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    aws sns publish \
        --region "$REGION" \
        --topic-arn "$SNS_TOPIC_ARN" \
        --subject "VecFormer Training Complete: $RUN_ID" \
        --message "Training completed successfully.\n\nResults: s3://$S3_BUCKET/results/$RUN_ID/\n\nDownload: aws s3 sync s3://$S3_BUCKET/results/$RUN_ID/ ./results/$RUN_ID/"
else
    aws sns publish \
        --region "$REGION" \
        --topic-arn "$SNS_TOPIC_ARN" \
        --subject "VecFormer Training FAILED: $RUN_ID" \
        --message "Training failed with exit code $TRAIN_EXIT_CODE.\n\nCheck logs on instance or CloudWatch."
fi

echo "Training finished at $(date)"
TRAINING_SCRIPT

# Replace placeholders
sed -i "s|__S3_BUCKET__|$S3_BUCKET|g" "$WORK_DIR/run-training.sh"
sed -i "s|__SNS_TOPIC_ARN__|$SNS_TOPIC_ARN|g" "$WORK_DIR/run-training.sh"
sed -i "s|__REGION__|$REGION|g" "$WORK_DIR/run-training.sh"
sed -i "s|__RUN_ID__|$RUN_ID|g" "$WORK_DIR/run-training.sh"
sed -i "s|__CHECKPOINT_DIR__|$CHECKPOINT_DIR|g" "$WORK_DIR/run-training.sh"
sed -i "s|__RESULTS_DIR__|$RESULTS_DIR|g" "$WORK_DIR/run-training.sh"
sed -i "s|__RESUME_CHECKPOINT__|$RESUME_CHECKPOINT|g" "$WORK_DIR/run-training.sh"
sed -i "s|__CHECKPOINT_EPOCHS__|$CHECKPOINT_EPOCHS|g" "$WORK_DIR/run-training.sh"
sed -i "s|__NUM_EPOCHS__|$NUM_EPOCHS|g" "$WORK_DIR/run-training.sh"
sed -i "s|__BATCH_SIZE_PER_GPU__|$BATCH_SIZE_PER_GPU|g" "$WORK_DIR/run-training.sh"
sed -i "s|__NUM_GPUS__|$NUM_GPUS|g" "$WORK_DIR/run-training.sh"

chmod +x "$WORK_DIR/run-training.sh"
chown ubuntu:ubuntu "$WORK_DIR/run-training.sh"

# ── Create Systemd Service ─────────────────────────────────────

cat > /etc/systemd/system/vecformer-training.service << EOF
[Unit]
Description=VecFormer Training
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$WORK_DIR
ExecStart=$WORK_DIR/run-training.sh
ExecStopPost=/bin/bash -c 'shutdown -h now'
StandardOutput=append:/var/log/vecformer-training.log
StandardError=append:/var/log/vecformer-training.log
Restart=no

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vecformer-training

# ── Create Spot Interruption Handler ───────────────────────────

cat > /usr/local/bin/spot-interrupt-handler.sh << 'INTERRUPT_SCRIPT'
#!/bin/bash
# Poll for spot interruption notice and handle gracefully

while true; do
    # Check for interruption notice
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://169.254.169.254/latest/meta-data/spot/instance-action)

    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "Spot interruption notice received at $(date)"

        # Get action details
        ACTION=$(curl -s http://169.254.169.254/latest/meta-data/spot/instance-action)
        echo "Action: $ACTION"

        # Trigger graceful shutdown of training
        systemctl stop vecformer-training

        # Emergency sync to S3
        aws s3 sync /home/ubuntu/training/results "s3://__S3_BUCKET__/checkpoints/__RUN_ID__/" --region __REGION__ || true

        # Notify
        aws sns publish \
            --region "__REGION__" \
            --topic-arn "__SNS_TOPIC_ARN__" \
            --subject "VecFormer Training Interrupted: __RUN_ID__" \
            --message "Spot instance interrupted. Checkpoint saved to S3. Re-run terraform apply to resume."

        exit 0
    fi

    sleep 5
done
INTERRUPT_SCRIPT

sed -i "s|__S3_BUCKET__|$S3_BUCKET|g" /usr/local/bin/spot-interrupt-handler.sh
sed -i "s|__SNS_TOPIC_ARN__|$SNS_TOPIC_ARN|g" /usr/local/bin/spot-interrupt-handler.sh
sed -i "s|__REGION__|$REGION|g" /usr/local/bin/spot-interrupt-handler.sh
sed -i "s|__RUN_ID__|$RUN_ID|g" /usr/local/bin/spot-interrupt-handler.sh
chmod +x /usr/local/bin/spot-interrupt-handler.sh

# Start interrupt handler as a service
cat > /etc/systemd/system/spot-interrupt-handler.service << EOF
[Unit]
Description=Spot Interruption Handler
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/spot-interrupt-handler.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable spot-interrupt-handler
systemctl start spot-interrupt-handler

# ── Start Training ─────────────────────────────────────────────

echo "Starting training service..."
notify "VecFormer Training Started: $RUN_ID" "Training started on $(hostname) at $(date).\n\nInstance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)\nGPUs: $NUM_GPUS\nBatch/GPU: $BATCH_SIZE_PER_GPU\nEpochs: $NUM_EPOCHS"

systemctl start vecformer-training

echo "User-data script completed at $(date)"
