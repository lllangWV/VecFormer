# GPU Spot Instance Management

set dotenv-load := false

# List available recipes
default:
    @just --list

# ── Configuration ─────────────────────────────────────────────

# Environment: "us-east-2" (GPU spot instance)
# Override with: just --set env <name> <command>
env := "us-east-2"
tf_dir := "infra/envs/" + env
tf_vars := "-var-file=embd.tfvars"

# ── Terraform ──────────────────────────────────────────────────

# Show terraform plan
tf-plan:
    cd {{tf_dir}} && AWS_PROFILE=ykk terraform plan {{tf_vars}}

# Apply terraform (provision the instance)
tf-apply:
    cd {{tf_dir}} && AWS_PROFILE=ykk terraform apply {{tf_vars}}

# Destroy terraform resources
tf-destroy:
    cd {{tf_dir}} && AWS_PROFILE=ykk terraform destroy {{tf_vars}}

# Show terraform outputs (instance_id, public_ip, ssh_command)
tf-output:
    @cd {{tf_dir}} && AWS_PROFILE=ykk terraform output

# Initialize terraform
tf-init:
    cd {{tf_dir}} && AWS_PROFILE=ykk terraform init

# ── EC2 Instance Management ───────────────────────────────────
# All commands read instance ID and region from terraform output.

# Helper: get instance ID from terraform
_tf_id := "cd " + tf_dir + " && AWS_PROFILE=ykk terraform output -raw instance_id"

# Helper: get region from tfvars
_get_region := "grep -oP 'aws_region\\s*=\\s*\"\\K[^\"]+' " + tf_dir + "/embd.tfvars 2>/dev/null || echo 'us-east-2'"

# SSH into the instance via EC2 Instance Connect
ssh-instance:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$({{_get_region}})
    ip=$(AWS_PROFILE=ykk aws ec2 describe-instances --instance-ids "$id" --region "$region" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "Connecting to $ip..."
    AWS_PROFILE=ykk aws ec2-instance-connect send-ssh-public-key \
        --instance-id "$id" --instance-os-user ubuntu \
        --ssh-public-key file://~/.ssh/id_ed25519.pub --region "$region" > /dev/null
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip"

# Start the EC2 instance
start-instance:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$({{_get_region}})
    echo "Starting instance $id..."
    AWS_PROFILE=ykk aws ec2 start-instances --instance-ids "$id" --region "$region"

# Stop the EC2 instance
stop-instance:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$({{_get_region}})
    echo "Stopping instance $id..."
    AWS_PROFILE=ykk aws ec2 stop-instances --instance-ids "$id" --region "$region"

# Check the EC2 instance status
instance-status:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$({{_get_region}})
    AWS_PROFILE=ykk aws ec2 describe-instance-status \
        --instance-ids "$id" \
        --region "$region" \
        --include-all-instances \
        --query 'InstanceStatuses[0].InstanceState.Name' \
        --output text

# Show instance details (type, IP, state)
instance-info:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$({{_get_region}})
    AWS_PROFILE=ykk aws ec2 describe-instances \
        --instance-ids "$id" \
        --region "$region" \
        --query 'Reservations[0].Instances[0].{InstanceId:InstanceId,InstanceType:InstanceType,State:State.Name,PublicIP:PublicIpAddress,LaunchTime:LaunchTime}' \
        --output table

# Run nvidia-smi on the instance
nvidia-smi:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$({{_get_region}})
    ip=$(AWS_PROFILE=ykk aws ec2 describe-instances --instance-ids "$id" --region "$region" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    AWS_PROFILE=ykk aws ec2-instance-connect send-ssh-public-key \
        --instance-id "$id" --instance-os-user ubuntu \
        --ssh-public-key file://~/.ssh/id_ed25519.pub --region "$region" > /dev/null
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip" "nvidia-smi"

# ── SSH Config Management ─────────────────────────────────────

# Add "Host vecformer" to ~/.ssh/config
ssh-config-add:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$({{_get_region}})
    ip=$(AWS_PROFILE=ykk aws ec2 describe-instances --instance-ids "$id" --region "$region" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

    if [[ "$ip" == "None" || -z "$ip" ]]; then
        echo "Error: Instance has no public IP. Is it running?"
        exit 1
    fi

    config_file="$HOME/.ssh/config"

    # Remove existing vecformer entry if present
    if grep -q "^Host vecformer$" "$config_file" 2>/dev/null; then
        echo "Updating existing 'Host vecformer' entry..."
        # Remove old entry (from "Host vecformer" to next "Host" or EOF)
        awk '/^Host vecformer$/{skip=1; next} /^Host /{skip=0} !skip' "$config_file" > "$config_file.tmp"
        mv "$config_file.tmp" "$config_file"
    fi

    # Append new entry
    {
        echo ""
        echo "Host vecformer"
        echo "    HostName $ip"
        echo "    User ubuntu"
        echo "    IdentityFile $HOME/.ssh/id_ed25519"
        echo "    StrictHostKeyChecking no"
    } >> "$config_file"

    echo "Added 'Host vecformer' to ~/.ssh/config (IP: $ip)"
    echo "You can now use: ssh vecformer"

# Remove "Host vecformer" from ~/.ssh/config
ssh-config-remove:
    #!/usr/bin/env bash
    set -euo pipefail
    config_file="$HOME/.ssh/config"

    if ! grep -q "^Host vecformer$" "$config_file" 2>/dev/null; then
        echo "No 'Host vecformer' entry found in ~/.ssh/config"
        exit 0
    fi

    # Remove entry (from "Host vecformer" to next "Host" or EOF)
    awk '/^Host vecformer$/{skip=1; next} /^Host /{skip=0} !skip' "$config_file" > "$config_file.tmp"
    mv "$config_file.tmp" "$config_file"

    # Clean up trailing blank lines
    sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$config_file"

    echo "Removed 'Host vecformer' from ~/.ssh/config"


# ════════════════════════════════════════════════════════════════
# Training Infrastructure
# ════════════════════════════════════════════════════════════════

# ── Training S3 (Option 1: S3-centric, ephemeral instance) ─────

# Initialize terraform for S3-centric training
train-s3-init:
    cd infra/services/training-s3 && AWS_PROFILE=ykk terraform init

# Plan S3-centric training infrastructure
train-s3-plan:
    cd infra/services/training-s3 && AWS_PROFILE=ykk terraform plan -var-file=../../envs/{{env}}/training-s3.tfvars

# Start S3-centric training (provisions instance, starts training automatically)
train-s3-start:
    cd infra/services/training-s3 && AWS_PROFILE=ykk terraform apply -var-file=../../envs/{{env}}/training-s3.tfvars

# Show S3 training outputs
train-s3-output:
    @cd infra/services/training-s3 && AWS_PROFILE=ykk terraform output

# SSH into S3 training instance
train-s3-ssh:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-s3 && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running. Run 'just train-s3-start' first."
        exit 1
    fi
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip"

# Check S3 training status (logs)
train-s3-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-s3 && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running."
        exit 1
    fi
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip" "tail -100 /var/log/vecformer-training.log"

# Check S3 training GPU status
train-s3-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-s3 && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running."
        exit 1
    fi
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip" "nvidia-smi"

# Stop S3 training (destroys instance, results in S3)
train-s3-stop:
    cd infra/services/training-s3 && AWS_PROFILE=ykk terraform destroy -var-file=../../envs/{{env}}/training-s3.tfvars

# Download results from S3
train-s3-results run_id:
    #!/usr/bin/env bash
    set -euo pipefail
    bucket=$(cd infra/services/training-s3 && AWS_PROFILE=ykk terraform output -raw s3_bucket 2>/dev/null)
    echo "Downloading results from s3://$bucket/results/{{run_id}}/"
    mkdir -p results/{{run_id}}
    AWS_PROFILE=ykk aws s3 sync "s3://$bucket/results/{{run_id}}/" "results/{{run_id}}/"

# ── Training EBS (Option 3: Persistent EBS, auto-resume) ──────

# Initialize terraform for EBS-centric training
train-ebs-init:
    cd infra/services/training-ebs && AWS_PROFILE=ykk terraform init

# Plan EBS-centric training infrastructure
train-ebs-plan:
    cd infra/services/training-ebs && AWS_PROFILE=ykk terraform plan -var-file=../../envs/{{env}}/training-ebs.tfvars

# Start EBS-centric training (creates EBS + instance, starts training)
train-ebs-start:
    cd infra/services/training-ebs && AWS_PROFILE=ykk terraform apply -var-file=../../envs/{{env}}/training-ebs.tfvars

# Show EBS training outputs
train-ebs-output:
    @cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output

# SSH into EBS training instance
train-ebs-ssh:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running. Run 'just train-ebs-start' first."
        exit 1
    fi
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip"

# Check EBS training status (logs)
train-ebs-logs:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running."
        exit 1
    fi
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip" "tail -100 /var/log/vecformer-training.log"

# Check EBS training GPU status
train-ebs-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running."
        exit 1
    fi
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip" "nvidia-smi"

# Check training state on EBS
train-ebs-state:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running."
        exit 1
    fi
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip" "cat /data/.training-state 2>/dev/null || echo 'unknown'"

# Stop EBS training instance (keeps EBS volume for resume)
train-ebs-stop:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$(cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output -raw instance_id 2>/dev/null || echo "")
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' infra/envs/{{env}}/training-ebs.tfvars 2>/dev/null || echo 'us-east-2')
    if [[ -z "$id" || "$id" == "" ]]; then
        echo "Error: No instance found."
        exit 1
    fi
    echo "Stopping instance $id (EBS volume will persist)..."
    AWS_PROFILE=ykk aws ec2 stop-instances --instance-ids "$id" --region "$region"

# Resume EBS training (starts stopped instance)
train-ebs-resume:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$(cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output -raw instance_id 2>/dev/null || echo "")
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' infra/envs/{{env}}/training-ebs.tfvars 2>/dev/null || echo 'us-east-2')
    if [[ -z "$id" || "$id" == "" ]]; then
        echo "Error: No instance found. Run 'just train-ebs-start' first."
        exit 1
    fi
    echo "Starting instance $id (training will auto-resume)..."
    AWS_PROFILE=ykk aws ec2 start-instances --instance-ids "$id" --region "$region"

# Destroy EBS training infrastructure (DELETES EBS volume and all data!)
train-ebs-destroy:
    @echo "WARNING: This will DELETE the EBS volume and ALL training data!"
    @echo "Press Ctrl+C to cancel, or Enter to continue..."
    @read _
    cd infra/services/training-ebs && AWS_PROFILE=ykk terraform destroy -var-file=../../envs/{{env}}/training-ebs.tfvars

# Download results from EBS instance
train-ebs-results run_id:
    #!/usr/bin/env bash
    set -euo pipefail
    ip=$(cd infra/services/training-ebs && AWS_PROFILE=ykk terraform output -raw public_ip 2>/dev/null || echo "")
    if [[ -z "$ip" || "$ip" == "" ]]; then
        echo "Error: No instance running. Start the instance first."
        exit 1
    fi
    echo "Downloading results from instance..."
    mkdir -p results/{{run_id}}
    scp -o StrictHostKeyChecking=no -r "ubuntu@$ip:/data/results/{{run_id}}/*" "results/{{run_id}}/"
