# Docker management (YKK-Vision)

set dotenv-load := false

image := "ykkvision"

# Registry
aws_account := "064561338865"
aws_region := "us-east-1"
ecr := aws_account + ".dkr.ecr." + aws_region + ".amazonaws.com"
ghcr_cache := "ghcr.io/ykk-xtechlab-engineering/ykkvision"

# List available recipes
default:
    @just --list

# Build the ykkvision image
build:
    docker buildx build \
        --cache-from=type=registry,ref={{ghcr_cache}}/cache:buildcache \
        --cache-to=type=registry,ref={{ghcr_cache}}/cache:buildcache,mode=max \
        --load \
        -t {{image}} .

# Build without registry cache (local layer cache only)
build-local:
    DOCKER_BUILDKIT=1 docker build -t {{image}} .

# Run a single service: just run sam3
run service *args:
    docker run --rm --gpus all \
        -p 8000:8000 \
        -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        {{args}} \
        {{image}} \
        bash -c "source /shell-hook.sh && exec uvicorn ykkvision.{{service}}.server:app --host 0.0.0.0 --port 8000"

# Run combined server (all services)
run-all *args:
    docker run --rm --gpus all \
        -p 8000:8000 \
        -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        {{args}} \
        {{image}}

# Run all services via docker compose (combined mode)
up *args:
    docker compose up {{args}}

# Run individual services via docker compose (split mode)
up-split *args:
    docker compose --profile split up {{args}}

# Stop all services
down *args:
    docker compose down {{args}}

# Show image size
images:
    @docker images --format "table {{{{.Repository}}}}\t{{{{.Tag}}}}\t{{{{.Size}}}}" | grep {{image}}

# Login to ECR
ecr-login:
    AWS_PROFILE=ykk aws ecr get-login-password --region {{aws_region}} | \
        docker login --username AWS --password-stdin {{ecr}}

ecr_repo := "ec2-serve/ykkvision"

# Environment: "embd-us-east-2" (embedding service)
# Override with: just --set env <name> <command>
env := "embd-us-east-2"
tf_dir := "infra/envs/" + env
tf_vars := "-var-file=embd.tfvars"

# Push to ECR
push: build
    docker tag {{image}} {{ecr}}/{{ecr_repo}}:latest
    docker push {{ecr}}/{{ecr_repo}}:latest

# ── Terraform ──────────────────────────────────────────────────

# Show terraform plan for dev environment
tf-plan:
    cd {{tf_dir}} && AWS_PROFILE=ykk terraform plan {{tf_vars}}

# Apply terraform for dev environment
tf-apply:
    cd {{tf_dir}} && AWS_PROFILE=ykk terraform apply {{tf_vars}}

# Destroy terraform dev environment
tf-destroy:
    cd {{tf_dir}} && AWS_PROFILE=ykk terraform destroy {{tf_vars}}

# Show terraform outputs
tf-output:
    @cd {{tf_dir}} && AWS_PROFILE=ykk terraform output

# ── EC2 Instance Management ───────────────────────────────────
# All commands read instance ID and region from terraform output.
# Switch env with: just --set env dev-us-east-2 <command>

# Helper: get instance ID and region from terraform
_tf_id := "cd " + tf_dir + " && AWS_PROFILE=ykk terraform output -raw instance_id"
_tf_region := "cd " + tf_dir + " && AWS_PROFILE=ykk terraform output -raw ecr_registry | grep -oP 'ecr\\.\\K[^.]+'"

# Start the ykkvision EC2 instance
start-instance:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    AWS_PROFILE=ykk aws ec2 start-instances --instance-ids "$id" --region "$region"

# Stop the ykkvision EC2 instance
stop-instance:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    AWS_PROFILE=ykk aws ec2 stop-instances --instance-ids "$id" --region "$region"

# Check the ykkvision EC2 instance status
instance-status:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    AWS_PROFILE=ykk aws ec2 describe-instance-status \
        --instance-ids "$id" \
        --region "$region" \
        --include-all-instances \
        --query 'InstanceStatuses[0].InstanceState.Name' \
        --output text

# SSH into the instance via EC2 Instance Connect
ssh-instance:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    ip=$(AWS_PROFILE=ykk aws ec2 describe-instances --instance-ids "$id" --region "$region" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    AWS_PROFILE=ykk aws ec2-instance-connect send-ssh-public-key \
        --instance-id "$id" --instance-os-user ubuntu \
        --ssh-public-key file://~/.ssh/id_ed25519.pub --region "$region" > /dev/null
    ssh -o StrictHostKeyChecking=no "ubuntu@$ip"

# Show docker compose service status on the instance
instance-services:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    cmd_id=$(AWS_PROFILE=ykk aws ssm send-command \
        --instance-ids "$id" \
        --region "$region" \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=["cd /opt/ec2-serve && docker compose ps"]' \
        --query 'Command.CommandId' --output text)
    sleep 5
    AWS_PROFILE=ykk aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$id" \
        --region "$region" \
        --query 'StandardOutputContent' --output text

# Tail logs from all services on the instance
instance-logs *args:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    cmd_id=$(AWS_PROFILE=ykk aws ssm send-command \
        --instance-ids "$id" \
        --region "$region" \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=["cd /opt/ec2-serve && docker compose logs --tail 50 {{args}}"]' \
        --query 'Command.CommandId' --output text)
    sleep 5
    AWS_PROFILE=ykk aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$id" \
        --region "$region" \
        --query 'StandardOutputContent' --output text

# Restart services on the instance
instance-restart:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    cmd_id=$(AWS_PROFILE=ykk aws ssm send-command \
        --instance-ids "$id" \
        --region "$region" \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=["cd /opt/ec2-serve && docker compose restart"]' \
        --query 'Command.CommandId' --output text)
    sleep 10
    AWS_PROFILE=ykk aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$id" \
        --region "$region" \
        --query 'StandardOutputContent' --output text


# instance-nvidia-smi:


# ── EC2 Service Redeployment ──────────────────────────────────
# Redeploy compose config to the instance and recreate services.
# This renders the terraform template, pushes it via SSM, and restarts.
# Use after editing docker-compose.yml.tftpl without needing full tf-apply.

# Redeploy all services on the EC2 instance (push updated compose + restart)
instance-redeploy:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    # Render the compose template using current terraform outputs
    cd {{tf_dir}}
    compose=$(AWS_PROFILE=ykk terraform output -raw compose_content 2>/dev/null || echo "")
    if [ -z "$compose" ]; then
        echo "Error: compose_content not in terraform output. Run 'just tf-apply' first."
        exit 1
    fi
    cd - > /dev/null
    # Escape for SSM command (write via heredoc)
    encoded=$(echo "$compose" | base64 -w 0)
    echo "Pushing updated docker-compose.yml to instance..."
    cmd_id=$(AWS_PROFILE=ykk aws ssm send-command \
        --instance-ids "$id" \
        --region "$region" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"echo '$encoded' | base64 -d > /opt/ec2-serve/docker-compose.yml && cd /opt/ec2-serve && docker compose up -d --force-recreate --remove-orphans\"]" \
        --query 'Command.CommandId' --output text)
    echo "Waiting for redeploy..."
    sleep 15
    AWS_PROFILE=ykk aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$id" \
        --region "$region" \
        --query '[StandardOutputContent, StandardErrorContent]' --output text

# Redeploy a single service (recreate without touching other services)
# Usage: just instance-redeploy-service qwen35-9b
instance-redeploy-service service:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    cd {{tf_dir}}
    compose=$(AWS_PROFILE=ykk terraform output -raw compose_content 2>/dev/null || echo "")
    if [ -z "$compose" ]; then
        echo "Error: compose_content not in terraform output. Run 'just tf-apply' first."
        exit 1
    fi
    cd - > /dev/null
    encoded=$(echo "$compose" | base64 -w 0)
    echo "Pushing updated compose and recreating {{service}}..."
    cmd_id=$(AWS_PROFILE=ykk aws ssm send-command \
        --instance-ids "$id" \
        --region "$region" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"echo '$encoded' | base64 -d > /opt/ec2-serve/docker-compose.yml && cd /opt/ec2-serve && docker compose up -d --force-recreate --no-deps {{service}}\"]" \
        --query 'Command.CommandId' --output text)
    echo "Waiting for {{service}} to restart..."
    sleep 15
    AWS_PROFILE=ykk aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$id" \
        --region "$region" \
        --query '[StandardOutputContent, StandardErrorContent]' --output text

# View logs for a service on the instance (via SSH + EC2 Instance Connect)
# Usage: just instance-logs-service qwen35-9b
instance-logs-service service:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    ip=$(AWS_PROFILE=ykk aws ec2 describe-instances --instance-ids "$id" --region "$region" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    AWS_PROFILE=ykk aws ec2-instance-connect send-ssh-public-key \
        --instance-id "$id" --instance-os-user ubuntu \
        --ssh-public-key file://~/.ssh/id_ed25519.pub --region "$region" > /dev/null
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "ubuntu@$ip" \
        "cd /opt/ec2-serve && docker compose logs --tail 80 {{service}} 2>&1"

# Stream live logs for a service on the instance (via SSH + EC2 Instance Connect)
# Usage: just instance-logs-follow qwen35-9b
instance-logs-follow service:
    #!/usr/bin/env bash
    set -euo pipefail
    id=$({{_tf_id}})
    region=$(grep -oP 'aws_region\s*=\s*"\K[^"]+' {{tf_dir}}/embd.tfvars 2>/dev/null || echo "us-east-1")
    ip=$(AWS_PROFILE=ykk aws ec2 describe-instances --instance-ids "$id" --region "$region" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    AWS_PROFILE=ykk aws ec2-instance-connect send-ssh-public-key \
        --instance-id "$id" --instance-os-user ubuntu \
        --ssh-public-key file://~/.ssh/id_ed25519.pub --region "$region" > /dev/null
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -t "ubuntu@$ip" \
        "cd /opt/ec2-serve && docker compose logs --tail 80 -f {{service}} 2>&1"

# Desktop deployment
desktop := "desktop"

# ── Layout service ────────────────────────────────────────────

layout_image := "ykkvision-layout"
layout_desktop_dir := "~/layout"

# Build the layout service image
build-layout:
    DOCKER_BUILDKIT=1 docker build -t {{layout_image}} .

# Copy layout image to desktop via ssh
copy-layout: build-layout
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Saving image..."
    docker save {{layout_image}} | zstd -T0 > /tmp/{{layout_image}}.tar.zst
    echo "Copying to {{desktop}}..."
    scp /tmp/{{layout_image}}.tar.zst {{desktop}}:/tmp/
    echo "Loading image on {{desktop}}..."
    ssh {{desktop}} "zstd -d /tmp/{{layout_image}}.tar.zst --stdout | docker load"
    ssh {{desktop}} "rm /tmp/{{layout_image}}.tar.zst"
    rm /tmp/{{layout_image}}.tar.zst

# Deploy layout service to desktop (full: build + copy image + deploy)
deploy-layout: copy-layout
    ssh {{desktop}} "mkdir -p {{layout_desktop_dir}}"
    scp infra/services/layout/docker-compose.yml {{desktop}}:{{layout_desktop_dir}}/
    ssh {{desktop}} "cd {{layout_desktop_dir}} && docker compose up -d"

# Redeploy layout config to desktop (no image rebuild)
redeploy-layout:
    ssh {{desktop}} "mkdir -p {{layout_desktop_dir}}"
    scp infra/services/layout/docker-compose.yml {{desktop}}:{{layout_desktop_dir}}/
    ssh {{desktop}} "cd {{layout_desktop_dir}} && docker compose up -d --force-recreate"

# Print the layout service URL
layout-url:
    @echo "http://$(ssh {{desktop}} "ip -4 route get 1 | awk '{print \$7; exit}'"):8001"

# Stop layout on desktop
stop-layout:
    ssh {{desktop}} "cd {{layout_desktop_dir}} && docker compose down"

# View layout logs on desktop
logs-layout:
    ssh {{desktop}} "cd {{layout_desktop_dir}} && docker compose logs -f"

# ── Qwen3 embedding service ──────────────────────────────────

desktop_dir := "~/qwen3-embedding"

# Deploy qwen3-embedding service to desktop
deploy-embedding:
    ssh {{desktop}} "mkdir -p {{desktop_dir}}"
    scp infra/services/qwen3-embedding/docker-compose.yml {{desktop}}:{{desktop_dir}}/
    ssh {{desktop}} "cd {{desktop_dir}} && docker compose up -d"

# Stop qwen3-embedding on desktop
stop-embedding:
    ssh {{desktop}} "cd {{desktop_dir}} && docker compose down"

# View qwen3-embedding logs on desktop
logs-embedding:
    ssh {{desktop}} "cd {{desktop_dir}} && docker compose logs -f"

# Benchmark qwen3-embedding on desktop
bench-embedding:
    ssh {{desktop}} "docker exec qwen3-embedding-qwen3-embedding-1 \
        vllm bench serve \
        --model Qwen/Qwen3-Embedding-8B \
        --backend openai-embeddings \
        --dataset-name random \
        --host 127.0.0.1 \
        --port 8888 \
        --endpoint /v1/embeddings \
        --tokenizer Qwen/Qwen3-Embedding-8B \
        --random-input 200 \
        --save-result \
        --result-dir /tmp/bench"
    ssh {{desktop}} "docker cp qwen3-embedding-qwen3-embedding-1:/tmp/bench/. {{desktop_dir}}/bench/"


