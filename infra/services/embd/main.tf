# infra/services/embd
#
# Simple GPU compute service: EC2 spot instance with SSH access.
# No Docker deployment - just a bare instance you can SSH into.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 5.84.0" }
  }
}

locals {
  prefix = "${var.project}-${var.environment}"

  # User data that sets up the instance for development
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail

    export HOME=/home/ubuntu
    export USER=ubuntu

    # ── SSH public key ─────────────────────────────────────────────
    %{if var.ssh_public_key != ""}
    mkdir -p /home/ubuntu/.ssh
    echo "${var.ssh_public_key}" >> /home/ubuntu/.ssh/authorized_keys
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh
    chmod 700 /home/ubuntu/.ssh
    chmod 600 /home/ubuntu/.ssh/authorized_keys
    echo "SSH public key installed"
    %{endif}

    # ── Git configuration ──────────────────────────────────────────
    %{if var.git_user_name != "" && var.git_user_email != ""}
    sudo -u ubuntu git config --global user.name "${var.git_user_name}"
    sudo -u ubuntu git config --global user.email "${var.git_user_email}"
    sudo -u ubuntu git config --global init.defaultBranch main
    sudo -u ubuntu git config --global pull.rebase false
    echo "Git configured for ${var.git_user_name} <${var.git_user_email}>"
    %{endif}

    # ── System packages ────────────────────────────────────────────
    echo "Installing system packages..."
    apt-get update -qq
    apt-get install -y nvtop
    echo "System packages installed"

    # ── Pixi ───────────────────────────────────────────────────────
    echo "Installing Pixi..."
    sudo -u ubuntu bash -c 'curl -fsSL https://pixi.sh/install.sh | bash'
    echo "Pixi installed"

    # ── Claude Code CLI ────────────────────────────────────────────
    echo "Installing Claude Code CLI..."
    sudo -u ubuntu bash -c 'curl -fsSL https://claude.ai/install.sh | bash'
    echo "Claude Code CLI installed"

    # ── Done ───────────────────────────────────────────────────────
    echo "Instance setup complete"
  EOF
}

# ── Security Group ─────────────────────────────────────────────

resource "aws_security_group" "this" {
  name        = "${local.prefix}-sg"
  description = "Security group for ${local.prefix}"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.prefix}-sg" }
}
