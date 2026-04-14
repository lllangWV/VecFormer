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

  # Minimal user data - just ensure SSH is ready
  user_data = <<-EOF
    #!/bin/bash
    # Minimal bootstrap for SSH-only instance
    echo "Instance ready for SSH access"
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
