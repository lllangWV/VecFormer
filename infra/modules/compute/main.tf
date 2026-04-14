# infra/modules/compute
#
# Generic GPU compute: Deep Learning AMI, EC2 instance, Elastic IP.
# Knows nothing about what runs on the instance — accepts user_data_base64.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 5.84.0" }
  }
}

locals {
  prefix = "${var.project}-${var.environment}"
}

# ── AMI: Deep Learning AMI ──────────────────────────────────────

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# ── EC2 Instance ────────────────────────────────────────────────

resource "aws_instance" "this" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  subnet_id              = length(var.subnet_ids) > 0 ? var.subnet_ids[0] : null
  vpc_security_group_ids = [var.security_group_id]
  iam_instance_profile   = var.iam_instance_profile_name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
      }
    }
  }

  root_block_device {
    volume_size = var.root_volume_size
    volume_type = "gp3"
  }

  user_data_base64 = var.user_data_base64

  tags = merge(
    {
      Name        = "${local.prefix}-instance"
      project     = var.project
      environment = var.environment
    },
    var.extra_tags,
  )

  lifecycle {
    ignore_changes = [ami, user_data, user_data_base64]
  }
}

# ── Elastic IP ─────────────────────────────────────────────────

resource "aws_eip" "this" {
  domain = "vpc"
  tags   = { Name = "${local.prefix}-eip" }
}

resource "aws_eip_association" "this" {
  instance_id   = aws_instance.this.id
  allocation_id = aws_eip.this.id
}
