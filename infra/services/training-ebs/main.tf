# infra/services/training-ebs
#
# EBS-centric training infrastructure:
# - Persistent EBS volume survives instance termination
# - Persistent spot request auto-restarts instance when capacity returns
# - Training auto-resumes from checkpoint on boot
# - SNS notifications on completion/interruption
# - Self-terminates when training completes

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 5.84.0" }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  prefix     = "${var.project}-${var.environment}"
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# ── Modules ────────────────────────────────────────────────────

module "ebs" {
  source = "../../modules/persistent-ebs"

  project           = var.project
  environment       = var.environment
  region            = local.region
  account_id        = local.account_id
  availability_zone = var.availability_zone
  volume_size_gb    = var.ebs_volume_size
}

module "sns" {
  source = "../../modules/sns-notifications"

  project            = var.project
  environment        = var.environment
  notification_email = var.notification_email
}

# ── AMI (AWS Deep Learning AMI) ────────────────────────────────

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── Security Group ─────────────────────────────────────────────

resource "aws_security_group" "training" {
  name        = "${local.prefix}-training-ebs-sg"
  description = "Security group for EBS-centric training instance"
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

  tags = { Name = "${local.prefix}-training-ebs-sg" }
}

# ── IAM Role ───────────────────────────────────────────────────

resource "aws_iam_role" "training" {
  name = "${local.prefix}-training-ebs-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ebs_attach" {
  role       = aws_iam_role.training.name
  policy_arn = module.ebs.ebs_attach_policy_arn
}

resource "aws_iam_role_policy_attachment" "sns_publish" {
  role       = aws_iam_role.training.name
  policy_arn = module.sns.sns_publish_policy_arn
}

# Allow instance to terminate itself and manage spot requests
resource "aws_iam_role_policy" "instance_management" {
  name = "instance-management"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:TerminateInstances",
          "ec2:CancelSpotInstanceRequests"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:ResourceTag/Name" = "${local.prefix}-training-ebs"
          }
        }
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training" {
  name = "${local.prefix}-training-ebs-profile"
  role = aws_iam_role.training.name
}

# ── Launch Template ────────────────────────────────────────────

resource "aws_launch_template" "training" {
  name = "${local.prefix}-training-ebs-lt"

  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.training.name
  }

  vpc_security_group_ids = [aws_security_group.training.id]

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.root_volume_size
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/scripts/user-data.sh", {
    ebs_volume_id      = module.ebs.volume_id
    sns_topic_arn      = module.sns.topic_arn
    region             = local.region
    run_id             = var.run_id
    git_repo           = var.git_repo
    git_branch         = var.git_branch
    ssh_public_key     = var.ssh_public_key
    git_user_name      = var.git_user_name
    git_user_email     = var.git_user_email
    checkpoint_epochs  = var.checkpoint_epochs
    num_epochs         = var.num_epochs
    batch_size_per_gpu = var.batch_size_per_gpu
    num_gpus           = var.num_gpus
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${local.prefix}-training-ebs"
      Project     = var.project
      Environment = var.environment
      RunID       = var.run_id
    }
  }

  tags = {
    Name = "${local.prefix}-training-ebs-lt"
  }
}

# ── Persistent Spot Instance Request ───────────────────────────

resource "aws_spot_instance_request" "training" {
  launch_template {
    id      = aws_launch_template.training.id
    version = "$Latest"
  }

  spot_price                     = var.spot_max_price
  spot_type                      = "persistent"
  instance_interruption_behavior = "stop"
  wait_for_fulfillment           = true

  subnet_id = var.subnet_id

  tags = {
    Name        = "${local.prefix}-training-ebs"
    Project     = var.project
    Environment = var.environment
    RunID       = var.run_id
  }
}

# Tag the instance
resource "aws_ec2_tag" "instance_name" {
  resource_id = aws_spot_instance_request.training.spot_instance_id
  key         = "Name"
  value       = "${local.prefix}-training-ebs"
}
