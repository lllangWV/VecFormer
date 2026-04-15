# infra/services/training-s3
#
# S3-centric training infrastructure:
# - Ephemeral spot instance
# - S3 bucket for datasets, checkpoints, results
# - Automatic checkpoint sync every 10 epochs
# - SNS notifications on completion
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

module "s3_bucket" {
  source = "../../modules/s3-training-bucket"

  project     = var.project
  environment = var.environment
  account_id  = local.account_id
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
  name        = "${local.prefix}-training-s3-sg"
  description = "Security group for S3-centric training instance"
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

  tags = { Name = "${local.prefix}-training-s3-sg" }
}

# ── IAM Role ───────────────────────────────────────────────────

resource "aws_iam_role" "training" {
  name = "${local.prefix}-training-s3-role"

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

resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.training.name
  policy_arn = module.s3_bucket.s3_access_policy_arn
}

resource "aws_iam_role_policy_attachment" "sns_publish" {
  role       = aws_iam_role.training.name
  policy_arn = module.sns.sns_publish_policy_arn
}

# Allow instance to terminate itself
resource "aws_iam_role_policy" "self_terminate" {
  name = "self-terminate"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ec2:TerminateInstances"
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:ResourceTag/Name" = "${local.prefix}-training-s3"
          }
        }
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training" {
  name = "${local.prefix}-training-s3-profile"
  role = aws_iam_role.training.name
}

# ── User Data Script ───────────────────────────────────────────

locals {
  user_data = templatefile("${path.module}/scripts/user-data.sh", {
    s3_bucket          = module.s3_bucket.bucket_name
    sns_topic_arn      = module.sns.topic_arn
    region             = local.region
    run_id             = var.run_id
    git_repo           = var.git_repo
    git_branch         = var.git_branch
    ssh_public_key     = var.ssh_public_key
    git_user_name      = var.git_user_name
    git_user_email     = var.git_user_email
    training_config    = var.training_config
    checkpoint_epochs  = var.checkpoint_epochs
    num_epochs         = var.num_epochs
    batch_size_per_gpu = var.batch_size_per_gpu
    num_gpus           = var.num_gpus
  })
}

# ── Spot Instance Request ──────────────────────────────────────

resource "aws_spot_instance_request" "training" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  spot_price             = var.spot_max_price
  spot_type              = "one-time"
  wait_for_fulfillment   = true

  vpc_security_group_ids = [aws_security_group.training.id]
  subnet_id              = var.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.training.name

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = base64encode(local.user_data)

  tags = {
    Name        = "${local.prefix}-training-s3"
    Project     = var.project
    Environment = var.environment
    RunID       = var.run_id
  }
}

# Tag the instance (spot request tags don't propagate automatically)
resource "aws_ec2_tag" "instance_name" {
  resource_id = aws_spot_instance_request.training.spot_instance_id
  key         = "Name"
  value       = "${local.prefix}-training-s3"
}
