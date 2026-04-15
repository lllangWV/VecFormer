# modules/persistent-ebs
#
# Persistent EBS volume for training data that survives instance termination.
# Contains: cached dataset, checkpoints, outputs, repo clone

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 5.84.0" }
  }
}

locals {
  volume_name = "${var.project}-training-data"
}

# ── EBS Volume ─────────────────────────────────────────────────

resource "aws_ebs_volume" "training_data" {
  availability_zone = var.availability_zone
  size              = var.volume_size_gb
  type              = "gp3"
  iops              = var.iops
  throughput        = var.throughput_mbps

  tags = {
    Name        = local.volume_name
    Project     = var.project
    Environment = var.environment
    Purpose     = "training-data"
  }
}

# ── IAM Policy for Volume Attachment ───────────────────────────

resource "aws_iam_policy" "ebs_attach" {
  name        = "${var.project}-ebs-attach"
  description = "Allow EC2 instances to attach/detach training EBS volume"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:AttachVolume",
          "ec2:DetachVolume",
          "ec2:DescribeVolumes",
          "ec2:DescribeVolumeStatus"
        ]
        Resource = [
          "arn:aws:ec2:${var.region}:${var.account_id}:volume/${aws_ebs_volume.training_data.id}",
          "arn:aws:ec2:${var.region}:${var.account_id}:instance/*"
        ]
      }
    ]
  })
}
