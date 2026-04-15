# modules/s3-training-bucket
#
# S3 bucket for VecFormer training artifacts:
# - Cached datasets
# - Training checkpoints
# - Final results/outputs

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 5.84.0" }
  }
}

locals {
  bucket_name = "${var.project}-training-${var.account_id}"
}

# ── S3 Bucket ──────────────────────────────────────────────────

resource "aws_s3_bucket" "training" {
  bucket = local.bucket_name

  tags = {
    Name        = local.bucket_name
    Project     = var.project
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "training" {
  bucket = aws_s3_bucket.training.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "training" {
  bucket = aws_s3_bucket.training.id

  # Keep only last 3 versions of checkpoints
  rule {
    id     = "checkpoint-cleanup"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }

    # Delete incomplete multipart uploads
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  # Archive old results after 30 days
  rule {
    id     = "results-archive"
    status = "Enabled"

    filter {
      prefix = "results/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "training" {
  bucket = aws_s3_bucket.training.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "training" {
  bucket = aws_s3_bucket.training.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── IAM Policy for Instance Access ─────────────────────────────

resource "aws_iam_policy" "s3_access" {
  name        = "${var.project}-s3-training-access"
  description = "Allow EC2 instances to access training S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.training.arn,
          "${aws_s3_bucket.training.arn}/*"
        ]
      }
    ]
  })
}
