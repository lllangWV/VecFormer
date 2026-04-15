# modules/sns-notifications
#
# SNS topic for training notifications (completion, errors, interruptions)

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 5.84.0" }
  }
}

locals {
  topic_name = "${var.project}-training-alerts"
}

# ── SNS Topic ──────────────────────────────────────────────────

resource "aws_sns_topic" "training_alerts" {
  name = local.topic_name

  tags = {
    Name        = local.topic_name
    Project     = var.project
    Environment = var.environment
  }
}

# ── Email Subscription ─────────────────────────────────────────

resource "aws_sns_topic_subscription" "email" {
  count = var.notification_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.training_alerts.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

# ── IAM Policy for Publishing ──────────────────────────────────

resource "aws_iam_policy" "sns_publish" {
  name        = "${var.project}-sns-publish"
  description = "Allow EC2 instances to publish to training alerts SNS topic"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "sns:Publish"
        Resource = aws_sns_topic.training_alerts.arn
      }
    ]
  })
}
