output "topic_arn" {
  description = "SNS topic ARN"
  value       = aws_sns_topic.training_alerts.arn
}

output "topic_name" {
  description = "SNS topic name"
  value       = aws_sns_topic.training_alerts.name
}

output "sns_publish_policy_arn" {
  description = "IAM policy ARN for SNS publishing"
  value       = aws_iam_policy.sns_publish.arn
}
