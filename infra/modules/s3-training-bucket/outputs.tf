output "bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.training.id
}

output "bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.training.arn
}

output "s3_access_policy_arn" {
  description = "IAM policy ARN for S3 access"
  value       = aws_iam_policy.s3_access.arn
}
