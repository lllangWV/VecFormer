output "ecr_repository_urls" {
  description = "Map of name → ECR repository URL"
  value       = { for k, v in aws_ecr_repository.repos : k => v.repository_url }
}

output "ecr_registry" {
  description = "ECR registry URL (e.g. 064561338865.dkr.ecr.us-east-1.amazonaws.com). Empty when no ECR repos are created."
  value       = length(aws_ecr_repository.repos) > 0 ? split("/", values(aws_ecr_repository.repos)[0].repository_url)[0] : ""
}

output "iam_instance_profile_name" {
  description = "IAM instance profile name for EC2"
  value       = aws_iam_instance_profile.ec2.name
}
