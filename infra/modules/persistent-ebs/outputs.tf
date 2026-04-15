output "volume_id" {
  description = "EBS volume ID"
  value       = aws_ebs_volume.training_data.id
}

output "volume_arn" {
  description = "EBS volume ARN"
  value       = aws_ebs_volume.training_data.arn
}

output "availability_zone" {
  description = "Availability zone of the volume"
  value       = aws_ebs_volume.training_data.availability_zone
}

output "ebs_attach_policy_arn" {
  description = "IAM policy ARN for EBS attachment"
  value       = aws_iam_policy.ebs_attach.arn
}
