output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_spot_instance_request.training.spot_instance_id
}

output "spot_request_id" {
  description = "Spot instance request ID"
  value       = aws_spot_instance_request.training.id
}

output "public_ip" {
  description = "Public IP of the instance"
  value       = aws_spot_instance_request.training.public_ip
}

output "ebs_volume_id" {
  description = "Persistent EBS volume ID"
  value       = module.ebs.volume_id
}

output "sns_topic_arn" {
  description = "SNS topic for notifications"
  value       = module.sns.topic_arn
}

output "run_id" {
  description = "Training run ID"
  value       = var.run_id
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/id_ed25519 ubuntu@${aws_spot_instance_request.training.public_ip}"
}

output "results_location" {
  description = "Location of training results on EBS"
  value       = "/data/results/${var.run_id}"
}
