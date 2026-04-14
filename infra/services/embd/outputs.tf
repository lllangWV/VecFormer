output "user_data_base64" {
  description = "Base64-encoded user data script for EC2 bootstrap"
  value       = base64encode(local.user_data)
}

output "security_group_id" {
  description = "Security group ID for the embedding service"
  value       = aws_security_group.this.id
}

output "compose_content" {
  description = "Rendered docker-compose.yml content for the embedding service"
  value       = local.compose_content
}
