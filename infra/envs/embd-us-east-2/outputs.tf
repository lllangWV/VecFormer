output "instance_id" {
  description = "EC2 instance ID"
  value       = module.compute.instance_id
}

output "public_ip" {
  description = "Elastic IP address"
  value       = module.compute.public_ip
}

output "embd_8b_url" {
  description = "Qwen3-VL-Embedding-8B endpoint (OpenAI-compatible /v1/embeddings)"
  value       = "http://${module.compute.public_ip}:8081"
}

output "embd_2b_url" {
  description = "Qwen3-VL-Embedding-2B endpoint (OpenAI-compatible /v1/embeddings)"
  value       = "http://${module.compute.public_ip}:8082"
}

output "compose_content" {
  description = "Rendered docker-compose.yml for the embedding service"
  value       = module.embd.compose_content
  sensitive   = false
}
