# infra/services/embd
#
# Embedding service: vLLM pooling-mode containers serving
# multimodal embedding models (Qwen3-VL) via OpenAI-compatible
# /v1/embeddings and /pooling endpoints.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 5.84.0" }
  }
}

locals {
  prefix = "${var.project}-${var.environment}-embd"

  compose_content = templatefile("${path.module}/docker-compose.yml.tftpl", {
    aws_region                  = var.aws_region
    vllm_gpu_memory_utilization = var.vllm_gpu_memory_utilization
  })

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    aws_region      = var.aws_region
    compose_content = local.compose_content
  })
}

# ── Security Group ─────────────────────────────────────────────

resource "aws_security_group" "this" {
  name        = "${local.prefix}-sg"
  description = "Security group for ${local.prefix}"
  vpc_id      = var.vpc_id

  ingress {
    description = "Qwen3-VL-Embedding-8B (vLLM pooling)"
    from_port   = 8081
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  ingress {
    description = "Qwen3-VL-Embedding-2B (vLLM pooling)"
    from_port   = 8082
    to_port     = 8082
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.prefix}-sg" }
}
