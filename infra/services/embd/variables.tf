variable "project" {
  description = "Project name prefix"
  type        = string
  default     = "ec2-serve"
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region (used for CloudWatch log groups)"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID for the security group"
  type        = string
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to access services and SSH"
  type        = list(string)
  default     = ["167.77.192.18/32", "71.182.199.107/32"]
}

variable "vllm_gpu_memory_utilization" {
  description = "Fraction of GPU memory for the 8B embedding model (e.g. 0.25 = 24GB on 96GB)"
  type        = string
  default     = "0.25"
}
