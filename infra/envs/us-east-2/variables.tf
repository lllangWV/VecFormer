variable "project" {
  description = "Project name prefix"
  type        = string
  default     = "ec2-serve"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "vpc_id" {
  description = "VPC ID for EC2 instances"
  type        = string
}

variable "subnet_ids" {
  description = "Explicit subnet IDs for EC2 placement (overrides auto-discovery)"
  type        = list(string)
  default     = []
}

variable "ssh_public_key" {
  description = "SSH public key material for EC2 access"
  type        = string
  default     = ""
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to access services and SSH"
  type        = list(string)
  default     = ["167.77.192.18/32", "71.182.199.107/32"]
}

variable "model_bucket" {
  description = "S3 bucket name for model artifacts (used by IAM policy)"
  type        = string
  default     = "ykk-serve-models-064561338865"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g7e.2xlarge"
}

variable "spot_max_price" {
  description = "Maximum hourly price for spot instance (empty = on-demand price cap)"
  type        = string
  default     = ""
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 300
}

variable "git_user_name" {
  description = "Git user.name to configure on the instance"
  type        = string
  default     = ""
}

variable "git_user_email" {
  description = "Git user.email to configure on the instance"
  type        = string
  default     = ""
}
