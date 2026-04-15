variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "project" {
  description = "Project name prefix"
  type        = string
  default     = "vecformer"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for instance placement"
  type        = string
}

variable "availability_zone" {
  description = "Availability zone (must match subnet)"
  type        = string
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = []
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "p5en.48xlarge"
}

variable "spot_max_price" {
  description = "Maximum spot price (empty = on-demand price cap)"
  type        = string
  default     = "20.00"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 100
}

variable "ebs_volume_size" {
  description = "Persistent EBS data volume size in GB"
  type        = number
  default     = 200
}

variable "notification_email" {
  description = "Email for training notifications"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key for instance access"
  type        = string
  default     = ""
}

variable "git_user_name" {
  description = "Git user.name"
  type        = string
  default     = ""
}

variable "git_user_email" {
  description = "Git user.email"
  type        = string
  default     = ""
}

# ── Training Configuration ─────────────────────────────────────

variable "run_id" {
  description = "Unique identifier for this training run"
  type        = string
}

variable "git_repo" {
  description = "Git repository URL"
  type        = string
  default     = "https://github.com/lllangWV/VecFormer.git"
}

variable "git_branch" {
  description = "Git branch to checkout"
  type        = string
  default     = "main"
}

variable "checkpoint_epochs" {
  description = "Save checkpoint every N epochs"
  type        = number
  default     = 10
}

variable "num_epochs" {
  description = "Total number of training epochs"
  type        = number
  default     = 500
}

variable "batch_size_per_gpu" {
  description = "Batch size per GPU"
  type        = number
  default     = 10
}

variable "num_gpus" {
  description = "Number of GPUs to use"
  type        = number
  default     = 8
}
