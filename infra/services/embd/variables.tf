variable "project" {
  description = "Project name prefix"
  type        = string
  default     = "ec2-serve"
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
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

variable "ssh_public_key" {
  description = "SSH public key to install on the instance for persistent access"
  type        = string
  default     = ""
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
