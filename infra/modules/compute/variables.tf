variable "project" {
  description = "Project name prefix"
  type        = string
  default     = "ec2-serve"
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g7e.2xlarge"
}

variable "ami_name_filter" {
  description = "AMI name filter pattern"
  type        = string
  default     = "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"
}

variable "subnet_ids" {
  description = "Subnet IDs to place the instance in (empty = AWS chooses AZ)"
  type        = list(string)
  default     = []
}

variable "security_group_id" {
  description = "Security group ID for the instance"
  type        = string
}

variable "iam_instance_profile_name" {
  description = "IAM instance profile name"
  type        = string
}

variable "ssh_key_name" {
  description = "SSH key pair name (empty to disable SSH key)"
  type        = string
  default     = ""
}

variable "use_spot" {
  description = "Use spot instances"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum hourly price for spot instance (empty = on-demand price cap)"
  type        = string
  default     = ""
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 150
}

variable "user_data_base64" {
  description = "Base64-encoded user data script"
  type        = string
}

variable "extra_tags" {
  description = "Additional tags to apply to the instance"
  type        = map(string)
  default     = {}
}
