variable "project" {
  description = "Project name prefix"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
}

variable "availability_zone" {
  description = "Availability zone for EBS volume (must match instance AZ)"
  type        = string
}

variable "volume_size_gb" {
  description = "EBS volume size in GB"
  type        = number
  default     = 200
}

variable "iops" {
  description = "Provisioned IOPS for gp3 volume"
  type        = number
  default     = 4000
}

variable "throughput_mbps" {
  description = "Throughput in MB/s for gp3 volume"
  type        = number
  default     = 250
}
