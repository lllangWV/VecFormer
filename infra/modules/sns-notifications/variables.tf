variable "project" {
  description = "Project name prefix"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "notification_email" {
  description = "Email address for notifications"
  type        = string
  default     = ""
}
