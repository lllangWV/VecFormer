variable "project" {
  description = "Project name prefix (used as ECR namespace)"
  type        = string
  default     = "ec2-serve"
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
}

variable "ecr_repo_names" {
  description = "List of names to create ECR repositories for"
  type        = list(string)
  default     = ["sam3", "glm-ocr", "pp-doclayout", "combined", "ykkvision", "glm-ocr-vllm"]
}

variable "model_bucket" {
  description = "S3 bucket name for model artifacts"
  type        = string
  default     = "ykk-serve-models-064561338865"
}
