# infra/envs/embd-us-east-2
#
# Embedding service environment (us-east-2): vLLM pooling-mode
# containers for multimodal embedding models on spot instances.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.84.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Subnet discovery + SSH key ─────────────────────────────────

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }
  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

resource "aws_key_pair" "this" {
  count      = var.ssh_public_key != "" ? 1 : 0
  key_name   = "${var.project}-embd-key"
  public_key = var.ssh_public_key
}

locals {
  ssh_key_name = var.ssh_public_key != "" ? aws_key_pair.this[0].key_name : ""
  subnet_ids   = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.public.ids
}

# ── Storage (IAM only — no ECR repos needed for public images) ─

module "storage" {
  source = "../../modules/storage"

  project        = var.project
  environment    = "embd"
  ecr_repo_names = []
  model_bucket   = var.model_bucket
}

# ── Service (embd: security group + user data) ────────────────

module "embd" {
  source = "../../services/embd"

  project                     = var.project
  environment                 = "embd"
  aws_region                  = var.aws_region
  vpc_id                      = var.vpc_id
  allowed_cidrs               = var.allowed_cidrs
  vllm_gpu_memory_utilization = var.vllm_gpu_memory_utilization
}

# ── Compute (EC2 + EIP) ───────────────────────────────────────

module "compute" {
  source = "../../modules/compute"

  project                   = var.project
  environment               = "embd"
  instance_type             = var.instance_type
  subnet_ids                = var.subnet_ids
  security_group_id         = module.embd.security_group_id
  iam_instance_profile_name = module.storage.iam_instance_profile_name
  ssh_key_name              = local.ssh_key_name
  use_spot                  = true
  spot_max_price            = var.spot_max_price
  root_volume_size          = var.root_volume_size
  user_data_base64          = module.embd.user_data_base64
  extra_tags                = { service = "embd" }
}
