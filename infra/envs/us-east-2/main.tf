# infra/envs/us-east-2
#
# Simple GPU spot instance for SSH access (no Docker deployment).

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
  key_name   = "${var.project}-key"
  public_key = var.ssh_public_key
}

locals {
  ssh_key_name = var.ssh_public_key != "" ? aws_key_pair.this[0].key_name : ""
  subnet_ids   = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.public.ids
}

# ── Storage (IAM only — for SSM access) ────────────────────────

module "storage" {
  source = "../../modules/storage"

  project        = var.project
  environment    = "gpu"
  ecr_repo_names = []
  model_bucket   = var.model_bucket
}

# ── Service (security group) ──────────────────────────────────

module "service" {
  source = "../../services/embd"

  project        = var.project
  environment    = "gpu"
  vpc_id         = var.vpc_id
  allowed_cidrs  = var.allowed_cidrs
  ssh_public_key = var.ssh_public_key
  git_user_name  = var.git_user_name
  git_user_email = var.git_user_email
}

# ── Compute (EC2 + EIP) ───────────────────────────────────────

module "compute" {
  source = "../../modules/compute"

  project                   = var.project
  environment               = "gpu"
  instance_type             = var.instance_type
  subnet_ids                = var.subnet_ids
  security_group_id         = module.service.security_group_id
  iam_instance_profile_name = module.storage.iam_instance_profile_name
  ssh_key_name              = local.ssh_key_name
  use_spot                  = true
  spot_max_price            = var.spot_max_price
  root_volume_size          = var.root_volume_size
  user_data_base64          = module.service.user_data_base64
  extra_tags                = { service = "gpu" }
}
