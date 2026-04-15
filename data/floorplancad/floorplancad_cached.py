"""
Cached FloorPlanCAD Dataset - Loads pre-processed tensor files for faster training.

Usage:
    1. Run scripts/precache_dataset.py to generate cached tensors
    2. Use this dataset class instead of FloorPlanCAD

Expected speedup: 3-5x faster data loading compared to JSON-based loading.
"""

import os
from typing import Dict, Any

import torch
from torch.utils.data import Dataset

from utils.svg_util import scan_dir
from .dataclass_define import VecData, VecDataTransformArgs
from .augment_utils import random_flip, random_rotate, random_scale, random_translation
from .floorplancad import FloorPlanCAD  # For collate_fn


class FloorPlanCADCached(Dataset):
    """
    Cached version of FloorPlanCAD that loads pre-processed .pt files.

    Pre-processing (normalization) is done once during caching.
    Augmentation (flip, rotate, scale, translate) is still done on-the-fly during training.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        train_transform_args: Dict[str, Any],
        eval_transform_args: Dict[str, Any],
    ):
        self.root_dir = root_dir
        self.split = split
        self.train_transform_args = train_transform_args
        self.eval_transform_args = eval_transform_args
        self.data_dir = os.path.join(root_dir, split)
        self.data_paths = scan_dir(self.data_dir, suffix=".pt")

        # Optional: preload all data to RAM for maximum speed
        self._cache = None

    def preload_to_ram(self):
        """Load entire dataset to RAM. Call once before training for max speed."""
        print(f"Preloading {len(self)} samples to RAM...")
        self._cache = []
        for i in range(len(self)):
            data_path = os.path.join(self.data_dir, self.data_paths[i])
            self._cache.append(torch.load(data_path, weights_only=True))
        print(f"Preloaded {len(self._cache)} samples")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load from RAM cache or disk
        if self._cache is not None:
            tensor_dict = self._cache[idx]
        else:
            data_path = os.path.join(self.data_dir, self.data_paths[idx])
            tensor_dict = torch.load(data_path, weights_only=True)

        # Apply augmentation if training
        coords = tensor_dict['coords'].clone()  # Clone to avoid modifying cache
        transform_args = self._get_transform_args()

        if self.split == "train":
            coords = self._augment(coords, transform_args)

        # Build VecData
        vec_data = VecData(
            data_path=self.data_paths[idx],
            coords=coords,
            feats=tensor_dict['feats'],
            prim_ids=tensor_dict['prim_ids'],
            layer_ids=tensor_dict['layer_ids'],
            sem_ids=tensor_dict['sem_ids'],
            inst_ids=tensor_dict['inst_ids'],
            prim_lengths=tensor_dict['prim_lengths'],
        )

        return vec_data

    def _get_transform_args(self) -> VecDataTransformArgs:
        if self.split == "train":
            return VecDataTransformArgs(**self.train_transform_args)
        else:
            return VecDataTransformArgs(**self.eval_transform_args)

    def _augment(self, coords: torch.Tensor, args: VecDataTransformArgs) -> torch.Tensor:
        """Apply data augmentation (same as original, but coords already normalized)."""
        min_val, max_val = args.norm_range

        # Reshape for augmentation if needed
        if coords.shape[-1] == 4:
            coords = coords.reshape(-1, 2, 2)

        # Apply augmentations
        coords = random_flip(coords, min_val, max_val, "vertical", args.random_vertical_flip)
        coords = random_flip(coords, min_val, max_val, "horizontal", args.random_horizontal_flip)
        coords = random_rotate(coords, min_val, max_val, args.random_rotate)
        coords = random_scale(coords, min_val, max_val, args.random_scale[0], args.random_scale[1])
        coords = random_translation(coords, args.random_translation[0], args.random_translation[1])

        # Reshape back
        if len(coords.shape) == 3:
            coords = coords.reshape(-1, 4)

        return coords

    # Use the same collate_fn as the original dataset
    collate_fn = FloorPlanCAD.collate_fn


class FloorPlanCADInMemory(Dataset):
    """
    Fully in-memory version that loads everything at initialization.

    Best for: Maximum training speed when dataset fits in RAM.
    Trade-off: Longer initialization time, higher memory usage.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        train_transform_args: Dict[str, Any],
        eval_transform_args: Dict[str, Any],
    ):
        self.split = split
        self.train_transform_args = train_transform_args
        self.eval_transform_args = eval_transform_args

        # Load all data at init
        data_dir = os.path.join(root_dir, split)
        data_paths = scan_dir(data_dir, suffix=".pt")

        print(f"Loading {split} dataset to memory ({len(data_paths)} samples)...")
        self.data = []
        self.paths = []

        for path in data_paths:
            full_path = os.path.join(data_dir, path)
            self.data.append(torch.load(full_path, weights_only=True))
            self.paths.append(path)

        print(f"Loaded {len(self.data)} samples to memory")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_dict = self.data[idx]
        coords = tensor_dict['coords'].clone()

        # Apply augmentation if training
        if self.split == "train":
            transform_args = VecDataTransformArgs(**self.train_transform_args)
            coords = self._augment(coords, transform_args)

        return VecData(
            data_path=self.paths[idx],
            coords=coords,
            feats=tensor_dict['feats'],
            prim_ids=tensor_dict['prim_ids'],
            layer_ids=tensor_dict['layer_ids'],
            sem_ids=tensor_dict['sem_ids'],
            inst_ids=tensor_dict['inst_ids'],
            prim_lengths=tensor_dict['prim_lengths'],
        )

    def _augment(self, coords: torch.Tensor, args: VecDataTransformArgs) -> torch.Tensor:
        min_val, max_val = args.norm_range
        if coords.shape[-1] == 4:
            coords = coords.reshape(-1, 2, 2)

        coords = random_flip(coords, min_val, max_val, "vertical", args.random_vertical_flip)
        coords = random_flip(coords, min_val, max_val, "horizontal", args.random_horizontal_flip)
        coords = random_rotate(coords, min_val, max_val, args.random_rotate)
        coords = random_scale(coords, min_val, max_val, args.random_scale[0], args.random_scale[1])
        coords = random_translation(coords, args.random_translation[0], args.random_translation[1])

        if len(coords.shape) == 3:
            coords = coords.reshape(-1, 4)
        return coords

    collate_fn = FloorPlanCAD.collate_fn
