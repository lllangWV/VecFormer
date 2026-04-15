#!/usr/bin/env python3
"""
Pre-cache FloorPlanCAD dataset to tensor format for faster training.

This script processes the JSON dataset once and saves as PyTorch tensors,
eliminating JSON parsing and tensor conversion overhead during training.

Usage:
    pixi run python scripts/precache_dataset.py

Expected speedup: 3-5x faster data loading
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.floorplancad.dataclass_define import SVGData, SVGDataTensor, VecData, VecDataTransformArgs
from data.floorplancad.transform_utils import to_tensor, norm_coords, to_vec_data


def process_single_file(args: tuple) -> tuple[str, bool, str]:
    """Process a single JSON file and save as tensor."""
    json_path, output_path, transform_args = args

    try:
        # Load JSON
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Convert to dataclass
        svg_data = SVGData(**json_data)

        # Convert to tensor (without augmentation - that happens at train time)
        svg_data_tensor = to_tensor(svg_data)

        # Reshape coords if needed
        if svg_data_tensor.coords.shape[-1] == 4:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 2, 2)

        # Normalize coordinates
        svg_data_tensor.coords = norm_coords(
            coords=svg_data_tensor.coords,
            bbox=svg_data_tensor.viewBox,
            min_val=transform_args['norm_range'][0],
            max_val=transform_args['norm_range'][1]
        )

        # Reshape back
        if len(svg_data_tensor.coords.shape) == 3:
            svg_data_tensor.coords = svg_data_tensor.coords.reshape(-1, 4)

        # Convert to VecData
        vec_data = to_vec_data(svg_data_tensor)

        # Save as tensor dict (more efficient than pickling dataclass)
        tensor_dict = {
            'coords': vec_data.coords,
            'feats': vec_data.feats,
            'prim_ids': vec_data.prim_ids,
            'layer_ids': vec_data.layer_ids,
            'sem_ids': vec_data.sem_ids,
            'inst_ids': vec_data.inst_ids,
            'prim_lengths': vec_data.prim_lengths,
        }

        torch.save(tensor_dict, output_path)
        return (json_path, True, "")

    except Exception as e:
        return (json_path, False, str(e))


def main():
    # Configuration
    base_dir = Path("datasets/FloorPlanCAD-sampled-as-line-jsons")
    cache_dir = Path("datasets/FloorPlanCAD-cached")
    splits = ["train", "val", "test"]

    # Default transform args (normalization only, no augmentation)
    transform_args = {
        'norm_range': [-0.5, 0.5],
    }

    print("=" * 60)
    print("FloorPlanCAD Dataset Pre-caching")
    print("=" * 60)
    print(f"Source: {base_dir}")
    print(f"Cache:  {cache_dir}")
    print()

    for split in splits:
        input_dir = base_dir / split
        output_dir = cache_dir / split

        if not input_dir.exists():
            print(f"Skipping {split} (not found)")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all JSON files
        json_files = list(input_dir.glob("*.json"))
        print(f"Processing {split}: {len(json_files)} files")

        # Prepare arguments for parallel processing
        tasks = []
        for json_path in json_files:
            output_path = output_dir / (json_path.stem + ".pt")
            tasks.append((str(json_path), str(output_path), transform_args))

        # Process in parallel
        success_count = 0
        error_count = 0

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_single_file, task): task for task in tasks}

            with tqdm(total=len(tasks), desc=f"  {split}") as pbar:
                for future in as_completed(futures):
                    path, success, error = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        if error_count <= 5:  # Only show first 5 errors
                            print(f"\n  Error in {path}: {error}")
                    pbar.update(1)

        print(f"  Done: {success_count} success, {error_count} errors")

        # Check output size
        total_size = sum(f.stat().st_size for f in output_dir.glob("*.pt"))
        print(f"  Cache size: {total_size / 1e9:.2f} GB")
        print()

    print("=" * 60)
    print("Pre-caching complete!")
    print()
    print("To use the cached dataset, modify your data config to point to:")
    print(f"  {cache_dir}")
    print()
    print("Or create a CachedFloorPlanCAD dataset class that loads .pt files")
    print("=" * 60)


if __name__ == "__main__":
    main()
