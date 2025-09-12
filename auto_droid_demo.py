#!/usr/bin/env python3
"""
Automatic DROID-SLAM demo script with predefined datasets
"""

import argparse
import os
import subprocess

from data_config.dataset_config import DATASETS


def run_droid_demo(dataset_name, **kwargs):
    """Run DROID demo with specified dataset configuration"""

    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found in predefined datasets.")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False

    dataset = DATASETS[dataset_name]

    # Check if dataset is valid
    if not dataset.is_valid:
        if not dataset.exists:
            print(f"Error: Image directory not found: {dataset.imagedir}")
            print(f"Please update the path for dataset '{dataset_name}' in dataset_config.py.")
        if not dataset.calib_exists:
            print(f"Error: Calibration file not found: {dataset.calib}")
        return False

    # Build command
    cmd = [
        "conda", "run", "-n", "droid", "python", "droid_demo.py",
        f"--imagedir={dataset.imagedir}",
        f"--calib={dataset.calib}"
    ]

    # Add depth directory if specified
    if dataset.depthdir:
        cmd.append(f"--depthdir={dataset.depthdir}")

    # Add start and end indices from dataset config or kwargs
    start_idx = dataset.start_idx if dataset.start_idx is not None else kwargs.get("start_idx", 0)
    end_idx = dataset.end_idx if dataset.end_idx is not None else kwargs.get("end_idx")
    if start_idx != 0:
        cmd.append(f"--start_idx={start_idx}")
    if end_idx is not None:
        cmd.append(f"--end_idx={end_idx}")

    # Add optional arguments
    if kwargs.get("stride", 3) != 3:
        cmd.append(f"--stride={kwargs['stride']}")

    if kwargs.get("disable_vis", True):
        cmd.append("--disable_vis")

    if kwargs.get("export_colmap", False):
        cmd.append("--export_colmap")

    if kwargs.get("rerun", False):
        cmd.append("--rerun")

    if kwargs.get("t0", 0) != 0:
        cmd.append(f"--t0={kwargs['t0']}")

    if kwargs.get("reconstruction_path"):
        cmd.append(f"--reconstruction_path={kwargs['reconstruction_path']}")

    print(f"Running DROID-SLAM on dataset: {dataset_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=os.getcwd(), check=True)
        print(f"Successfully completed processing dataset: {dataset_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running DROID-SLAM: {e}")
        return False

def run_all_datasets(**kwargs):
    """Run DROID demo on all available datasets"""
    success_count = 0
    total_count = 0

    for dataset_name in DATASETS.keys():
        print(f"\n{'='*60}")
        print(f"Processing dataset {total_count + 1}/{len(DATASETS)}: {dataset_name}")
        print(f"{'='*60}")

        if run_droid_demo(dataset_name, **kwargs):
            success_count += 1
        total_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{total_count} datasets processed successfully")
    print(f"{'='*60}")

def list_datasets():
    """List all available predefined datasets"""
    print("Available predefined datasets:")
    print("-" * 80)
    for name, dataset in DATASETS.items():
        img_status = "✓" if dataset.exists else "✗"
        calib_status = "✓" if dataset.calib_exists else "✗"
        print(f"{img_status}{calib_status} {name:<15}")
        print(f"    Image dir: {dataset.imagedir}")
        print(f"    Calib: {dataset.calib}")
        if dataset.depthdir:
            print(f"    Depth dir: {dataset.depthdir}")
        print()
    print("✓ = Path exists, ✗ = Path not found (first symbol: images, second: calib)")

def main():
    parser = argparse.ArgumentParser(description="Automatic DROID-SLAM demo with predefined datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name to process (use --list to see available)")
    parser.add_argument("--all", action="store_true", help="Process all available datasets")
    parser.add_argument("--list", action="store_true", help="List all available datasets")

    # Optional DROID parameters
    parser.add_argument("--stride", type=int, default=3, help="Frame stride (default: 3)")
    parser.add_argument("--start_idx", type=int, help="Starting image index (overrides dataset config)")
    parser.add_argument("--end_idx", type=int, help="Ending image index (overrides dataset config)")
    parser.add_argument("--disable_vis", action="store_true", help="Disable visualization")
    parser.add_argument("--export_colmap", action="store_true", help="Export to COLMAP format")
    parser.add_argument("--rerun", action="store_true", help="Rerun processing")
    parser.add_argument("--t0", type=int, default=0, help="Starting frame")
    parser.add_argument("--reconstruction_path", type=str, help="Path to save reconstruction")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # Extract DROID parameters
    droid_params = {
        "stride": args.stride,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "disable_vis": args.disable_vis,
        "export_colmap": args.export_colmap,
        "rerun": args.rerun,
        "t0": args.t0,
        "reconstruction_path": args.reconstruction_path
    }

    if args.all:
        run_all_datasets(**droid_params)
    elif args.dataset:
        run_droid_demo(args.dataset, **droid_params)
    else:
        print("Please specify either --dataset <name>, --all, or --list")
        print("Use --help for more information")

if __name__ == "__main__":
    main()