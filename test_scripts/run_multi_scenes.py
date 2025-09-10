#!/usr/bin/env python3
"""
Multi-scene runner for DROID-SLAM
Runs dpvo_demo.py on multiple scene directories with the same parameters
"""
import os
import subprocess
import sys

# Base configuration
BASE_DIR = "/media/shuo/T7/f2nerf/s_h"
SCENES = [
    # "forest1", "forest2", "forest3",
    "garden1", "garden2", "garden3",
    "indoor", "playground",
    "university1", "university2", "university3", "university4"
]

# Fixed parameters
STRIDE = 1
SKIP = 0
BUFFER = 2048
EXPORT_COLMAP = True
CPU_CORES = "0-16"  # CPU cores to use (e.g., "0-7", "0,2,4,6", or "0-16")

def run_scene(scene_name):
    """Run DROID-SLAM on a single scene with CPU core limiting"""
    image_dir = f"{BASE_DIR}/{scene_name}/images"

    # Check if images directory exists
    if not os.path.exists(image_dir):
        print(f"Warning: {image_dir} does not exist, skipping {scene_name}")
        return False

    print(f"\n{'='*50}")
    print(f"Processing scene: {scene_name}")
    print(f"Image directory: {image_dir}")
    print(f"{'='*50}")

    # Build command with taskset to limit CPU cores
    cmd = [
        "taskset", "-c", CPU_CORES,
        "conda", "run", "-n", "droid", "python", "dpvo_demo.py",
        f"--imagedir={image_dir}",
        f"--stride={STRIDE}",
        f"--skip={SKIP}",
        f"--buffer={BUFFER}"
    ]

    if EXPORT_COLMAP:
        cmd.append("--export_colmap")

    try:
        # Run the command
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Successfully processed {scene_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {scene_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted while processing {scene_name}")
        return False

def main():
    """Main function to run all scenes"""
    print(f"DROID-SLAM Multi-Scene Runner")
    print(f"Base directory: {BASE_DIR}")
    print(f"Scenes to process: {len(SCENES)}")
    print(f"Parameters: stride={STRIDE}, skip={SKIP}, buffer={BUFFER}, export_colmap={EXPORT_COLMAP}")
    print(f"CPU cores: {CPU_CORES}")

    # Check if base directory exists
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory {BASE_DIR} does not exist!")
        sys.exit(1)

    # Process each scene
    successful = 0
    failed = 0

    for scene in SCENES:
        try:
            if run_scene(scene):
                successful += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print(f"\n⚠ Processing interrupted by user")
            break

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"✓ Successfully processed: {successful} scenes")
    print(f"✗ Failed: {failed} scenes")
    print(f"Total: {successful + failed} scenes")

if __name__ == "__main__":
    main()