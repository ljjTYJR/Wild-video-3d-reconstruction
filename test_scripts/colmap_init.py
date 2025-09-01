import os

from dpvo.dpvo_colmap_init import run_colmap_initialization

if __name__ == "__main__":
    import argparse
    import time

    def test_colmap_initialization():
        """Test script for run_colmap_initialization function with CLI arguments"""
        parser = argparse.ArgumentParser(description='Test COLMAP initialization with flow detection')
        parser.add_argument('--imagedir', type=str, required=True,
                          help='Directory containing input images')
        parser.add_argument('--output_path', type=str,
                          default='./colmap_test_output',
                          help='Output directory for COLMAP initialization (default: ./colmap_test_output)')
        parser.add_argument('--skip', type=int, default=0,
                          help='Number of images to skip at the beginning (default: 0)')
        parser.add_argument('--warmup_frames', type=int, default=50,
                          help='Number of frames for COLMAP initialization (default: 50)')
        parser.add_argument('--init_stride', type=int, default=2,
                          help='Stride for image sampling (default: 2)')
        parser.add_argument('--flow_threshold', type=float, default=2.0,
                          help='Minimum optical flow threshold for image selection (default: 2.0)')
        parser.add_argument('--use_flow_selection', action='store_true', default=True,
                          help='Use flow-based image selection (default: True)')
        parser.add_argument('--use_sharpness_selection', dest='use_flow_selection',
                          action='store_false',
                          help='Use sharpness-based image selection instead of flow-based')
        parser.add_argument('--test_mode', action='store_true',
                          help='Run in test mode (shows parameters without executing COLMAP)')

        args = parser.parse_args()

        print("=" * 60)
        print("COLMAP Initialization Test Script")
        print("=" * 60)
        print(f"Image directory: {args.imagedir}")
        print(f"Output path: {args.output_path}")
        print(f"Skip frames: {args.skip}")
        print(f"Warmup frames: {args.warmup_frames}")
        print(f"Init stride: {args.init_stride}")
        print(f"Flow threshold: {args.flow_threshold}")
        print(f"Use flow selection: {args.use_flow_selection}")
        print(f"Test mode: {args.test_mode}")
        print("-" * 60)

        # Validate input directory
        if not os.path.exists(args.imagedir):
            print(f"ERROR: Image directory '{args.imagedir}' does not exist!")
            return

        # Check for images in directory
        from itertools import chain
        from pathlib import Path
        img_exts = ["*.png", "*.jpeg", "*.jpg"]
        image_files = list(chain.from_iterable(Path(args.imagedir).glob(e) for e in img_exts))

        if len(image_files) == 0:
            print(f"ERROR: No images found in directory '{args.imagedir}'!")
            return

        print(f"Found {len(image_files)} images in input directory")

        if args.test_mode:
            print("\nTEST MODE - Parameters validated successfully!")
            print("To run actual COLMAP initialization, remove --test_mode flag")
            return

        try:
            print(f"\nStarting COLMAP initialization...")
            start_time = time.time()

            # Call the run_colmap_initialization function
            cam_params = run_colmap_initialization(
                imagedir=args.imagedir,
                path=args.output_path,
                skip=args.skip,
                warmup_frames=args.warmup_frames,
                init_stride=args.init_stride,
                flow_threshold=args.flow_threshold,
                use_flow_selection=args.use_flow_selection
            )

            end_time = time.time()
            duration = end_time - start_time

            print("\n" + "=" * 60)
            print("COLMAP INITIALIZATION COMPLETED")
            print("=" * 60)
            print(f"Duration: {duration:.2f} seconds")
            print(f"Estimated intrinsics:")
            print("estimated camera parameters")
            print(f"  fx = {cam_params[0]:.4f}")
            print(f"  fy = {cam_params[1]:.4f}")
            print(f"  cx = {cam_params[2]:.4f}")
            print(f"  cy = {cam_params[3]:.4f}")
            if len(cam_params) > 4:
                print(f"  k1 = {cam_params[4]:.4f}")
                print(f"  k2 = {cam_params[5]:.4f}")
                print(f"  p1 = {cam_params[6]:.4f}")
                print(f"  p2 = {cam_params[7]:.4f}")
            print(f"Output saved to: {args.output_path}/initialized/")

        except Exception as e:
            print(f"\nERROR during COLMAP initialization: {e}")
            import traceback
            traceback.print_exc()

    # Example usage information
    def print_usage_examples():
        """Print usage examples for the test script"""
        examples = [
            "# Test with flow-based selection (default)",
            "python dpvo_colmap_init.py --imagedir /path/to/images",
            "",
            "# Test with custom parameters",
            "python dpvo_colmap_init.py --imagedir /path/to/images --warmup_frames 30 --flow_threshold 1.5",
            "",
            "# Use sharpness-based selection instead of flow",
            "python dpvo_colmap_init.py --imagedir /path/to/images --use_sharpness_selection",
            "",
            "# Test mode (validate parameters without running COLMAP)",
            "python dpvo_colmap_init.py --imagedir /path/to/images --test_mode",
            "",
            "# Full example with EuRoC dataset",
            "python dpvo_colmap_init.py --imagedir /media/shuo/T7/rgbd_slam/euroc/MH_03_medium/mav0/cam0/data --output_path ./euroc_test --warmup_frames 50 --flow_threshold 2.0"
        ]

        print("\nUsage Examples:")
        print("-" * 40)
        for example in examples:
            print(example)

    # Check if help is requested
    import sys
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print_usage_examples()

    # Run the test
    test_colmap_initialization()