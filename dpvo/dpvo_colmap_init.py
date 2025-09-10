# deep image matching-related library
import os
import shutil
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import yaml
from deep_image_matching import logger, timer
from deep_image_matching.config import Config
from deep_image_matching.image_matching import ImageMatching
from deep_image_matching.io.h5_to_db import export_to_colmap
from deep_image_matching.io.h5_to_openmvg import export_to_openmvg
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_binary,
    read_images_binary,
)

from dpvo.utils import evaluate_sharpness, measure_motion


class COLMAP_ARGS:
    def __init__(self, dir):
        self.gui = False
        self.dir = dir
        self.images = None
        self.pipeline = 'superpoint+lightglue'
        self.config_file= "colmap_configs/superpoint+lightglue.yaml"
        self.quality="high"
        self.tiling="none"
        self.strategy="sequential"
        self.pair_file=None
        self.overlap=10
        self.global_feature="netvlad"
        self.db_path=None
        self.upright=False
        self.skip_reconstruction=False
        self.force=True
        self.verbose=False
        self.graph=False
        self.openmvg=None
        self.camera_options="colmap_configs/cameras.yaml"

    def output(self):
        # return as a dictionary, the keys are self.name
        return self.__dict__

class DPVOColmapInit:
    def __init__(self, imgs_dir):
        self.dir = Path(imgs_dir)
        # self.output_dir = self.imgs_dir.parent.joinpath("colmap")
        # self.pair_file = self.output_dir.joinpath("pairs.txt")
        self.colmap_args = COLMAP_ARGS(self.dir).output()
        self.colmap_cfg = Config(self.colmap_args)
        self.imgs_dir = self.colmap_cfg.general["image_dir"]
        self.output_dir = self.colmap_cfg.general["output_dir"]

    def run(self):
        # Initialize ImageMatching class
        img_matching = ImageMatching(
            imgs_dir=self.imgs_dir,
            output_dir=self.output_dir,
            matching_strategy=self.colmap_cfg.general["matching_strategy"],
            local_features=self.colmap_cfg.extractor["name"],
            matching_method=self.colmap_cfg.matcher["name"],
            pair_file=self.colmap_cfg.general["pair_file"],
            retrieval_option=self.colmap_cfg.general["retrieval"],
            overlap=self.colmap_cfg.general["overlap"],
            existing_colmap_model=self.colmap_cfg.general["db_path"],
            custom_config=self.colmap_cfg.as_dict(),
        )

        # Generate pairs to be matched
        pair_path = img_matching.generate_pairs()
        timer.update("generate_pairs")

        # Try to rotate images so they will be all "upright", useful for deep-learning approaches that usually are not rotation invariant
        if self.colmap_cfg.general["upright"]:
            img_matching.rotate_upright_images()
            timer.update("rotate_upright_images")

        # Extract features
        feature_path = img_matching.extract_features()
        timer.update("extract_features")

        # Matching
        match_path = img_matching.match_pairs(feature_path)
        timer.update("matching")

        # If features have been extracted on "upright" images, this function bring features back to their original image orientation
        if self.colmap_cfg.general["upright"]:
            img_matching.rotate_back_features(feature_path)
            timer.update("rotate_back_features")

        # Export in colmap format
        with open(self.colmap_cfg.general["camera_options"], "r") as file:
            camera_options = yaml.safe_load(file)
        database_path = self.output_dir / "database.db"
        export_to_colmap(
            img_dir=self.imgs_dir,
            feature_path=feature_path,
            match_path=match_path,
            database_path=database_path,
            camera_options=camera_options,
        )
        timer.update("export_to_colmap")

        use_pycolmap = True # by default, using COLMAP
        try:
            # To be sure, check if pycolmap is available, otherwise skip reconstruction
            pycolmap = import_module("pycolmap")
            logger.info(f"Using pycolmap version {pycolmap.__version__}")
        except ImportError:
            logger.error("Pycomlap is not available.")
            use_pycolmap = False

        model = None
        if use_pycolmap:
            # import reconstruction module
            reconstruction = import_module("deep_image_matching.reconstruction")
            # reconst_opts = pycolmap.IncrementalPipelineOptions()
            # reconst_opts.num_threads=min(multiprocessing.cpu_count(), 16)
            reconst_opts = {}

            # Run reconstruction
            try:
                model = reconstruction.main(
                    database=self.output_dir / "database.db",
                    image_dir=self.imgs_dir,
                    sfm_dir=self.output_dir,
                    reconst_opts=reconst_opts,
                    verbose=self.colmap_cfg.general["verbose"],
                )
            except Exception as e:
                logger.error(f"Reconstruction failed: {e}")
                use_pycolmap = False

        if not use_pycolmap:
            logger.error("COLMAP failed to initialize, switching to glomap.")
            use_glomap = True

        num_imgs = model.num_reg_images() if model else 0
        num_all_imgs = len(list(self.imgs_dir.glob("*")))
        target_dir = self.output_dir / "reconstruction"
        if num_imgs < num_all_imgs * 0.7:
            logger.warning(f"Only {num_imgs} images registered out of {num_all_imgs}.")

            logger.info("Switch to the glomap mapper.")
            cmd = f"glomap mapper --database_path {database_path} --image_path {self.imgs_dir} --output_path {self.output_dir}/glomap_model"
            os.system(cmd)
            # check the resulted-in the glomap model.
            image_bin_file = f"{self.output_dir}/glomap_model/0/images.bin"
            img_id_to_image = read_images_binary(image_bin_file)
            if len(img_id_to_image) < num_all_imgs * 0.7:
                logger.error("Glomap failed to register enough images.")
                return
            else:
                target_dir = Path(f"{self.output_dir}/glomap_model/0")

        camera_txt = target_dir / "cameras.txt"
        cam_params = []
        if os.path.exists(camera_txt):
            with open(camera_txt, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if "SIMPLE_PINHOLE" in line:
                        line = line.split()
                        f, cx, cy = float(line[4]), float(line[5]), float(line[6])
                        cam_params.extend([f, f, cx, cy])
                        break
                    elif "OPENCV" in line:
                        line = line.split()
                        fx, fy, cx, cy, k1, k2, p1, p2 = float(line[4]), float(line[5]), float(line[6]), \
                            float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11])
                        cam_params.extend([fx, fy, cx, cy, k1, k2, p1, p2])
                        break
        else:
            camera_bin = target_dir / "cameras.bin"
            cam_id_to_camera = read_cameras_binary(camera_bin)
            if cam_id_to_camera[1].model == "SIMPLE_PINHOLE":
                f, cx, cy = cam_id_to_camera[1].params
                cam_params.extend([f, f, cx, cy])
            elif cam_id_to_camera[1].model == "OPENCV":
                fx, fy, cx, cy, k1, k2, p1, p2 = cam_id_to_camera[1].params
                cam_params.extend([fx, fy, cx, cy, k1, k2, p1, p2])
            else:
                f, _, cx , cy = cam_id_to_camera[1].params
        return cam_params

def run_colmap_initialization(imagedir, output_path, skip=0, warmup_frames=50, init_stride=2, flow_threshold=2.0,
                              use_flow_selection=True):
    colmap_initial_path = f"{output_path}/initialized/"
    colmap_initial_images = f"{colmap_initial_path}/images"
    os.makedirs(colmap_initial_images, exist_ok=True)

    from itertools import chain
    from pathlib import Path

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::init_stride]

    if len(image_list) < warmup_frames:
        raise ValueError(f"Number of images in the directory is less than {warmup_frames}")

    selected_images = []

    if use_flow_selection:
        # Use flow-based selection similar to motion_flow.py
        print(f"Using flow-based image selection (threshold: {flow_threshold})")
        i = 0
        while i < len(image_list) and len(selected_images) < warmup_frames:
            base_img_path = str(image_list[i])
            base_img = cv2.imread(base_img_path, cv2.IMREAD_COLOR)

            if base_img is None:
                i += 1
                continue

            # Check sharpness first
            sharpness = evaluate_sharpness(base_img)
            if sharpness < 30.0:  # Lower threshold for basic quality
                i += 1
                continue

            # Try to find an image with sufficient flow
            found_match = False
            for offset in range(1, min(20, len(image_list) - i)):
                compare_idx = i + offset
                compare_img_path = str(image_list[compare_idx])
                compare_img = cv2.imread(compare_img_path, cv2.IMREAD_COLOR)

                if compare_img is None:
                    continue

                # Measure motion between images
                mean_flow, mean_flow_scaled, _, mean_norm, _, _ = measure_motion(base_img, compare_img, threshold=1.0)

                if mean_flow_scaled >= flow_threshold:
                    # Found sufficient flow - save both images
                    cv2.imwrite(f"{colmap_initial_images}/{len(selected_images):06d}.png", base_img)
                    selected_images.append(image_list[i])

                    if len(selected_images) < warmup_frames:
                        cv2.imwrite(f"{colmap_initial_images}/{len(selected_images):06d}.png", compare_img)
                        selected_images.append(image_list[compare_idx])

                    print(f"Selected pair with flow {mean_flow_scaled:.2f}: images {i} and {compare_idx}")
                    i = compare_idx + 1
                    found_match = True
                    break

            if not found_match:
                i += 1

    else:
        # Original sharpness-based selection
        print("Using sharpness-based image selection")
        sharpness_threshold = 50.0

        for imfile in image_list:
            if len(selected_images) >= warmup_frames:
                break

            image = cv2.imread(str(imfile), cv2.IMREAD_COLOR)
            sharpness = evaluate_sharpness(image)
            if sharpness > sharpness_threshold:
                cv2.imwrite(f"{colmap_initial_images}/{len(selected_images):06d}.png", image)
                selected_images.append(imfile)

    # Fallback if not enough images selected
    if len(selected_images) < warmup_frames:
        print(f"Warning: Only found {len(selected_images)} suitable images out of {warmup_frames} required")
        print("Filling remaining slots with any available images...")
        remaining_needed = warmup_frames - len(selected_images)
        available_images = [img for img in image_list if img not in selected_images]

        for imfile in available_images[:remaining_needed]:
            image = cv2.imread(str(imfile), cv2.IMREAD_COLOR)
            if image is not None:
                cv2.imwrite(f"{colmap_initial_images}/{len(selected_images):06d}.png", image)
                selected_images.append(imfile)

    print(f"Selected {len(selected_images)} images for COLMAP initialization")

    init_recon = DPVOColmapInit(colmap_initial_path)
    cam_params = init_recon.run()
    print(f"COLMAP initialization: fx={cam_params[0]}, fy={cam_params[1]}, cx={cam_params[2]}, cy={cam_params[3]}")
    return np.array(cam_params)

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
            print(f"  fx = {fx:.2f}")
            print(f"  fy = {fy:.2f}")
            print(f"  cx = {cx:.2f}")
            print(f"  cy = {cy:.2f}")
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