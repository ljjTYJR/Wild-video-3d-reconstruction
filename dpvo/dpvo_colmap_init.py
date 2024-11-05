# deep image matching-related library
import os
from importlib import import_module
from pathlib import Path
import yaml
import multiprocessing
from deep_image_matching import logger, timer
from deep_image_matching.config import Config
from deep_image_matching.image_matching import ImageMatching
from deep_image_matching.io.h5_to_db import export_to_colmap
from deep_image_matching.io.h5_to_openmvg import export_to_openmvg
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary, read_cameras_binary
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

        use_pycolmap = True
        try:
            # To be sure, check if pycolmap is available, otherwise skip reconstruction
            pycolmap = import_module("pycolmap")
            logger.info(f"Using pycolmap version {pycolmap.__version__}")
        except ImportError:
            logger.error("Pycomlap is not available.")
            use_pycolmap = False

        if use_pycolmap:
            # import reconstruction module
            reconstruction = import_module("deep_image_matching.reconstruction")
            reconst_opts = pycolmap.IncrementalPipelineOptions()
            reconst_opts.num_threads=min(multiprocessing.cpu_count(), 16)

            # Run reconstruction
            model = reconstruction.main(
                database=self.output_dir / "database.db",
                image_dir=self.imgs_dir,
                sfm_dir=self.output_dir,
                reconst_opts=reconst_opts,
                verbose=self.colmap_cfg.general["verbose"],
            )
        # use_glomap=True
        # if use_glomap:
        #     # glomap mapper --database_path 'database.db' --image_path '../images' --output_path 'glomap_models'
        #     cmd = f"glomap mapper --database_path {database_path} --image_path {self.imgs_dir} --output_path {self.output_dir}/glomap"
        #     os.system(cmd)

        num_imgs = model.num_reg_images() # number of registered images for the largest model
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
        if os.path.exists(camera_txt):
            with open(camera_txt, "r") as file:
                lines = file.readlines()
                for line in lines:
                    if "SIMPLE_PINHOLE" in line:
                        line = line.split()
                        f, cx, cy = float(line[4]), float(line[5]), float(line[6])
                        break
        else:
            camera_bin = target_dir / "cameras.bin"
            cam_id_to_camera = read_cameras_binary(camera_bin)
            f, _, cx , cy = cam_id_to_camera[1].params
        fx, fy = f, f
        cx, cy = cx, cy
        return fx, fy, cx, cy

if __name__ == "__main__":
    img_dir = '/media/shuo/T7/duslam/video_images/china_walking_park/seq1/first50'
    dpvo_colmap_init = DPVOColmapInit(img_dir)
    dpvo_colmap_init.run()