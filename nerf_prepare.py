"""
Prepare the achieved DPVO dataset for NeRF-facto training
Specifically, we want to specify the start index and the end index in the dataset.

The output is the directory with the transforms.json file preparing for the NeRF training
"""
import numpy as np
import json
import os
import argparse
from pathlib import Path
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
)
from nerfstudio.utils.rich_utils import CONSOLE
from formatter.colmap_utilis import parse_colmap_camera_params
from os import path

class NeRFPrepare:
    def __init__(self, db_path, start_idx, end_idx):
        self.dataset_dir = db_path
        self.recon_dir = Path(db_path) / "colmap/sparse/0"
        if not os.path.exists(self.recon_dir):
            print("Not the DPVO dataset, the COLMAP dataset")
            self.recon_dir = Path(db_path) / "reconstruction"
        self.start_idx = start_idx
        self.end_idx = end_idx

        parts = self.dataset_dir.split("/")
        sub_paths = parts[-3:-1]
        self.output_dir = Path(self.dataset_dir) / f"{sub_paths[0]}_{sub_paths[1]}_select_{self.start_idx}_{self.end_idx}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.keep_original_world_coordinate = False
        self.image_rename_map = None

    def generate_nf_transform(self):
        cam_id_to_camera = read_cameras_binary(self.recon_dir / "cameras.bin")
        img_id_to_image = read_images_binary(self.recon_dir / "images.bin")
        print("finish reading files")

        use_single_camera_mode = True
        if set(cam_id_to_camera.keys()) != {1}:
            CONSOLE.print(f"[bold yellow]Warning: More than one camera is found in {self.recon_dir}")
            print(cam_id_to_camera)
            use_single_camera_mode = False  # update bool: one camera per frame
            out = {}  # out = {"camera_model": parse_colmap_camera_params(cam_id_to_camera[1])["camera_model"]}
        else:  # one camera for all frames
            out = parse_colmap_camera_params(cam_id_to_camera[1])

        frames = []
        for img_id, im_data in img_id_to_image.items():
            if img_id < self.start_idx or img_id > self.end_idx:
                continue
            else:
                # TODO(1480) BEGIN use pycolmap API
                # rotation = im_data.rotation_matrix()
                rotation = qvec2rotmat(im_data.qvec)

                translation = im_data.tvec.reshape(3, 1)
                w2c = np.concatenate([rotation, translation], 1)
                w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
                c2w = np.linalg.inv(w2c)
                # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
                c2w[0:3, 1:3] *= -1
                if not self.keep_original_world_coordinate:
                    c2w = c2w[np.array([0, 2, 1, 3]), :]
                    c2w[2, :] *= -1

                name = im_data.name
                if self.image_rename_map is not None:
                    name = self.image_rename_map[name]
                # name = Path(f"./images/{name}")
                name = Path(f"../../images/{name}")

                frame = {
                    "file_path": name.as_posix(),
                    "transform_matrix": c2w.tolist(),
                    "colmap_im_id": img_id,
                }
                # todo: mask and depth

                if not use_single_camera_mode:  # add the camera parameters for this frame
                    frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

                frames.append(frame)

        out["frames"] = frames
        applied_transform = None
        if not self.keep_original_world_coordinate:
            applied_transform = np.eye(4)[:3, :]
            applied_transform = applied_transform[np.array([0, 2, 1]), :]
            applied_transform[2, :] *= -1
            out["applied_transform"] = applied_transform.tolist()

        with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)


# test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='/media/shuo/T7/duslam/video_images/china_walking_park/seq4/dpvo_colmap_20241031-163917') # the path to the colmap dataset
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10_000)
    args = parser.parse_args()
    nerf_prepare = NeRFPrepare(args.db_path, args.start_idx, args.end_idx)
    nerf_prepare.generate_nf_transform()