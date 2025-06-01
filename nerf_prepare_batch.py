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

Datasets = {
    # 'YanshanPark':{
    #     "clips":[
    #         [0, 500],
    #         [500, 1000],
    #         [1000, 1500],
    #         [1500, 2000],
    #         [2000, 2500],
    #         [2500, 3000],
    #     ],
    #     "colmap":'/media/shuo/T7/duslam/video_images/china_classical_park_2k/custom_sfm_0_3327/colmap_model/0',
    #     "colmap_scale":0.25,
    #     "glomap":'/media/shuo/T7/duslam/video_images/china_classical_park_2k/custom_sfm_0_3327/glomap_default/0',
    #     "glomap_scale":0.25,
    #     "ours":'/media/shuo/T7/duslam/video_images/china_classical_park_2k/custom_sfm_0_3327/dpvo/dpvo_2k_retri',
    #     "ours_scale":0.25,
    #     "output_dir":'/media/shuo/T7/duslam/video_images/china_classical_park_512/nerf_eval',
    #     'frames':False
    # },

    # 'TaicangPark':{
    #     "clips":[
    #         [0, 500],
    #         [500, 1000],
    #         [1000, 1500],
    #         [1500, 2000],
    #         [2000, 2500],
    #     ],
    #     "colmap":'/media/shuo/T7/duslam/video_images/youtube_taicang_LJf7LKLvmUc/image/2k/colmap_fast/0',
    #     "colmap_scale":0.25,
    #     "glomap":'/media/shuo/T7/duslam/video_images/youtube_taicang_LJf7LKLvmUc/image/2k/glomap_default/0',
    #     "glomap_scale":0.25,
    #     "ours":'/media/shuo/T7/duslam/video_images/youtube_taicang_LJf7LKLvmUc/image/2k/dpvo_2k/re_tri_2',
    #     "ours_scale":0.25,
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_taicang_LJf7LKLvmUc/image/512/nerf_eval',
    #    'frames':False
    # },

    # 'Upplasa':{
    #     "clips":[
    #         [0, 500],
    #         [500, 1000],
    #         [1000, 1500],
    #         [1500, 2000],
    #         [2000, 2500],
    #     ],
    #     "colmap":'/media/shuo/T7/duslam/video_images/youtube_upplasa_aVh_jTIP2cE/image/2k/colmap_fast/1',
    #     "colmap_scale":0.25,
    #     "glomap":'/media/shuo/T7/duslam/video_images/youtube_upplasa_aVh_jTIP2cE/image/2k/glomap_default/0',
    #     "glomap_scale":0.25,
    #     "ours":'/media/shuo/T7/duslam/video_images/youtube_upplasa_aVh_jTIP2cE/image/2k/dpvo/dense/re_tri',
    #     "ours_scale":0.25,
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_upplasa_aVh_jTIP2cE/image/512/nerf_eval',
    #      'frames':False
    # },

    # TODO: Nan xun

    # TODO there is problem with it!
    # 'Helsi':{
    #     "clips":[
    #         [0, 500],
    #         [500, 1000],
    #         [1000, 1500],
    #         [1500, 2000],
    #         [2000, 2500],
    #     ],
    #     # TODO: manually modify the json file!
    #     "colmap":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/2k/colmap_0_2700/0',
    #     "colmap_scale":1.0,
    #     "glomap":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/2k/glomap_0_2700/0',
    #     "glomap_scale":1.0,
    #     "ours":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/dpvo_colmap_20250226-010644_0_2700/colmap/sparse/0',
    #     "ours_scale":1.0,
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/nerf_eval',
    #     'frames':False
    # },

    # TODO manually modify the json file!
    'Helsi2':{
        "clips":[
            [0, 500],
            [500, 1000],
            [1000, 1500],
            [1500, 2000],
            [2000, 2500],
        ],
        # "colmap":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/2k/colmap_2699_5400_fast/1',
        # "colmap_scale":1.0,
        # "glomap":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/2k/glomap_2699_5400/0',
        # "glomap_scale":1.0,
        "ours":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/dpvo_colmap_20250227-111036_2699_None/colmap/sparse/0',
        "ours_scale":1.0,
        "output_dir":'/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/nerf_eval_2',
        'frames':False
    },



    # 'Lund':{
    #     "clips":[
    #         [1, 500],
    #         [500, 1000],
    #         [1000, 1500],
    #         [1500, 2000],
    #         [2000, 2500],
    #     ],
    #     "colmap":'/media/shuo/T7/duslam/video_images/youtube_lund_Nhc5BNlfDms/images/2k/colmap_fast/0',
    #     "colmap_scale":0.25,
    #     "glomap":'/media/shuo/T7/duslam/video_images/youtube_lund_Nhc5BNlfDms/images/2k/glomap_default/0',
    #     "glomap_scale":0.25,
    #     "ours":'/media/shuo/T7/duslam/video_images/youtube_lund_Nhc5BNlfDms/images/2k/dpvo_2k/re_tri',
    #     "ours_scale":0.25,
    #     "output_dir":'/media/shuo/T7/duslam/video_images/youtube_lund_Nhc5BNlfDms/images/512/nerf_eval',
    #     'frames':True
    # },
}

class NeRFPrepare:
    def __init__(self, db_path, start_idx, end_idx, intrinsic_scale, output_path):
        self.keep_original_world_coordinate = False
        self.image_rename_map = None

    def prepare_nerfdataset(self, recon_dir, out_dir, start_idx, end_idx, scale=1.0, frame_name=False):
        cam_id_to_camera = read_cameras_binary(Path(recon_dir) / "cameras.bin")
        img_id_to_image = read_images_binary(Path(recon_dir) / "images.bin")

        use_single_camera_mode = True
        if set(cam_id_to_camera.keys()) != {1}:
            CONSOLE.print(f"[bold yellow]Warning: More than one camera is found in {self.recon_dir}")
            print(cam_id_to_camera)
            use_single_camera_mode = False  # update bool: one camera per frame
            out = {}  # out = {"camera_model": parse_colmap_camera_params(cam_id_to_camera[1])["camera_model"]}
        else:  # one camera for all frames
            out = parse_colmap_camera_params(cam_id_to_camera[1])
            # rescale the camera intrinsic
            out["w"] *= scale
            out["h"] *= scale
            out["fl_x"] *= scale
            out["fl_y"] *= scale
            out["cx"] *= scale
            out["cy"] *= scale

            # # Just debug
            # out["w"] = 512.0
            # out["h"] = 288.0
            # out["cx"] = 256.0
            # out["cy"] = 144.0

        frames = []
        for img_id, im_data in img_id_to_image.items():
            if img_id < start_idx or img_id > end_idx:
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
                # name = Path(f"./images/{name}") # todo:!

                # todo: just for this
                name = Path(f"../../../images/{img_id+2698:06d}.png")
                # TODO recover it
                # name = Path(f"../../../images/{name}")

                frame = {
                    "file_path": name.as_posix(),
                    "transform_matrix": c2w.tolist(),
                    "colmap_im_id": img_id,
                }
                # todo: mask and depth

                if not use_single_camera_mode:  # add the camera parameters for this frame
                    frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

                frames.append(frame)

        collected_frame_ids = [frame["colmap_im_id"] for frame in frames]
        if len(collected_frame_ids) == 0:
            print(f"No images found in the dataset from {start_idx} to {end_idx}")
            return
        min_collect_frame_id = min(collected_frame_ids)
        # add images if there is no enough images in the dataset!
        for i in range(start_idx, end_idx):
            if i not in collected_frame_ids:
                if frame_name:
                    name = f"../../../images/frames_{i+2699:06d}.png"
                else:
                    name = f"../../../images/{i+2699:06d}.png"
                frame = {
                    "file_path": name,
                    "transform_matrix": np.eye(4).tolist(),
                    "colmap_im_id": i,
                }
                if not use_single_camera_mode:
                    frame.update(parse_colmap_camera_params(cam_id_to_camera[1]))
                frames.append(frame)

        out["frames"] = frames
        applied_transform = None
        if not self.keep_original_world_coordinate:
            applied_transform = np.eye(4)[:3, :]
            applied_transform = applied_transform[np.array([0, 2, 1]), :]
            applied_transform[2, :] *= -1
            out["applied_transform"] = applied_transform.tolist()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(out_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    def generate_nf_transform(self):
        # YanshanPark
        for dataset in Datasets:
            for clip in Datasets[dataset]["clips"]:
                start_idx = clip[0]
                end_idx = clip[1]
                print(f"Processing {dataset} clip {start_idx} to {end_idx}")

                output_dir = Path(Datasets[dataset]["output_dir"]) / f"select_{start_idx}_{end_idx}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                # make dataset separately
                # self.prepare_nerfdataset(Datasets[dataset]["colmap"], output_dir/f"colmap", start_idx, end_idx, Datasets[dataset]["colmap_scale"], Datasets[dataset]["frames"])
                # self.prepare_nerfdataset(Datasets[dataset]["glomap"], output_dir/f"glomap", start_idx, end_idx, Datasets[dataset]["glomap_scale"], Datasets[dataset]["frames"])
                self.prepare_nerfdataset(Datasets[dataset]["ours"], output_dir/f"ours", start_idx, end_idx, Datasets[dataset]["ours_scale"], Datasets[dataset]["frames"])

# test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='/media/shuo/T7/duslam/video_images/china_walking_park/seq4/dpvo_colmap_20241031-163917') # the path to the colmap dataset
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10_000)
    # Whether to re-scale the camera intrinsic parameter, since COLMAP runs on 2k while DPVO runs on 512.
    parser.add_argument('--intrinsic_scale', type=float, default=1.0)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    nerf_prepare = NeRFPrepare(args.db_path, args.start_idx, args.end_idx, args.intrinsic_scale, args.output_path)
    nerf_prepare.generate_nf_transform()