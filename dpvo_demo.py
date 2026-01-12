import atexit
import datetime
import glob
import os
import os.path as osp
import random
import signal
import sys
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from evo.core.trajectory import PoseTrajectory3D
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.dpvo_colmap_init import run_colmap_initialization
from dpvo.netvlad_retrieval import RetrievalNetVLADOffline
from dpvo.plot_utils import (
    plot_trajectory,
    save_output_for_COLMAP,
    save_trajectory_tum_format,
)
from dpvo.stream import image_stream, image_stream_limit, video_stream
from dpvo.utils import Timer


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def int_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer or 'None': {value}")

def seed_all(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_tensor_inputs(image, depth, mask, intrinsics):
    image = torch.from_numpy(image).permute(2,0,1).cuda()
    depth = torch.from_numpy(depth).cuda() if depth is not None else None
    mask = torch.from_numpy(mask).cuda() if mask is not None else None
    intrinsics = torch.from_numpy(intrinsics).cuda()
    return image, depth, mask, intrinsics

@torch.no_grad()
def run(
    cfg,
    network,
    imagedir,
    depthdir,
    maskdir,
    calib,
    stride=1,
    skip=0,
    viz=False,
    timeit=False,
    save_reconstruction=False,
    path=None,
    end=None,
    rerun=False):

    colmap_init = calib is None

    if colmap_init:
        calib = run_colmap_initialization(imagedir, path, skip)
    elif isinstance(calib, str):
        calib = np.loadtxt(calib, delimiter=" ")
    elif isinstance(calib, np.ndarray):
        calib = calib

    queue = Queue(maxsize=8)

    stream_target = image_stream if os.path.isdir(imagedir) else video_stream
    stream_args = (queue, imagedir, depthdir, maskdir, calib, stride, skip, end) if os.path.isdir(imagedir) else (queue, imagedir, calib, stride, skip)
    reader = Process(target=stream_target, args=stream_args)

    retrieval = None
    if cfg.loop_enabled:
        print("Extracting global descriptors...")
        retrieval = RetrievalNetVLADOffline(imagedir, skip, end, stride)
        retrieval.insert_img_offline()
        retrieval.end_and_clean()

    reader.start()
    slam = None

    with tqdm(desc="Processing frames", unit="frame") as pbar:
        while True:
            t, image, depth, mask, intrinsics = queue.get()
            if t < 0:
                break

            image, depth, mask, intrinsics = process_tensor_inputs(image, depth, mask, intrinsics)

            if slam is None:
                slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz, path=path, nvlad_db=retrieval, rerun=rerun)

            with Timer("SLAM", enabled=timeit):
                slam(t, image, depth, mask, intrinsics)

            pbar.update(1)

    for _ in range(12):
        slam.update()

    reader.join()

    # try to save the DPVO instance
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if save_reconstruction:
        points = slam.points_.cpu().numpy()[:slam.m]
        colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
        vertex_data = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                               dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        el = PlyElement.describe(vertex_data, 'vertex')
        return slam.terminate(), PlyData([el], text=True)

    points, colors, (intrinsic, H, W) = slam.get_pts_clr_intri(inlier=True)

    # (poses, tstamps), (points, colors, (intrinsic, h, w))
    return slam.terminate(), (points, colors, (*intrinsic, H, W))
    # return slam.terminate_keyframe(), (points, colors, (*intrinsic, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='checkpoints/dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--depthdir', type=str)
    parser.add_argument('--maskdir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--buffer', type=int, default=1024)
    parser.add_argument('--config', default="dpvo_configs/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_reconstruction', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--export_colmap', action="store_true")
    parser.add_argument('--set_seed', action="store_true")
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--end', type=int_or_none, default=None)
    parser.add_argument('--loop_enabled', action="store_true")
    parser.add_argument('--rerun', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BUFFER_SIZE = args.buffer
    cfg.loop_enabled = args.loop_enabled

    if args.set_seed:
        seed_all(42)

    torch.multiprocessing.set_start_method('spawn', force=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    output_path = Path(args.imagedir).parent / f"dpvo_colmap_{timestamp}_{args.skip}_{args.end}"

    (poses, tstamps), (points, colors, calib) = run(
        cfg, args.network, args.imagedir, args.depthdir, args.maskdir, args.calib,
        args.stride, args.skip, args.viz, args.timeit, args.save_reconstruction,
        output_path, args.end, args.rerun
    )

    sequence_name = Path(args.imagedir).stem
    trajectory = PoseTrajectory3D(
        positions_xyz=poses[:,:3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=tstamps
    )

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        save_trajectory_tum_format(trajectory, f"saved_trajectories/{sequence_name}.txt")

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory,
                        title=f"DPVO Trajectory Prediction for {sequence_name}",
                        filename=f"trajectory_plots/{sequence_name}.pdf")

    if args.export_colmap:
        save_output_for_COLMAP(output_path, tstamps, trajectory, points, colors, True, *calib)
        with open(f"{output_path}/config.yaml", "w") as f:
            f.write(cfg.dump())
            yaml.dump(vars(args), f, default_flow_style=False)

    print("DPVO execution completed successfully!")
