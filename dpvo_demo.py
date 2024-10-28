import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from plyfile import PlyElement, PlyData
from evo.core.trajectory import PoseTrajectory3D
from loguru import logger
import datetime

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format, save_output_for_COLMAP

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(
    cfg,
    network,
    imagedir,
    calib,
    stride=1,
    skip=0,
    viz=False,
    timeit=False,
    save_reconstruction=False,
    mast3r=False,
    colmap_init=False,
    motion_filter=False,
    path=None):

    slam = None
    queue = Queue(maxsize=8)

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break
        print(f"Processing frame {t}")
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz, mast3r=mast3r, colmap_init=colmap_init, motion_filter=motion_filter, path=path)

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    reader.join()

    if save_reconstruction:
        points = slam.points_.cpu().numpy()[:slam.m]
        colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
        points = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                          dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        el = PlyElement.describe(points, 'vertex',{'some_property': 'f8'},{'some_property': 'u4'})
        return slam.terminate(), PlyData([el], text=True)

    points, colors, (intrinsic, H, W) = slam.get_pts_clr_intri()

    # (poses, tstamps), (points, colors, (intrinsic, h, w))
    return slam.terminate(), (points, colors, (*intrinsic, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='dpvo')
    parser.add_argument('--network', type=str, default='checkpoints/dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--buffer', type=int, default=1024)
    parser.add_argument('--config', default="dpvo_configs/default.yaml")
    parser.add_argument('--mast3r', action="store_true")
    parser.add_argument('--colmap_init', action="store_true")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_reconstruction', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--motion_filter', action="store_true")
    parser.add_argument('--export_colmap', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BUFFER_SIZE = args.buffer

    # create the logging file
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(f"experiments/{args.exp_name}_{current_time}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f'{exp_dir}/exp.log',
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
                level="INFO")

    print("Running with config...")
    logger.info(cfg)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = (Path(args.imagedir).parent).joinpath(f"dpvo_colmap_{time}")

    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit, args.save_reconstruction,
                    args.mast3r, args.colmap_init, args.motion_filter, path)
    name = Path(args.imagedir).stem
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    # if args.save_reconstruction:
    #     pred_traj, ply_data = pred_traj
    #     ply_data.write(f"{name}.ply")
    #     print(f"Saved {name}.ply")

    # if args.save_trajectory:
    #     Path("saved_trajectories").mkdir(exist_ok=True)
    #     save_trajectory_tum_format(pred_traj, f"saved_trajectories/{name}.txt")

    # if args.plot:
    #     Path("trajectory_plots").mkdir(exist_ok=True)
    #     plot_trajectory(pred_traj, title=f"DPVO Trajectory Prediction for {name}", filename=f"trajectory_plots/{name}.pdf")

    if args.export_colmap:
        save_output_for_COLMAP(path, tstamps, trajectory, points, colors, True, *calib)