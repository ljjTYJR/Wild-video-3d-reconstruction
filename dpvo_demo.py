import cv2
import glob
import os
import torch
import random
import datetime
import numpy as np
import yaml
import os.path as osp
from pathlib import Path
from multiprocessing import Process, Queue
from plyfile import PlyElement, PlyData
from evo.core.trajectory import PoseTrajectory3D
from loguru import logger
from tqdm import tqdm

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream, image_stream_limit
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format, save_output_for_COLMAP
from dpvo.dpvo_colmap_init import DPVOColmapInit
from dpvo.netvlad_retrieval import RetrievalNetVLADOffline


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

def initialize_colmap(queue, path, warmup_frames=50):
    colmap_initial_path = f"{path}/initialized/"
    colmap_initial_images = f"{colmap_initial_path}/images"
    os.makedirs(colmap_initial_images, exist_ok=True)

    for _ in range(warmup_frames):
        t, image = queue.get()
        cv2.imwrite(f"{colmap_initial_images}/{t:06d}.png", image)

    init_recon = DPVOColmapInit(colmap_initial_path)
    colmap_fx, colmap_fy, colmap_cx, colmap_cy = init_recon.run()
    print(f"COLMAP initialization: fx={colmap_fx}, fy={colmap_fy}, cx={colmap_cx}, cy={colmap_cy}")

    while not queue.empty():
        queue.get()

    return np.array([colmap_fx, colmap_fy, colmap_cx, colmap_cy])

def run_colmap_initialization(imagedir, path, skip):
    # TODO: just read images instead of streaming
    queue = Queue(maxsize=8)
    init_stride = 2
    warmup_frames = 50
    reader = Process(target=image_stream_limit, args=(queue, imagedir, init_stride, skip))
    reader.start()
    calib = initialize_colmap(queue, path, warmup_frames)
    reader.join()
    return calib

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
    parser.add_argument('--exp_name', type=str, default='dpvo')
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
    exp_dir = Path(f"experiments/{args.exp_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.add(f'{exp_dir}/exp.log',
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
               level="INFO")
    logger.info(f"Running DPVO: {cfg}")

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