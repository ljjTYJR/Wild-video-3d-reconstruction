from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from pathlib import Path
import os
from loguru import logger

from formatter.colmap_utilis import colmap_to_json
from itertools import chain

def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj, tstamps = args
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)

def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: ' '.join(map(str, a))
    with Path(filename).open('w') as f:
        for i in range(traj.num_poses):
            f.write(f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[1,2,3,0]])}\n")
    print(f"Saved {filename}")

def save_output_for_COLMAP(name: str, tstamp: np.ndarray, traj: PoseTrajectory3D, points: np.ndarray, colors: np.ndarray, nerf_studio_format, fx, fy, cx, cy, H=480, W=640):
    """ Saves the sparse point cloud and camera poses such that it can be opened in COLMAP """

    colmap_dir = Path(name)
    colmap_dir.mkdir(exist_ok=True)
    scale = 1.0 # for visualization

    logger.info(f"Saving COLMAP-compatible reconstruction in {colmap_dir.resolve()}")

    original_image_path = Path(name).parent.joinpath("images")
    traj = PoseTrajectory3D(poses_se3=list(map(np.linalg.inv, traj.poses_se3)), timestamps=traj.timestamps)

    image_list=None
    if os.path.exists(original_image_path):
        img_exts = ["*.png", "*.jpeg", "*.jpg"]
        image_list = sorted(chain.from_iterable(Path(original_image_path).glob(e) for e in img_exts))
        if not image_list:
            logger.error(f"No images found in {original_image_path}")
            return
    images = ""
    for tstamp, idx, (x,y,z), (qw, qx, qy, qz) in zip(tstamp, range(1,traj.num_poses+1), traj.positions_xyz*scale, traj.orientations_quat_wxyz):
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME (https://colmap.github.io/format.html)
        img_name = image_list[int(tstamp)].name if image_list else "image"
        images += f"{idx} {qw} {qx} {qy} {qz} {x} {y} {z} 1 {img_name}\n\n"
    (colmap_dir / "images.txt").write_text(images)

    # points
    points3D = ""
    colors_uint = (colors * 255).astype(np.uint8).tolist()
    for i, (p,c) in enumerate(zip((points*scale).tolist(), colors_uint), start=1):
        points3D += f"{i} " + ' '.join(map(str, p + c)) + " 0.0 0 0 0 0 0 0\n"
    (colmap_dir / "points3D.txt").write_text(points3D)

    # camera
    (colmap_dir / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}")

    if nerf_studio_format:
        nerf_studio_colmap_dir = colmap_dir / "colmap/sparse/0"
        if os.path.exists(nerf_studio_colmap_dir):
            os.system(f"rm -rf {nerf_studio_colmap_dir}/*")
        else:
            nerf_studio_colmap_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"colmap model_converter --input_path {colmap_dir} --output_path {nerf_studio_colmap_dir} --output_type BIN"
        os.system(cmd)
        logger.info(f"Saved COLMAP-compatible reconstruction in {nerf_studio_colmap_dir.resolve()}")

        # colmap_image_path = colmap_dir / "images"
        # if not os.path.exists(colmap_image_path):
            # extfat does not support symlinks
            # logger.info(f"Linked images from {original_image_path} to {colmap_image_path}")
            # os.system(f"ln -s {original_image_path} {colmap_image_path}")
            # os.system(f"cp -r {original_image_path} {colmap_image_path}")
            # logger.info(f"Copied images from {original_image_path} to {colmap_image_path}")

        # create json file
        colmap_to_json(nerf_studio_colmap_dir, colmap_dir)

    print(f"Saved COLMAP-compatible reconstruction in {colmap_dir.resolve()}")