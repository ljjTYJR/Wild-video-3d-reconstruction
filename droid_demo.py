import sys

sys.path.append('droid_slam')

import argparse
import glob
import os
import time
from multiprocessing import Queue

import cv2
import lietorch
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch.multiprocessing import Process
from tqdm import tqdm

from dpvo.stream import video_stream
from droid_slam.droid import Droid


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, depthdir, calib, stride, start_idx=0, end_idx=None):
    """ image generator """

    K = np.eye(3)
    use_gt_calib = True if calib is not None else False
    if use_gt_calib:
        calib = np.loadtxt(calib, delimiter=" ")
        fx, fy, cx, cy = calib[:4]

        K[0,0] = fx
        K[0,2] = cx
        K[1,1] = fy
        K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))
    if end_idx is None:
        end_idx = len(image_list)
    image_list = image_list[start_idx:end_idx:stride]
    
    depth_list=None
    if depthdir:
        depth_list = sorted(os.listdir(depthdir))
        depth_list = depth_list[start_idx:end_idx:stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if use_gt_calib:
            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        # h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        # w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        max_dimension = max(h0, w0)
        if max_dimension!=512:
            h1 = int(h0 * 512/max_dimension)
            w1 = int(w0 * 512/max_dimension)
        else:
            h1 = h0
            w1 = w0

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        depth=None
        if depth_list:
            depth_file = os.path.join(depthdir, depth_list[t])
            # if npy file
            if depth_file.endswith(".npy"):
                depth = np.load(depth_file)
            else:
                depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                depth = depth.astype(np.float32) / 1000.0  # convert to meters
            depth = depth[:h1-h1%16, :w1-w1%16]
            depth_median = np.median(depth[depth > 0])
            depth[depth > 10 * depth_median] = 10 * depth_median
            depth = torch.as_tensor(depth).float()

        if use_gt_calib:
            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield t, image[None], depth, intrinsics
        else:
            yield t, image[None], depth, torch.zeros(4)


def save_colmap_from_w2c_traj(w2c_traj, intrinsics, output_path, image_dir=None, image_names=None):
    """Save reconstruction in COLMAP format from w2c trajectory (Nx7: [xyz, qx, qy, qz, qw])"""
    import os
    from pathlib import Path

    colmap_dir = Path(output_path)
    colmap_dir.mkdir(exist_ok=True, parents=True)

    N = w2c_traj.shape[0]

    # Extract positions and quaternions from w2c_traj
    positions = w2c_traj[:, :3]  # xyz
    quaternions_xyzw = w2c_traj[:, 3:]  # qx, qy, qz, qw

    # Convert quaternions from [qx, qy, qz, qw] to COLMAP format [qw, qx, qy, qz]
    quaternions_wxyz = np.column_stack([quaternions_xyzw[:, 3], quaternions_xyzw[:, :3]])

    # Get image list from image directory if provided
    image_list = None
    if image_dir and os.path.exists(image_dir):
        image_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    elif image_names is not None:
        image_list = image_names

    # Save images.txt
    images_txt = ""
    for i in range(N):
        qw, qx, qy, qz = quaternions_wxyz[i]
        x, y, z = positions[i]

        # Use actual image name if available, otherwise generate one
        if image_list is not None and i < len(image_list):
            img_name = image_list[i]
        else:
            img_name = f"image_{i:06d}.jpg"

        images_txt += f"{i+1} {qw} {qx} {qy} {qz} {x} {y} {z} 1 {img_name}\n\n"

    with open(colmap_dir / "images.txt", "w") as f:
        f.write(images_txt)

    # Save empty points3D.txt (required by COLMAP format)
    with open(colmap_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    # Save cameras.txt
    if isinstance(intrinsics, np.ndarray) and intrinsics.ndim > 1:
        fx, fy, cx, cy = intrinsics[0]  # Use first frame intrinsics
    else:
        fx, fy, cx, cy = intrinsics

    # Try to get actual image dimensions from first image if image_dir is provided
    H, W = 480, 640  # default values
    if image_dir and os.path.exists(image_dir) and image_list:
        try:
            import cv2
            first_image_path = os.path.join(image_dir, image_list[0])
            img = cv2.imread(first_image_path)
            if img is not None:
                H, W = img.shape[:2]
        except:
            pass  # Use default values if reading fails

    with open(colmap_dir / "cameras.txt", "w") as f:
        f.write(f"# Camera list with one line of data per camera:\n")
        f.write(f"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    print(f"COLMAP reconstruction saved to: {colmap_dir.resolve()}")
    return str(colmap_dir)

def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--depthdir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")
    parser.add_argument("--start_idx", default=0, type=int, help="starting image index (default: 0)")
    parser.add_argument("--end_idx", default=None, type=int, help="ending image index (default: None, process all)")

    parser.add_argument("--droid_weights", default="checkpoints/droid.pth")
    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--export_colmap", action="store_true", help="export reconstruction in COLMAP format")
    parser.add_argument("--rerun", action="store_true")

    parser.add_argument("--sea_rafts_weights", default=None)
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid_slam = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, depth, intrinsics) in tqdm(image_stream(args.imagedir, args.depthdir, args.calib, args.stride, args.start_idx, args.end_idx)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid_slam is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid_slam = Droid(args)

        droid_slam.track(t, image, depth, intrinsics=intrinsics)

    if args.reconstruction_path is not None:
        save_reconstruction(droid_slam, args.reconstruction_path)

    w2c_traj, traj_est = droid_slam.terminate(image_stream(args.imagedir, args.depthdir, args.calib, args.stride, args.start_idx, args.end_idx))
    print("trajectory number of frames: ", len(traj_est))
    # Export to COLMAP format if requested
    if args.export_colmap:
        from pathlib import Path

        # Get reconstruction data from DROID
        t = droid_slam.video.counter.value
        intrinsics = droid_slam.video.intrinsics[0].cpu().numpy() * 8.0

        # Set up output path
        output_path = Path(args.imagedir).parent / f"droid_colmap"

        save_colmap_from_w2c_traj(
            w2c_traj,
            intrinsics,
            str(output_path),
            image_dir=args.imagedir
        )
