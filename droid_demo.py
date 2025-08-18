import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import argparse

from droid_slam.droid import Droid
from torch.multiprocessing import Process
import torch.nn.functional as F
from multiprocessing import Process, Queue

from dpvo.stream import video_stream

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, depthdir, calib, stride):
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

    image_list = sorted(os.listdir(imagedir))[::stride]
    depth_list=None
    if depthdir:
        depth_list = sorted(os.listdir(depthdir))[::stride]

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
            depth = np.load(os.path.join(depthdir, depth_list[t]))
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


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

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

    parser.add_argument("--droid_weights", default="checkpoints/droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
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
    parser.add_argument("--rerun", action="store_true")

    # parser.add_argument("--sea_rafts_weights", default="checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth")
    parser.add_argument("--sea_rafts_weights", default=None)
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid_slam = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    # process the input images
    # queue = Queue(maxsize=8) #image queue
    # if os.path.isdir(args.imagedir):
    #     reader = Process(target=image_stream, args=(queue, args.imagedir, None, args.calib, args.stride))
    # else:
    #     reader = Process(target=video_stream, args=(queue, args.imagedir, args.calib, args.stride, 0))
    # reader.start()

    tstamps = []
    for (t, image, depth, intrinsics) in tqdm(image_stream(args.imagedir, args.depthdir, args.calib, args.stride)):
    # while True:
    #     (t, image, depth, intrinsics) = queue.get()
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        # TODO: type conversion
        # image = torch.as_tensor(image).permute(2, 0, 1)
        # image = image[None]
        # intrinsics = torch.as_tensor(intrinsics).float()

        if droid_slam is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid_slam = Droid(args)

            # droid_slam.track(t, image, depth, intrinsics=intrinsics)
            droid_slam.track(t, image, None, intrinsics=intrinsics)

    # save the result
    if droid_slam is not None:
        """ Todo: save the trajectory result """
        pass

    # if args.reconstruction_path is not None:
    #     save_reconstruction(droid_slam, args.reconstruction_path)

    # traj_est = droid_slam.terminate(image_stream(args.imagedir, args.calib, args.stride))
