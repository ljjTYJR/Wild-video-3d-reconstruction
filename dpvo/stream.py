import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

def image_stream(queue, imagedir, depthdir, maskdir, calib, stride, skip=0, end=None):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    if end is not None:
        image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip:end:stride]
    else:
        image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    depth_exts = ["*.npy"]
    depth_list = None
    if depthdir:
        if end is not None:
            depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in depth_exts))[skip:end:stride]
        else:
            depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in depth_exts))[skip::stride]

    mask_exts = ["*.png", "*.jpeg", "*.jpg"]
    mask_list = None
    if maskdir:
        if end is not None:
            mask_list = sorted(chain.from_iterable(Path(maskdir).glob(e) for e in mask_exts))[skip:end:stride]
        else:
            mask_list = sorted(chain.from_iterable(Path(maskdir).glob(e) for e in mask_exts))[skip::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile), cv2.IMREAD_COLOR) # BGR
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])

        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        if depth_list:
            depth = np.load(str(depth_list[t]))
            depth = depth[:h-h%16, :w-w%16]
            depth_median = np.median(depth[depth > 0])
            depth[depth > 10 * depth_median] = 10 * depth_median
        else:
            depth = None
        if mask_list:
            mask = cv2.imread(str(mask_list[t]), cv2.IMREAD_GRAYSCALE)
            mask = mask[:h-h%16, :w-w%16]
            mask = mask.astype(bool)
        else:
            mask = None
        queue.put((t, image, depth, mask, intrinsics))
    queue.put((-1, image, depth, mask, intrinsics))

def image_stream_limit(queue, imagedir, calib, stride, skip=0, end_idx=50):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    if len(image_list) < end_idx:
        raise ValueError(f"Number of images in the directory is less than {end_idx}")
    count = 0
    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile), cv2.IMREAD_COLOR) # BGR
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])

        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics))
        count += 1
        if count >= end_idx:
            break


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

