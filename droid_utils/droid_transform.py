import numpy as np

def depth2points(depth, intrinsics, pose=None):
    """ convert depth map to 3D points """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics
    x = np.arange(w) - cx
    y = np.arange(h) - cy
    xx, yy = np.meshgrid(x, y)
    xx = xx * depth / fx
    yy = yy * depth / fy
    points = np.stack([xx, yy, depth], axis=-1)
    if pose is not None:
        points = np.dot(points, pose[:3, :3].T) + pose[:3, 3]
    return points