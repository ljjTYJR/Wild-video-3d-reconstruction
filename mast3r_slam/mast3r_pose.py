#####
# Utility functions for pose estimation
#####
from dust3r.utils.device import to_numpy
import numpy as np
import cv2
import torch
import roma

def pnp_pose(pts, pixels, intrinsic, pnp_iters=15, device='cuda'):
    """ Fast PnP to estimate the camera pose when given 3D-2D correspondences"""
    # all data moved to numpy
    pts = to_numpy(pts) # in the global coordinate
    pixels = to_numpy(pixels).astype(np.float32)
    intrinsic = to_numpy(intrinsic)
    K = np.float32([(intrinsic[0], 0, intrinsic[2]), (0, intrinsic[1], intrinsic[3]), (0, 0, 1)])
    success, R, T, inliers = cv2.solvePnPRansac(pts, pixels, K, None,
                                                    iterationsCount=pnp_iters, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP) # inlier: the index of the inliers

    # TODO: refine the pose using inliers
    R = cv2.Rodrigues(R)[0] # world2cam
    R, T = map(torch.from_numpy, (R, T))
    return R, T, inliers

def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf

def matrix_to_7dof_pose(matrix):
    """Convert a 4x4 matrix to 7dof pose: translation + quaternion (xyzw) """
    T = matrix[:3, 3]
    q = roma.rotmat_to_unitquat(matrix[:3, :3])
    return torch.cat((T, q), dim=0)