import numpy as np
import torch
import torch.nn.functional as F
from einops import asnumpy, reduce, repeat

from . import projective_ops as pops
from .lietorch import SE3
# from .loop_closure.optim_utils import reduce_edges
from .utils import *
from .utils import matrix_to_quaternion
import droid_backends

class PatchGraph:
    """ Dataclass for storing variables """

    def __init__(self, cfg, P, DIM, pmem, M, ht_resized, wd_resized, RES, **kwargs):
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        self.n = 0      # number of frames
        self.m = 0      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.M = M
        self.N = self.cfg.BUFFER_SIZE

        self.tstamps_ = np.zeros(self.N, dtype=np.int64)
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda") # NOTE: world2camera
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda") # inverse depths
        self.patches_est_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda") # inverse depths
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        ### edge information ###
        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        ### inactive edge information (i.e., no longer updated, but useful for BA) ###
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk_inac = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.weight_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")
        self.target_inac = torch.zeros(1, 0, 2, dtype=torch.long, device="cuda")

        self.ht_resized = ht_resized
        self.wd_resized = wd_resized
        self.RES=RES

    def edges_loop(self):
        """ Adding edges from old patches to new frames """
        return

    def normalize(self):
        """ normalize depth and poses """
        s = self.patches_[:self.n,:,2].mean()
        self.patches_[:self.n,:,2] /= s
        self.poses_[:self.n,:3] *= s
        for t, (t0, dP) in self.delta.items():
            self.delta[t] = (t0, dP.scale(s))
        self.poses_[:self.n] = (SE3(self.poses_[:self.n]) * SE3(self.poses_[[0]]).inv()).data

        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    def set_prior_depth(self, idx, depth):
        if depth is None:
            return
        patch = self.patches_[idx]
        # TODO: the sampling grids is spare, not the continuous grids in the original image
        x_coords = torch.clamp(patch[:, 0, :, :].long() * self.RES, 0, depth.shape[1] - 1)
        y_coords = torch.clamp(patch[:, 1, :, :].long() * self.RES, 0, depth.shape[0] - 1)

        extracted_depths = depth[y_coords, x_coords]
        median_depths = torch.median(extracted_depths.view(extracted_depths.shape[0], -1), dim=1).values

        patch[:, 2, :, :] = 1 / median_depths.view(-1, 1, 1)
        self.patches_est_[idx] = patch
        self.patches_[idx] = patch # initialize the depth

    def init_from_prior(self, depths, poses, indices, images=None):
        """ Init the depth and camera poses given known prior information (by indices)
        @depths: (N, H, W) (full resolution, real depths)
        @poses: (N, 4, 4) world camera poses
        @indices: list of indices to be initialized
        """
        # depths = torch.stack(depths, dim=0).unsqueeze(0)
        depths = torch.stack(depths, dim=0)
        N, H, W = depths.shape
        dpvo_poses = create_se3_from_mat(poses).inv() # camera2world -> world2camera

        for idx in indices:
            patch = self.patches_[idx] # get indices of the patch
            depth = depths[idx]

            # Extract coordinates and clamp in a batch
            x_coords = torch.clamp(patch[:, 0, :, :].long() * self.RES, 0, depth.shape[1] - 1)
            y_coords = torch.clamp(patch[:, 1, :, :].long() * self.RES, 0, depth.shape[0] - 1)

            # Batch gather depths using advanced indexing
            extracted_depths = depth[y_coords, x_coords]
            median_depths = torch.median(extracted_depths.view(extracted_depths.shape[0], -1), dim=1).values

            # Update the patch in one step
            patch[:, 2, :, :] = 1 / median_depths.view(-1, 1, 1)

            # Save the updated patch;
            self.patches_est_[idx] = patch
            self.poses_[idx] = dpvo_poses[idx].data

def create_se3_from_mat(mats):
    """ Create SE3 from 4x4 matrix """
    Rs = mats[:, :3, :3]
    Ts = mats[:, :3, 3]
    quats = matrix_to_quaternion(Rs)[:, [1,2,3,0]] # (w, x,y,z)->(x,y,z,w)
    poses = torch.cat([Ts, quats], dim=1)
    return SE3(poses)

