import numpy as np
import torch
from einops import asnumpy, reduce, repeat

from . import projective_ops as pops
from .lietorch import SE3
# from .loop_closure.optim_utils import reduce_edges
from .utils import *
from .netvlad_retrieval import RetrievalNetVLAD


class PatchGraph:
    """ Dataclass for storing variables """

    def __init__(self, cfg, P, DIM, pmem, **kwargs):
        self.cfg = cfg
        self.P = P
        self.pmem = pmem
        self.DIM = DIM

        self.n = 0      # number of frames
        self.m = 0      # number of patches

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.tstamps_ = np.zeros(self.N, dtype=np.int64)
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
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

        # configuration of the loop closure
        if cfg.LOCAL_LOOP:
            self.local_loop_db = RetrievalNetVLAD(self.N)

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
