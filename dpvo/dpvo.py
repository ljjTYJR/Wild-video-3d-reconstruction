import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .net import VONet
from .utils import *
from . import projective_ops as pops
import cv2

import rerun as rr

from dust3r.inference import load_model
from dust3r.dust3r_type import set_as_dust3r_image, dust3r_inference
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import local_dust3r_ba

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

class DPVO:
    def __init__(self, cfg, network, ht=480, wd=640, viz=False, mast3r=False, all_frames=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False

        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda") # world to camera
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.viewer = None
        if viz:
            self.start_viewer()

        self.warm_up = 10

        # re-run visualization
        self.rr = True
        if self.rr:
            rr.init('DPVO Visualization')
            rr.connect()
            rr.set_time_sequence("#frame", 0)

        # Add the Dust3R model for inference
        self.mast3r_est=True if mast3r else False
        self.mast3r_model_path="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if self.mast3r_est:
            self.mast3r_model=AsymmetricMASt3R.from_pretrained(self.mast3r_model_path).to('cuda').eval()
        else:
            self.mast3r_model=None
        self.mast3r_image_buffer=[] # a mast3r frame buffer for mast3r inference

        self.all_frames=all_frames

    def rr_register_info(self,
        frame_n=None,
        point_label='world/points',
        path_label='world/path',
        camera_label='world/cameras',
        image_label='world/image'):
        if frame_n is not None:
            rr.set_time_sequence("#frame", frame_n)
        else:
            rr.set_time_sequence("#frame", self.n)

        scale = 100.0
        points, colors, _ = self.get_pts_clr_intri()
        points = points * scale # for visualization
        rr.log(point_label, rr.Points3D(points, colors=colors))

        # register camera poses and visualize
        poses = SE3(self.poses_[:self.n])
        poses = poses.inv().data.cpu().numpy()
        translations = poses[:, :3] * scale
        rotations = poses[:, 3:]
        # rr.log(path_label, rr.LineStrips3D([translations], colors=[[255, 0, 0]]))

        # log the images
        image = self.image_.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        rr.log(image_label, rr.Image(image))
        rr.log(f"world/camera/{self.n}", rr.Pinhole(focal_length=float(self.intrinsics[0][0][0].cpu()), height=self.ht/self.RES, width=self.wd/self.RES))
        rr.log(f"world/camera/{self.n}", rr.Transform3D(translation=translations[-1], rotation=rr.Quaternion(xyzw=rotations[-1]), scale=0.0050))

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

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

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pts_clr_intri(self):
        # self.m is the number of patches; self.m = self.N * self.M
        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3).cpu().numpy()
        colors = (self.colors_.view(-1, 3)[:self.m].cpu().numpy() if self.colors_.is_cuda else self.colors_.view(-1, 3)[:self.m]) * 255.0

        patches = self.patches_[:self.n][..., self.P // 2, self.P // 2]
        med_by_frame = patches[:, :, 2].median(dim=1).values
        mask = (patches[:, :, -1] > 0.5 * med_by_frame[:, None]).view(-1).cpu().numpy()

        intrinsic = self.intrinsics_[0].cpu().numpy() * self.RES
        H, W = self.ht, self.wd
        return points[mask], colors[mask], (intrinsic, H, W)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3) # now, the coords will de downsacled by 4 (not integer anymore)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, kk, jj):
        self.jj = torch.cat([self.jj, jj]) # target frame index
        self.kk = torch.cat([self.kk, kk]) # patch index
        self.ii = torch.cat([self.ii, self.ix[kk]]) # source frame index

        net = torch.zeros(1, len(kk), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update_get_corr(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject() # [B, N, 2, p, p] which indicates the corresponding tracks, it can be implemented by the pycolmap with mast3r initialization

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float() # so each patch will be treated as a whole, when P is 3, it is the center of the patch

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics,
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m]) # note that it will return all the points so far
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
        return points, target # the `points` are the 3D points in the scene; while the target is the re-projection of points

    def update(self):
        # visualize the points first
        # FIRST_FRAME_PATCHES = 96
        # points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.M], self.intrinsics, self.ix[:self.M]) # note that it will return all the points so far
        # points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        # # register the points in the rerun
        # points = points.cpu().numpy() if points.is_cuda else points
        # rr.log('world/points', rr.Points3D(points))

        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject() # [B, N, 2, p, p] which indicates the corresponding tracks, it can be implemented by the pycolmap with mast3r initialization

            with torch.amp.autocast('cuda', enabled=True):
                corr = self.corr(coords) # correleation
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float() # so each patch will be treated as a whole, when P is 3, it is the center of the patch

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1) # means, t0 at least should be larger than 1, to keep one fixed

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics,
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m]) # note that it will return all the points so far
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]

    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)
        self.image_ = image
        image = 2 * (image[None,None] / 255.0) - 0.5

        with torch.amp.autocast('cuda', enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    gradient_bias=self.cfg.GRADIENT_BIAS,
                    return_color=True)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        if self.mast3r_est:
            self.intrinsics_[self.n] = self.intrinsics_[0].clone() # if use the Mast3R estimation, we use the first one, which is enough
        else:
            self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])

                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        # TODO use the metric3D as initialization (do we need intrinsic parameters?)
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None]) # the patch dimension: [B, N, 3, p, p], the 3rd at 3 is the depth; 1st at 3 is W, 2rd at 3 in height
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1
        # DEBUG: use all frames during the SLAM
        # if self.n > 0 and not self.is_initialized:
            # if self.motion_probe() < 2.0:
                # self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                # return

        self.n += 1
        self.m += self.M

        # relative pose
        self.append_factors(*self.__edges_forw()) # connect previous patches to the current new frame
        self.append_factors(*self.__edges_back())

        if not self.is_initialized and self.mast3r_est:
            self.mast3r_image_buffer.append(self.image_.clone())

        # if self.n == 8 and not self.is_initialized:
        if self.n == self.warm_up and not self.is_initialized:
            self.is_initialized = True

            # optimize and get the estimated intrinsic parameters
            if len(self.mast3r_image_buffer) == self.warm_up:
                with torch.enable_grad():
                    scene = local_dust3r_ba(self.mast3r_image_buffer, self.mast3r_model)
            # focal, cx and cy are half of the original image size
            if self.mast3r_est:
                avg_intrinsics = scene.get_intrinsics().mean(dim=0)
                depths = torch.stack(scene.get_depthmaps(raw=False), dim=0)
                mast3r_intrinsics = torch.tensor([avg_intrinsics[0, 0], avg_intrinsics[0, 0],
                                                  avg_intrinsics[0, 2], avg_intrinsics[1, 2]], device='cuda') \
                                                / self.RES
                self.intrinsics_[:self.warm_up] = mast3r_intrinsics

                # depths = F.interpolate(depths, scale_factor=0.25, mode='bilinear').squeeze()
                # # initialize the path depth with the mast3r depth map
                # for idx in range(len(depths)):
                #     patch = self.patches_[idx].clone()
                #     depth = depths[idx]
                #     N, _, W_patch, H_patch = patch.shape
                #     for i in range(N):
                #         x_coords = patch[i, 0, :, :].long()
                #         y_coords = patch[i, 1, :, :].long()
                #         x_coords = torch.clamp(x_coords, 0, depth.shape[1] - 1)
                #         y_coords = torch.clamp(y_coords, 0, depth.shape[0] - 1)
                #         extracted_depths = depth[y_coords, x_coords]
                #         median_depth = torch.median(extracted_depths)
                #         patch[i, 2, :, :] = 1/median_depth # NOTE: we use the inverse depth
                #     self.patches_[idx] = patch
            for itr in range(12):
                if self.rr:
                    self.rr_register_info(itr)
                self.update()

            del self.mast3r_image_buffer
            del self.mast3r_model

        elif self.is_initialized:
            self.update()
            if not self.all_frames:
                self.keyframe()
            if self.rr:
                self.rr_register_info()

# NOTE: current not used
# def prepare_colmap_data(dpvo, idx0, idx1):
#     """ Prepare the coarse DPVO-BA result data for the COLMAP refinement """
#     extrinsics = SE3(dpvo.poses_[idx0:idx1]).inv().matrix().cpu().numpy() # camera to world

#     _intrinsic = dpvo.intrinsics_[0].cpu().numpy() * dpvo.RES
#     _intrinsic = np.array([[_intrinsic[0], 0, _intrinsic[2]], [0, _intrinsic[1], _intrinsic[3]], [0, 0, 1]])
#     intrinsic = np.tile(_intrinsic[None], (idx1 - idx0, 1, 1))