import torch
import numpy as np
import torch.nn.functional as F
import cv2
import os
import matplotlib
from matplotlib import pyplot as plt

from . import fastba
from . import altcorr
from . import lietorch
from . import ba
from .lietorch import SE3
from lightglue import SuperPoint
from .net import VONet
from .utils import *
from .patchgraph import PatchGraph
from . import projective_ops as pops
from .dpvo_mast3r_init import dpvo_mast3r_initialization, dpvo_mast3r_optimization

from mast3r.model import AsymmetricMASt3R
import rerun as rr

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

class DPVO:
    def __init__(self, cfg, network, ht=480, wd=640, viz=False, mast3r=False, colmap_init=False, motion_filter=False, path=''):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False

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

        # super-point extractor
        try:
            # self.sp_extractor = SuperPoint(max_num_keypoints=self.cfg.PATCHES_PER_FRAME).eval().cuda()  # load the extractor
            self.sp_extractor = None
        except:
            self.sp_extractor = None
        if self.sp_extractor is not None:
            self.M += self.cfg.PATCHES_PER_FRAME # random sampling + super point extraction

        ### network attributes ###
        self.pmem = self.mem = 36 # 32 was too small given default settings

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.image_buffer_ = torch.zeros(self.mem, 3, self.ht, self.wd, dtype=torch.uint8, device="cuda")

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, self.M, ht, wd, RES, **kwargs)
        self.warm_up = 10

        self.viewer = None
        if viz:
            self.start_viewer()

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

        self.motion_filter=motion_filter
        self.path = path

        self.inlier_ratio_record = {}
        self.inlier_ratio_threshold = 0.8

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def patches_est(self):
        return self.pg.patches_est_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

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
        poses = SE3(self.pg.poses_[:self.n])
        poses = poses.inv().data.cpu().numpy()
        translations = poses[:, :3] * scale
        rotations = poses[:, 3:]
        rr.log(path_label, rr.LineStrips3D([translations], colors=[[255, 0, 0]]))

        # log the images
        image = self.image_.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        rr.log(image_label, rr.Image(image))
        rr.log(f"world/camera/{self.n}", rr.Pinhole(focal_length=float(self.intrinsics[0][0][0].cpu()), height=self.ht/self.RES, width=self.wd/self.RES))
        rr.log(f"world/camera/{self.n}", rr.Transform3D(translation=translations[-1], rotation=rr.Quaternion(xyzw=rotations[-1]), scale=0.50))

    def save_inlier_ratio_record(self, path):
        # save the inlier ratio record to the path, `inlier_ratio_record` is a dictionary: {frame_id: inlier_ratio}
        inlier_ratio_record = self.inlier_ratio_record
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # process the last OPTIMIZATION_WINDOW frames
        for i in range(self.n-self.cfg.OPTIMIZATION_WINDOW+2, self.n+1):
            ref_frame, inlier_ratio = self.geo_consistency_check(i, i-1)
            inlier_ratio_record[self.pg.tstamps_[ref_frame].item()] = inlier_ratio.item()
        with open(f"{path}/inlier_ratio_record.txt", "w") as f:
            for key in inlier_ratio_record:
                f.write(f"{key} {inlier_ratio_record[key]}\n")
        with open(f"{path}/time_stamp.txt", "w") as f:
            for i in range(self.n):
                f.write(f"{self.pg.tstamps_[i].item()}\n")
        # draw the figure of inlier ratio respect to the frame id
        import matplotlib.pyplot as plt
        x_ = np.array(list(inlier_ratio_record.keys()))
        y_ = np.array(list(inlier_ratio_record.values()))
        plt.plot(x_, y_, label="inlier ratio")
        # plt.xticks(x_)
        plt.xlabel("frame timestamp")
        plt.ylabel("inlier ratio")
        plt.title("Inlier ratio respect to the frame id")
        plt.savefig(f"{path}/inlier_ratio_record.png")
        plt.close()

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
            self.pg.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    def get_pts_clr_intri(self, inlier=False):
        # self.pg.m is the number of patches; self.pg.m = self.N * self.M
        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.pg.m], self.intrinsics, self.ix[:self.pg.m])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3).cpu().numpy()
        colors = (self.pg.colors_.view(-1, 3)[:self.pg.m].cpu().numpy() if self.pg.colors_.is_cuda else self.pg.colors_.view(-1, 3)[:self.pg.m]) * 255.0

        patches = self.pg.patches_[:self.n][..., self.P // 2, self.P // 2]
        med_by_frame = patches[:, :, 2].median(dim=1).values
        mask_far = (patches[:, :, -1] > 1.0 * med_by_frame[:, None]).view(-1).cpu().numpy()
        mask_near = (patches[:, :, -1] < 4.0 * med_by_frame[:, None]).view(-1).cpu().numpy()
        mask = mask_far & mask_near

        intrinsic = self.pg.intrinsics_[0].cpu().numpy() * self.RES
        H, W = self.ht, self.wd

        # save the inlier ratio record
        if inlier:
            self.save_inlier_ratio_record(self.path)
        return points[mask], colors[mask], (intrinsic, H, W)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i].item()] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    def terminate_keyframe(self):
        """ Only report keyframes """
        self.traj = {}
        key_frame_timestamps = []
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i].item()] = self.pg.poses_[i]
            key_frame_timestamps.append(self.pg.tstamps_[i].item())
        poses=[]
        for i in range(self.n):
            poses.append(SE3(self.pg.poses_[i]))
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(key_frame_timestamps, dtype=float)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3) # now, the coords will de downsacled by 4 (not integer anymore)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, kk, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj]) # target frame index
        self.pg.kk = torch.cat([self.pg.kk, kk]) # patch index
        self.pg.ii = torch.cat([self.pg.ii, self.ix[kk]]) # source frame index

        net = torch.zeros(1, len(kk), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m):
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.pg.m-self.M, self.pg.m, device="cuda")
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
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def draw_img_matching_coord(self, key_idx, query_num):
        coords = self.reproject() # [B, N, 2, p, p]
        key_img = cv2.cvtColor(self.image_buffer_[key_idx % self.mem].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
        if query_num <= 3:
            fig, axes = plt.subplots(query_num, 2, figsize=(15, 20))
        else:
            row = 3
            col = 2
            fig, axes = plt.subplots(row, col, figsize=(15, 20))
        key_img_x = self.pg.patches_[key_idx][:, 0, self.P//2, self.P//2].cpu().numpy() * self.RES
        key_img_y = self.pg.patches_[key_idx][:, 1, self.P//2, self.P//2].cpu().numpy() * self.RES
        for ax_idx, ax in enumerate(axes.flat):
            i = ax_idx + 1
            tgt_idx = key_idx - i
            tgt_idx_mem = (key_idx - i) if (key_idx - i >= 0) else (key_idx - i + self.mem)
            tgt_img = cv2.cvtColor(self.image_buffer_[tgt_idx_mem % self.mem].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
            indices = ((self.pg.ii == key_idx) & (self.pg.jj == tgt_idx)).nonzero().squeeze()
            tgt_img_x = coords[0, indices, 0, self.P//2, self.P//2].cpu().numpy() * self.RES
            tgt_img_y = coords[0, indices, 1, self.P//2, self.P//2].cpu().numpy() * self.RES
            concat_img = np.concatenate((key_img, tgt_img), axis=1)
            adjusted_tgt_x = tgt_img_x + key_img.shape[1]
            ax.imshow(concat_img)
            ax.scatter(key_img_x, key_img_y, c='red', s=15, edgecolor='black', label='Keyframe Pixels')
            ax.scatter(adjusted_tgt_x, tgt_img_y, c='blue', s=15, edgecolor='black', label='Target Frame Pixels')
            for j in range(len(key_img_x)):
                ax.plot([key_img_x[j], adjusted_tgt_x[j]], [key_img_y[j], tgt_img_y[j]], color='green', linewidth=0.5)
            ax.set_title(f'Image Pair {i}')
            ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

    def draw_img_matching_target(self, key_idx, query_num, save=False):
        if save:
            # non-interactive mode
            matplotlib.use('Agg')
        coords = self.reproject() # [B, N, 2, p, p]
        with torch.amp.autocast('cuda', enabled=True):
            corr = self.corr(coords) # correleation
            ctx = self.imap[:,self.pg.kk % (self.M * self.mem)]
            _, (delta, weight, _) = \
                self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)
        target = coords[...,self.P//2,self.P//2] + delta.float()
        key_img = cv2.cvtColor(self.image_buffer_[key_idx % self.mem].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
        if query_num <= 3:
            fig, axes = plt.subplots(query_num, 2, figsize=(15, 20))
        else:
            row = query_num
            col = 1
            fig, axes = plt.subplots(row, col, figsize=(15, 20))
        key_img_x = self.pg.patches_[key_idx][:, 0, self.P//2, self.P//2].cpu().numpy() * self.RES
        key_img_y = self.pg.patches_[key_idx][:, 1, self.P//2, self.P//2].cpu().numpy() * self.RES
        for ax_idx, ax in enumerate(axes.flat):
            i = ax_idx + 1
            tgt_idx = key_idx - i
            tgt_idx_mem = (key_idx - i) if (key_idx - i >= 0) else (key_idx - i + self.mem)
            tgt_img = cv2.cvtColor(self.image_buffer_[tgt_idx_mem % self.mem].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
            indices = ((self.pg.ii == key_idx) & (self.pg.jj == tgt_idx)).nonzero().squeeze()
            tgt_img_x = target[0, indices, 0].cpu().numpy() * self.RES
            tgt_img_y = target[0, indices, 1].cpu().numpy() * self.RES
            tgt_weight_x = weight[0, indices, 0].cpu().numpy()
            tgt_weight_y = weight[0, indices, 1].cpu().numpy()
            concat_img = np.concatenate((key_img, tgt_img), axis=1)
            adjusted_tgt_x = tgt_img_x + key_img.shape[1]
            ax.imshow(concat_img)
            ax.scatter(key_img_x, key_img_y, c='red', s=15, edgecolor='black', label='Keyframe Pixels')

            tgt_weights = np.stack([tgt_weight_x, tgt_weight_y], axis=1)
            tgt_weights_norm = np.linalg.norm(tgt_weights, axis=1)
            scale = 1 / np.clip(tgt_weights_norm, 0.02, 1.0)

            # ax.scatter(adjusted_tgt_x, tgt_img_y, c='blue', s=15, edgecolor='black', label='Target Frame Pixels')
            ax.scatter(adjusted_tgt_x, tgt_img_y, c='blue', s=scale, edgecolor='black', label='Target Frame Pixels')
            for j in range(len(key_img_x)):
                ax.plot([key_img_x[j], adjusted_tgt_x[j]], [key_img_y[j], tgt_img_y[j]], color='green', linewidth=0.5)
            ax.set_title(f'Image Pair {i}')
            ax.axis('off')
        plt.tight_layout()
        if save:
            imgmatch_path = f'{self.path}/img_match'
            if not os.path.exists(imgmatch_path):
                os.makedirs(imgmatch_path, exist_ok=True)
            plt.savefig(f'{imgmatch_path}/img_match_{key_idx}.png', bbox_inches='tight', dpi=300)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()

    def keyframe(self, mast3r_update=True):
        cur_key = self.cfg.KEYFRAME_INDEX
        i = self.n - cur_key - 1
        j = self.n - cur_key + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        k = self.n - cur_key
        if m / 2 < self.cfg.KEYFRAME_THRESH and self.motion_filter:
            print(f"drop frame {self.pg.tstamps_[self.n - cur_key].item()} due to low motion")
            t0 = self.pg.tstamps_[k-1].item()
            t1 = self.pg.tstamps_[k].item()

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.patches_est_[i] = self.pg.patches_est_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

                self.image_buffer_[i%self.mem] = self.image_buffer_[(i+1) % self.mem]

            self.n -= 1
            self.pg.m -= self.M
        else:
            if self.cfg.LOCAL_LOOP:
                self.pg.local_loop_db.insert_img(k, self.image_buffer_[k % self.mem][[2, 1, 0], :, :]/255.0)
                query_val, quer_indices = self.pg.local_loop_db.query(k)

            if mast3r_update and self.mast3r_est:
                # Use the mast3r model to give the coarse camera poses estimation (only 3D-3D correspondence)
                if k > self.cfg.PATCH_LIFETIME:
                    # make use of prior poses as seeds to do the initializatioin
                    img_idx0 = k-self.cfg.PATCH_LIFETIME
                    img_idx1 = k+1
                    # img_idx1 = k-7
                    images = [self.image_buffer_[i % self.mem] for i in range(img_idx0, img_idx1)]
                    poses = [SE3(self.pg.poses_[i]) for i in range(img_idx0, img_idx1)]
                    dpvo_mast3r_optimization(images, poses, pts=None, mast3r_model=self.mast3r_model, fixed=2, intrinsics=self.pg.intrinsics_[img_idx0:img_idx1])

            # self.draw_img_matching_target(k, 6, save=True)
            if torch.isnan(self.pg.poses_[k]).any():
                print("Error: the estimated pose is nan!")
                raise Exception("Error: the estimated pose is nan!")

        """
        if self.n > self.cfg.OPTIMIZATION_WINDOW:
            ref_frame, inlier_ratio = self.geo_consistency_check(self.n-self.cfg.OPTIMIZATION_WINDOW+1, self.n-self.cfg.OPTIMIZATION_WINDOW)
            self.inlier_ratio_record[self.pg.tstamps_[ref_frame].item()] = inlier_ratio.item()
        """
        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update_get_corr(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject() # [B, N, 2, p, p] which indicates the corresponding tracks, it can be implemented by the pycolmap with mast3r initialization

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:,self.pg.kk % (self.M * self.mem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float() # so each patch will be treated as a whole, when P is 3, it is the center of the patch

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics,
                    target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.pg.m], self.intrinsics, self.ix[:self.pg.m]) # note that it will return all the points so far
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
        return points, target # the `points` are the 3D points in the scene; while the target is the re-projection of points

    def geo_consistency_check(self, query_frame, fixed_frame):
        coords = self.reproject()
        with torch.amp.autocast('cuda', enabled=True):
                corr = self.corr(coords) # correleation
                ctx = self.imap[:,self.pg.kk % (self.M * self.mem)]
                _, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

        weight = weight.float()
        target = coords[...,self.P//2,self.P//2] + delta.float() # so each patch will be treated as a whole, when P is 3, it is the center of the patch
        src_window_mask = self.pg.ii == query_frame # it is the fixed frame for
        tgt_window_mask = self.pg.jj <= fixed_frame
        mask = src_window_mask & tgt_window_mask # the final patch mask, starting from optimizable frames to fixed frames
        coords = coords.squeeze()[mask][:,:,1,1]
        target = target.squeeze()[mask]
        r = (coords-target).norm(dim=-1)
        cx = self.intrinsics[0][0][2]; cy = self.intrinsics[0][0][3]
        in_bounds = (coords[:,0] > -cx) & (coords[:,1] < 3 * cx) & (coords[:,1] > -cy) & (coords[:,1] < 3 * cy)
        low_ropj_error = r < 4.0
        inlier_ratio = ((low_ropj_error & in_bounds).sum().float() / mask.sum().float()).cpu().numpy()
        return query_frame, inlier_ratio

    def update(self, t0=None):

        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject() # [B, N, 2, p, p] which indicates the corresponding tracks, it can be implemented by the pycolmap with mast3r initialization

            with torch.amp.autocast('cuda', enabled=True):
                corr = self.corr(coords) # correleation
                ctx = self.imap[:,self.pg.kk % (self.M * self.mem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float() # so each patch will be treated as a whole, when P is 3, it is the center of the patch

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            t0_ = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0_, t0 or 1)

            # try:
                # fastba.BA(self.poses, self.patches, self.intrinsics,
                #     target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, 2)

            # if using Python-version BA
            try:
                bounds = [0-10, 0-10, self.wd+10, self.ht+10]
                lmbda=1e-4
                Gs, patches = ba.BA(SE3(self.poses), self.patches, self.intrinsics,
                    target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, bounds, fixedp=t0, patches_est=self.patches_est)
                self.pg.patches_[:] = patches.reshape(self.N, self.M, 3, self.P, self.P)
                self.pg.poses_[:] = Gs.vec().reshape(self.N, 7)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.pg.m], self.intrinsics, self.ix[:self.pg.m]) # note that it will return all the points so far
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.pg.m, device="cuda"),
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

    def __call__(self, tstamp, image, depth, mask, intrinsics):
        """ track new frame """

        if (self.pg.n+1) >= self.pg.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)
        self.image_ = image # cv2 read-in image

        with torch.amp.autocast('cuda', enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    gradient_bias=self.cfg.GRADIENT_BIAS,
                    return_color=True,
                    mask=mask,
                    sp_extractor=self.sp_extractor)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.pg.n + 1
        self.pg.index_map_[self.n + 1] = self.pg.m + self.pg.M

        if self.pg.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)
                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()

                # xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()

                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.pg.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # use the metric3D as initialization in the optimization
        # TODO: also, as the initialization!
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None]) # the patch dimension: [B, N, 3, p, p], the 3rd at 3 is the depth; 1st at 3 is W, 2rd at 3 in height
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches
        self.pg.set_prior_depth(self.n, depth)

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)
        self.image_buffer_[self.n % self.mem] = image

        self.counter += 1
        # use enough initial motions for initialization
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return
            else:
                if self.cfg.LOCAL_LOOP:
                    self.pg.local_loop_db.insert_img(self.n, image[[2, 1, 0], :, :]/255.0) # (RGB; (3,h,w), [0,1])

        self.pg.n += 1
        self.pg.m += self.M

        # relative pose
        self.append_factors(*self.__edges_forw()) # connect previous patches to the current new frame
        self.append_factors(*self.__edges_back())

        if self.n == self.warm_up and not self.is_initialized:
            self.is_initialized = True
            """
            elif self.n == self.warm_up:
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
                    self.pg.intrinsics_[:self.warm_up] = mast3r_intrinsics

                    # depths = F.interpolate(depths, scale_factor=0.25, mode='bilinear').squeeze()
                    # # initialize the path depth with the mast3r depth map
                    # for idx in range(len(depths)):
                    #     patch = self.pg.patches_[idx].clone()
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
                    #     self.pg.patches_[idx] = patch

                del self.mast3r_image_buffer
                del self.mast3r_model
                """
            if self.mast3r_est:
                mast3r_depths, mast3r_poses = dpvo_mast3r_initialization(self.image_buffer_[:self.warm_up], self.mast3r_model, intrinsics=self.pg.intrinsics_[:self.warm_up])
                self.pg.init_from_prior(mast3r_depths, mast3r_poses, list(range(self.warm_up)), images=self.image_buffer_[:self.warm_up])
            # self.draw_img_matching_coord(6, 6)

            for itr in range(12):
                if self.rr:
                    self.rr_register_info(itr)
                # self.draw_img_matching_target(6, 6)
                if self.mast3r_est:
                    self.update(t0=4) # we fix first two frames to get the scale
                else:
                    self.update()

        elif self.is_initialized:
            if self.mast3r_est:
                self.update(t0=4)
            else:
                self.update()
            self.keyframe(mast3r_update=False)
            if self.rr:
                self.rr_register_info()

# NOTE: current not used
# def prepare_colmap_data(dpvo, idx0, idx1):
#     """ Prepare the coarse DPVO-BA result data for the COLMAP refinement """
#     extrinsics = SE3(dpvo.poses_[idx0:idx1]).inv().matrix().cpu().numpy() # camera to world

#     _intrinsic = dpvo.intrinsics_[0].cpu().numpy() * dpvo.RES
#     _intrinsic = np.array([[_intrinsic[0], 0, _intrinsic[2]], [0, _intrinsic[1], _intrinsic[3]], [0, 0, 1]])
#     intrinsic = np.tile(_intrinsic[None], (idx1 - idx0, 1, 1))