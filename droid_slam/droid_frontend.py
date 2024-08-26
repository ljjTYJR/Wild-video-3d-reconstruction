import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph

import rerun as rr
import droid_backends

from dust3r.dust3r_type import set_as_dust3r_image, dust3r_inference
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import mast3r_inference
import torch.nn.functional as F
class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48, upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        # Mast3R related settings
        self.mast3r_pred = args.mast3r_pred
        self.mast3r_batch_size=1
        self.mast3r_schedule='cosine'
        self.mast3r_lr=0.01
        self.opt_iter=300
        self.mast3r_model_path="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if self.mast3r_pred:
            self.mast3r_model=AsymmetricMASt3R.from_pretrained(self.mast3r_model_path).to('cuda').eval()
        else:
            self.mast3r_model=None
        self.mast3r_image_buffer=[] # a mast3r frame buffer for mast3r inference
        self.RES = 8.0

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0,
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        ### use the Mast3R for prediction
        if self.mast3r_pred:
            # push in images
            for i in range(self.t1):
                mast3r_image = set_as_dust3r_image(self.video.images[i].clone(), i)
                self.mast3r_image_buffer.append(mast3r_image)
            # inference
            with torch.no_grad():
                # NOTE: the current inference is just aligned with the first frame using 3D-3D correspondence
                # TODO: use the Dust3R / 2D-2D correspondence to refine the result
                scene = mast3r_inference(self.mast3r_image_buffer, self.mast3r_model, 'cuda')
            # prepare the intrinsic parameters
            focals = scene['focals']; cx = self.video.wd / 2; cy = self.video.ht / 2
            avg_focal = focals.mean().item()
            self.video.intrinsics[:self.t1] = torch.tensor([avg_focal, avg_focal, cx, cy]).cuda() / self.RES
            # prepare the predicted depths
            depths = scene['depths'][None]
            depths = F.interpolate(depths, scale_factor=1/self.RES, mode='bilinear').squeeze()
            # feed the depths into the video frames
            self.video.disps_sens[:self.t1] = torch.where(depths > 0, 1.0/depths, depths)

        for itr in range(10):
            self.graph.update(1, use_inactive=True)
            """ code snippet for visualizing points in the rerun
            # dirty_index = torch.range(0, self.t1-1).cuda().long()
            # poses = torch.index_select(self.video.poses, 0, dirty_index)
            # disps = torch.index_select(self.video.disps, 0, dirty_index)
            # points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu()
            # points = points.view(-1, 3)
            # rr.set_time_sequence("#frame", itr)
            # rr.log("initial points", rr.Points3D(points))
            """


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone() # initialization of the new frame
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()


