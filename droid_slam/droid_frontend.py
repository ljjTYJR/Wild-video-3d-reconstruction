import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph

import droid_backends

from droid_slam.rerun_visualizer import get_current_color_points
import torch.nn.functional as F
from droid_utils import droid_transform
import roma
from PIL import Image

class DroidFrontend:
    def __init__(self, net, video, vis, args):
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

        self.opt_iter=300
        self.RES = 8.0

        # visualizer
        self.rr_vis = vis

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0,
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1]) # if true, select disps_sens, otherwise disps # initialize the new frame

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

    def __droid_refine(self):
        """ add edges, perform update """

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0,
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1]) # if true, select disps_sens, otherwise disps # initialize the new frame

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

        ### -------- DEBUG --------
        # debug: visualize the optimized point cloud
        """
        _poses = SE3(self.video.poses[:self.t1]).inv().data
        _depths = self.video.disps[:self.t1].clone()
        points = droid_backends.iproj(_poses, _depths, self.video.intrinsics[0])
        pcd_all = o3d.geometry.PointCloud()
        for i in range(len(points)):
            _points = points[i].cpu().numpy().reshape(-1, 3)
            # clip the depths of _points to 3*median
            _points = _points
            _pcd = o3d.geometry.PointCloud()
            _pcd.points = o3d.utility.Vector3dVector(_points)
            _color = self.video.images[i][[2,1,0], 3::8,3::8].cpu().numpy().transpose(1,2,0) / 255.0
            _color = _color.reshape(-1,3)
            _pcd.colors = o3d.utility.Vector3dVector(_color)
            pcd_all += _pcd
        o3d.visualization.draw_geometries([pcd_all])
        """
        ###

    def ___motion_only_ba_detection(self, motion_only_window = 10):
        """ based on the local map, run the motion-only bundle adjustment to estimate the new keyframe based on the distacne """

        self.count += 1
        self.t1 += 1

        """ construct a new factor graph just for the motion-only bundle adjustment """
        # the new frame is with the index of t1-1 (t1 always sychronize with the video.counter.value)
        local_graph = FactorGraph(self.video, self.update_op)
        # add factors in the graph, use all previous frames or latest M frames.
        # the latest frame index is `t1-1`
        i_start = max(self.t1 - motion_only_window - 1, 0)
        i_end = self.t1 - 1 #arange: [start, end)
        # TODO: is the ii-jj correct?
        local_graph.add_factors(torch.arange(i_start, i_end).cuda(), torch.tensor([self.t1-1]).cuda().repeat(i_end-i_start))

        for _ in range(6):
            # only the frame of t1-1 is optimized, and only motion is considered
            local_graph.update(self.t1-1, self.t1, motion_only=True)

        # set initial pose for next frame
        # TODO: check whether the distance changes or not
        d = self.video.distance([self.t1-2], [self.t1-1], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            #NOTE: remove the t1-1 frame in the video? No need to do so, we just need the counter and to control the video buffer and keyframe index.
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        else:
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

        rr_iter = 0
        for itr in range(8):
            self.graph.update(1, use_inactive=True)
            if self.rr_vis is not None:
                self.rr_vis(True, True, True, True, rr_iter)
                rr_iter += 1

        # 1. visualize the DROID optimized point cloud
        # 1. the DROID optimized point cloud
        """
        droid_depths = (1 / self.video.disps[:self.t1].clone())[None]
        droid_depths = F.interpolate(droid_depths, scale_factor=self.RES, mode='bilinear').squeeze()
        droid_depths = 1 / droid_depths
        ## to the point cloud
        pcd_droid = o3d.geometry.PointCloud()
        points = droid_backends.iproj(SE3(self.video.poses).inv().data, droid_depths, self.video.intrinsics[0] * self.RES)
        for i in range(self.t1):
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(points[i].reshape(-1,3).cpu().numpy())
            # color = self.video.images[i][:, [2,1,0]].cpu().numpy().transpose(1,2,0)
            color = self.video.images[i][[2,1,0]].cpu().numpy().transpose(1,2,0) / 255.0
            pcd_tmp.colors = o3d.utility.Vector3dVector(color.reshape(-1,3))
            pcd_droid += pcd_tmp
        o3d.visualization.draw_geometries([pcd_droid])
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
        elif self.is_initialized and self.t1 < self.video.counter.value: # means new frame comes in
            self.__update()

            if self.rr_vis is not None:
                self.rr_vis(True, True, True, True, None)


