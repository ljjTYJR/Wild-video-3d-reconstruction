import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph

import droid_backends

from droid_slam.rerun_visualizer import get_current_color_points
from dust3r.dust3r_type import set_as_dust3r_image, dust3r_inference
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import mast3r_inference_init, mast3r_inference
import torch.nn.functional as F
from droid_utils import droid_transform
import roma
from PIL import Image

import open3d as o3d
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

        self.use_gt_calib = True if args.calib is not None else False

        # Mast3R related settings
        self.mast3r_pred = args.mast3r_pred
        self.mast3r_init_only = args.mast3r_init_only
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
        self.mast3r_window = 8 # TODO: hyper-parameters
        # self.mast3r_window_pose = torch.zeros(self.mast3r_window, 4, 4).cuda()
        self.mast3r_window_depths = torch.zeros(self.mast3r_window, self.video.ht, self.video.wd).cuda()
        self.RES = 8.0
        self.mast3r_keyframe = 0

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

    def __mast3r_update0(self):
        """ This the the original mast3r update optimization, use the mast3r as the initialization, not used now. """
        # first, is there new keyframe coming in? # TODO: now, the keyframe control is based on the DROID strategy
        if self.mast3r_keyframe == self.t1: # No new keyframe
            return
        elif self.mast3r_keyframe < self.t1: # new keyframe
            assert self.mast3r_keyframe == self.t1 - 1
            # organize the Mast3R window and conducting the local alignment
            # what do we need? --> the image (for mast3r model inference); the camera poses (from the DROID); the fixed initial point cloud(for scale alignment)
            # we choose from the latest mast3r_window frames from droid.video and run dust3r-like global alignment
            mast3r_t0 = max(self.t1 - self.mast3r_window, 0)
            mast3r_t1 = self.t1 # it means there are totally t1 frames, the index is t1-1
            # prepare the Mast3R window, for images, the index always start from 0.
            mast3r_images = [set_as_dust3r_image(self.video.images[i].clone(), i-mast3r_t0) for i in range(mast3r_t0, mast3r_t1)]
            mast3r_cam_poses = self.video.poses[mast3r_t0:mast3r_t1].clone()
            mast3r_inv_depths = self.video.disps[mast3r_t0:mast3r_t1].clone()
            mast3r_intrinsic = self.video.intrinsics[0].clone()
            # do the global alignment to solve the last frame prediction only!
            scene = mast3r_inference(mast3r_images, mast3r_cam_poses, mast3r_inv_depths, self.RES, mast3r_intrinsic, self.mast3r_model, 'cuda')
            opt_depth = scene.get_opt_depth_droid # the returned depth is the real depth instead of inverse, for droid, there should be inverse
            opt_pose = scene.get_opt_pose_droid
            # sychronize the depth and pose to the DROID
            opt_depth = F.interpolate(opt_depth[None][None], scale_factor=1/self.RES, mode='bilinear').squeeze()
            # TODO: Do we need to initialize the disps? We only need to feed the disps_sens.
            # self.video.disps[self.t1 - 1] = 1.0 / opt_depth
            self.video.disps_sens[self.t1 - 1] = 1.0 / opt_depth
            # NOTE: the droid pose is world2camera, while the mast3r pose is camera2world
            droid_opt_pose = SE3(opt_pose).inv().data
            self.video.poses[self.t1 - 1] = droid_opt_pose

            ### DEBUG: visualize the optimized point cloud from mast3r prediction
            """
            fixed_pts3d = scene.get_fixed_pts3d()[:scene.known_cams] # BxNx3
            opt_pts3d = scene.get_opt_pts3d() # 1xNx3
            pts3d = torch.cat((fixed_pts3d, opt_pts3d), dim=0) # (B+1)xNx3
            pcd_all = o3d.geometry.PointCloud()
            for i in range(len(pts3d)):
                pcd_tmp = o3d.geometry.PointCloud()
                pcd_tmp.points = o3d.utility.Vector3dVector(pts3d[i].cpu().numpy())
                if i == len(pts3d) - 1:
                    # orange
                    pcd_tmp.paint_uniform_color([1, 0.706, 0])
                else:
                    # blue
                    pcd_tmp.paint_uniform_color([0, 0.651, 0.929])
                pcd_all += pcd_tmp
            o3d.visualization.draw_geometries([pcd_all])
            """

            # --------DBUEG: visualize the final optimized point cloud--------
            """
            _poses = SE3(self.video.poses[mast3r_t0:mast3r_t1]).inv().data
            _depths = self.video.disps[mast3r_t0:mast3r_t1].clone()
            points = droid_backends.iproj(_poses, _depths, self.video.intrinsics[0])
            pcd_all = o3d.geometry.PointCloud()
            for i in range(len(points)):
                _points = points[i].cpu().numpy().reshape(-1, 3)
                # clip the depths of _points to 3*median
                _points = _points
                _pcd = o3d.geometry.PointCloud()
                _pcd.points = o3d.utility.Vector3dVector(_points)
                _color = self.video.images[i+mast3r_t0][[2,1,0], 3::8,3::8].cpu().numpy().transpose(1,2,0) / 255.0
                _color = _color.reshape(-1,3)
                _pcd.colors = o3d.utility.Vector3dVector(_color)
                pcd_all += _pcd
            o3d.visualization.draw_geometries([pcd_all])
            """
            # -----------------------------------------------------------------

            # update the keyframe index
            self.mast3r_keyframe = self.mast3r_keyframe + 1

            # TODO: better organize the code
            self.__droid_refine()

        else: # error frames
            raise ValueError("Invalid keyframe index")

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value

        if self.use_gt_calib and self.mast3r_pred:
            raise ValueError("Error: choose one of the calibration method")
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
                scene = mast3r_inference_init(self.mast3r_image_buffer, self.mast3r_model, 'cuda') # on the device of the cuda.
            # prepare the intrinsic parameters
            focals = scene['focals']; cx = self.video.wd / 2; cy = self.video.ht / 2
            avg_focal = focals.mean().item()
            # initialize with the estimated camera poses
            cam2world = scene['poses']; world2cam = cam2world.inverse()
                #matrix -> quaternion + translation
            quats = roma.rotmat_to_unitquat(world2cam[:, :3, :3]) # (x,y,z,w)
            trans = world2cam[:, :3, 3] # (x,y,z), in the droid, the poses are (xyz, xyzw)
            mast3r_world2cam = torch.cat((trans, quats), dim=1)
            self.video.poses[:self.t1] = mast3r_world2cam

            self.video.intrinsics[:self.t1] = torch.tensor([avg_focal, avg_focal, cx, cy]).cuda() / self.RES
            # prepare the predicted depths
            mast3r_depths = scene['depths'][None]
            mast3r_depths = F.interpolate(mast3r_depths, scale_factor=1/self.RES, mode='bilinear').squeeze()
            # feed the depths into the video frames
            self.video.disps_sens[:self.t1] = torch.where(mast3r_depths > 0, 1.0/mast3r_depths, 0)

            if self.mast3r_init_only:
                # delete the model to reduce the memory usage
                del self.mast3r_model
                self.mast3r_model = None
                torch.cuda.empty_cache()

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
        # DEBUG: up-sample the generated depth map and compared to initial mast3r prediction.
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

        # 2. the Mast3R optimized point cloud
        """
        mast3r_poses = scene['poses'].cpu().numpy() # (N,4,4)
        mast3r_intr = self.video.intrinsics[0].cpu().numpy() * self.RES
        droid_poses = SE3(self.video.poses[:self.t1]).inv().matrix().cpu().numpy() # camera to world
        pcd_all = o3d.geometry.PointCloud()
        for i in range(self.t1):
            mast3r_depth = scene['depths'][i].cpu().numpy()
            mast3r_points = droid_transform.depth2points(mast3r_depth, mast3r_intr)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mast3r_points.reshape(-1,3))
            color = self.video.images[i][[2,1,0]].cpu().numpy().transpose(1,2,0) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(color.reshape(-1,3))
            # pcd.transform(mast3r_poses[i])
            pcd.transform(droid_poses[i])
            pcd_all += pcd
        o3d.visualization.draw_geometries([pcd_all])
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
        self.mast3r_keyframe = self.t1

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value: # means new frame comes in
            if self.mast3r_pred:
                # self.___motion_only_ba_detection()
                self.__update()
            else:
                self.__update()

            if self.rr_vis is not None:
                self.rr_vis(True, True, True, True, None)


