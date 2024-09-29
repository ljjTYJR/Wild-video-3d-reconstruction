##
# @ Desc:
#   The frontend module of the Mast3r-based SLAM
#   The frontend will receive the keyframe from the motion filter, and then do the local bundle adjustment based on local keyframes.
#   The frame information will be stored in the Mast3rVideo object.
# @ Author: Shuo Sun
##

from dust3r.utils.image import load_images, format_images, format_mast3r_out
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import paris_asymmetric_inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r_visualization import mast3r_visualize_matched_images
from dust3r.utils.geometry import geotrf, inv, get_med_dist_between_poses

from mast3r_pose import pnp_pose, sRT_to_4x4, matrix_to_7dof_pose
from mast3r_visualization import mast3r_o3d_vis_group_points

import open3d as o3d
import numpy as np
import torch
import roma
from lietorch import SE3

import droid_backends

class Mast3rFrontend:
    def __init__(self, mast3r_model, video):
        self.device='cuda:0'
        self.mast3r_model = mast3r_model
        self.video = video

        self.count = 0
        self.warmup = 8
        self.ba_window = 8

        # local ba setting
        self.niter = 200
        self.lr = 0.01
        self.schedule = 'cosine'
        self.batch_size = 1

        self.is_initialized = False

    def visualize_pcd(self, scene):
        """ Visualize the scene """
        pts = to_numpy(scene.get_pts3d()) # already global frame Nx[H,W,3]
        rgbimg = scene.imgs
        cams2world = scene.get_im_poses().cpu()
        imgs = to_numpy(rgbimg) # NXHxWx3 (0,1)
        pcd_all = o3d.geometry.PointCloud()
        for pt, img in zip(pts, imgs):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3))
            pcd_all += pcd
        o3d.visualization.draw_geometries([pcd_all])

    def local_dust3r_ba(self, images, scene_graph='complete', prefilter='seq3', symmetrize=False):
        images = format_images(images)
        pairs = make_pairs(images, scene_graph=scene_graph, prefilter=prefilter, symmetrize=symmetrize)
        out = paris_asymmetric_inference(pairs, self.mast3r_model, self.device)
        res = format_mast3r_out(pairs, out)
        scene = global_aligner(res, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr)
        return scene

    def dump_data_to_video(self, scene, indices):
        """ Dump the scene to the video """
        # camera pose(), inverse_depth, estimated intrinsics;
        depths = torch.stack(scene.get_depthmaps(raw=False), dim=0)
        self.video.disps[indices[0]:indices[-1]+1] = 1/depths # stored in the video is the inverse depth

        im_poses = scene.get_im_poses() # (N,4,4), cam2world
        world2cam = im_poses.inverse()
        quats = roma.rotmat_to_unitquat(world2cam[:, :3, :3]); trans = world2cam[:, :3, 3]
        self.video.poses[indices[0]:indices[-1]+1] = torch.cat((trans, quats), dim=1)

        intrinsics = scene.get_intrinsics() # (N, 3, 3)
        intrinsics = torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]], dim=1)
        self.video.intrinsics[indices[0]:indices[-1]+1] = intrinsics

    def frame_selection_strategy(self, new_model_idx):
        """ how to select the new model frame indices and the last model frame indices """
        last_model_idx = new_model_idx - 1
        if last_model_idx < 0:
            raise ValueError("The last model index should be larger than 0")
        return last_model_idx

    def prune_match_by_depth_conf(self, matches_im0, matches_im1, depth0, depth1, conf0, conf1):
        # TODO: test whether need to prune the matches
        # Note that the depths are the inverse of the real depth
        mask = (depth0[matches_im0[:, 1], matches_im0[:, 0]] > (depth0.median()/2) ) & \
                (depth1[matches_im1[:, 1], matches_im1[:, 0]] > (depth1.median()/2))
        mask = mask & (conf0[matches_im0[:, 1], matches_im0[:, 0]] > conf0.median()) & \
                (conf1[matches_im1[:, 1], matches_im1[:, 0]] > conf1.median())
        mask = mask.detach().cpu().numpy()
        matches_im0, matches_im1 = matches_im0[mask], matches_im1[mask]
        return matches_im0, matches_im1

    def get_points_in_video(self, image_idx, local=False):
        inverse_depth = self.video.disps[image_idx]
        intrinsic = self.video.intrinsics[image_idx]
        if local:
            pose = SE3(torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")).data # local coordinate
        else:
            pose = SE3(self.video.poses[image_idx]).inv().data # global coordinate
        points = droid_backends.iproj(pose[None], inverse_depth[None], intrinsic)
        return points.squeeze(0) # (H,W),3

    def model_matching(self, matches_im0, matches_im1, new_model_idx, last_model_idx):
        """ match the two models: first Run the PnP to estimate the pose and then Run scale-RANSAC to estimate the scale """
        last_model_points = self.get_points_in_video(last_model_idx) # global coordinate
        ref_points = last_model_points[matches_im1[:, 1], matches_im1[:, 0]] # (N,3)
        selected_pixels = matches_im0 # (N,2)
        intrinsic = self.video.intrinsics[new_model_idx]
        rot, trans, inlier_idx = pnp_pose(ref_points, selected_pixels, intrinsic, pnp_iters=15, device=self.device)
        cam2world = inv(sRT_to_4x4(1, rot, trans, self.device)) # here, the world coordinate is the last model coordinate

        # compute the scale by the scale-RANSAC
        new_model_points = self.get_points_in_video(new_model_idx, local=True) # local coordinate
        selected_points = new_model_points[matches_im0[:, 1], matches_im0[:, 0]] # (N,3)
        selected_points = geotrf(cam2world, selected_points) # (N,3)

        ref_points = ref_points[inlier_idx.squeeze()]
        selected_points = selected_points[inlier_idx.squeeze()]

        ab = torch.sum(ref_points * selected_points, dim=1, keepdim=True)
        b2 = torch.sum(selected_points ** 2, dim=1, keepdim=True)
        s = (ab / b2).mean()

        # DEBUG: transfer to video structure and then visualize them
        """
        trans_quats = matrix_to_7dof_pose(cam2world)
        new_pose = SE3(trans_quats).inv()
        self.video.poses[new_model_idx] = new_pose.data
        self.video.disps[new_model_idx] /= s
        poses = SE3(self.video.poses[new_model_idx-1:new_model_idx+1]).inv().data
        disps = self.video.disps[new_model_idx-1:new_model_idx+1]
        intrinsics = self.video.intrinsics[new_model_idx-1:new_model_idx+1][0]
        pts = droid_backends.iproj(poses, disps, intrinsics)
        mast3r_o3d_vis_group_points(pts)
        """
        return cam2world, s # new model's pose and the scale

    def model_adjustment(self, new_model_idx, cam2world, s):
        """ Adjust the new model camera pose and scene scale via the first image """
        trans_quats = matrix_to_7dof_pose(cam2world)
        new_pose = SE3(trans_quats).inv() # world2cam
        new_pose.data[:3] /= s
        original_pose = SE3(self.video.poses[new_model_idx]) # already world2cam

        delta = new_pose * original_pose.inv() # the delta pose
        delta_log = (delta).log() # the delta pose

        # TODO: to matrix operation, check how to do that in a batch way
        for i in range(new_model_idx, new_model_idx+self.ba_window):
            original_pose = SE3(self.video.poses[i])
            new_pose = (SE3.exp(delta_log) * original_pose).data
            self.video.poses[i] = new_pose

        self.video.disps[new_model_idx:new_model_idx+self.ba_window] /= s # the inverse depth, so divide the scale
        self.video.poses[new_model_idx:new_model_idx+self.ba_window][:, :3] *= s # the translation, the first camera is already fixed
        pass

    def align_two_traj(self, new_model_idx):
        """ align two separate trajectories """
        last_model_idx = self.frame_selection_strategy(new_model_idx)
        last_keyframe_img = self.video.images[last_model_idx]
        cur_keyframe_img = self.video.images[new_model_idx]
        imgs = torch.stack([last_keyframe_img, cur_keyframe_img], dim=0) # last:0; cur:1

        imgs = format_images(imgs) # order along the input order
        pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=False)
        if len(imgs) > 2:
            raise ValueError("Not implemented yet")
        else:
            output = paris_asymmetric_inference(pairs, self.mast3r_model, self.device)
            for key, values in output.items():
                key1, key2 = key
                key1, key2 = int(key1), int(key2) #image index
                assert key1 == 1 and key2 == 0
                cur_pred = values[0]
                ref_pred = values[1]
        # find 2D-2D matches between the two images
        cur_desc, ref_desc = cur_pred['desc'].squeeze(0).detach(), ref_pred['desc'].squeeze(0).detach()
        cur_desc_conf, ref_desc_conf = cur_pred['desc_conf'].squeeze(0).detach(), ref_pred['desc_conf'].squeeze(0).detach()
        # every 4 pixels sampling
        matches_im0, matches_im1 = fast_reciprocal_NNs(cur_desc, ref_desc, subsample_or_initxy1=4,
                                                   device=self.device, dist='dot', block_size=2**13) # the resulted mathces are 2d pixels, not 1D indices
        matches_im0, matches_im1 = self.prune_match_by_depth_conf(matches_im0, matches_im1, self.video.disps[new_model_idx], self.video.disps[last_model_idx], cur_desc_conf, ref_desc_conf)

        # mast3r_visualize_matched_images(cur_keyframe_img, last_keyframe_img, matches_im0, matches_im1, n_viz=20)
        cam2world, s = self.model_matching(matches_im0, matches_im1, new_model_idx, last_model_idx)
        self.model_adjustment(new_model_idx, cam2world, s)

    def __initialize__(self):
        """ Initialize the frontend; After the initialization, we will fix the intrinsic for following tracking; also, the scale. """
        initial_images = self.video.images[:self.ba_window]
        scene = self.local_dust3r_ba(initial_images, scene_graph='complete', prefilter='seq3', symmetrize=False)

        # append the value to the video
        self.dump_data_to_video(scene, indices=np.arange(self.ba_window))
        self.is_initialized = True

    def __track__(self):
        """ Video local Mast3r BA except the initialization; Use the initialized intrinsic """
        window_start = self.video.counter.value - self.ba_window
        window_end = self.video.counter.value

        traked_images = self.video.images[window_start:window_end]
        scene = self.local_dust3r_ba(traked_images, scene_graph='complete', prefilter='seq3', symmetrize=False)

        self.dump_data_to_video(scene, indices=np.arange(window_start, window_end))
        # align two model(or, the trajectory)
        self.align_two_traj(window_start)

        # poses = SE3(self.video.poses[:window_end]).inv().data
        # disps = self.video.disps[:window_end]
        # intrinsics = self.video.intrinsics[:window_end][0]
        # pts = droid_backends.iproj(poses, disps, intrinsics)
        # mast3r_o3d_vis_group_points(pts)

    def __call__(self):
        """ The main thread of the frontend """
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize__()
        elif self.is_initialized and self.video.counter.value % self.ba_window == 0:
            self.__track__()
        else:
            pass

