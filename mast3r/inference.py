import PIL
import torchvision.transforms as tvf
import numpy as np
import torch
import open3d as o3d
from collections import namedtuple
import torch.nn as nn
import roma
from tqdm import tqdm
import torch.nn.functional as F

from dust3r.image_pairs import make_pairs
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.cloud_opt.commons import (edge_str, signed_expm1, signed_log1p,
                                      ALL_DISTS, cosine_schedule, linear_schedule)
from dust3r.optim_factory import adjust_learning_rate_by_lr
from mast3r.cloud_opt.sparse_ga import paris_asymmetric_inference, paris_symmetric_inference
from mast3r.fast_nn import fast_reciprocal_NNs_sample
from mast3r.usage import rigid_points_registration

import droid_backends
from lietorch import SE3

PairOfSlices = namedtuple('ImgPair', 'img1, slice1, img2, slice2')

def mast3r_simple_align(preds, N, device='cuda'):
    sample_n = 4096 # number of correspondence samples
    p_dtype = torch.float32
    p_device = device

    H,W = preds[next(iter(preds))][0]['pts3d'].shape[1], preds[next(iter(preds))][0]['pts3d'].shape[2]
    # average the point clouds
    identity_points = [[] for _ in range(N)]
    identity_conf=[[] for _ in range(N)]
    average_points = []
    average_confs = []
    correspondence=[]
    # get the model predictions
    for pred in preds.items():
        img_idx1, img_idx2 = int(pred[0][0]), int(pred[0][1])
        X11, X21, X22, X12 = [r['pts3d'][0] for r in pred[1]]
        C11, C21, C22, C12 = [r['conf'][0] for r in pred[1]]
        identity_points[img_idx1].append(X11); identity_conf[img_idx1].append(C11)
        identity_points[img_idx2].append(X22); identity_conf[img_idx2].append(C22)

        # get the pair-wise mutual correspondence
        F11, F21, F22, F12 = [r['desc'][0] for r in pred[1]] # fetch features: H,W,24
        random_indices_1=torch.randperm(H*W)[:sample_n] # we can `samples` indices in the range of H*W
        random_indices_2=torch.randperm(H*W)[:sample_n] # we can `samples` indices in the range of H*W
        f11, f21 = fast_reciprocal_NNs_sample(F11, F21, random_indices_1, ret_xy='y_x', max_iter=10, dist='dot', block_size=2**13) # (N,2)
        f22, f12 = fast_reciprocal_NNs_sample(F22, F12, random_indices_2, ret_xy='y_x', max_iter=10, dist='dot', block_size=2**13) # (N,2)
        # fuse the two pairs
        f1 = np.r_[f11, f12] # the index begins with (H,W) order
        f2 = np.r_[f21, f22]
        # modify the correspondence, saving one pair of information
        correspondence.append(PairOfSlices(img_idx1, f1, img_idx2, f2))

    # initialize the initial first-view prediction
    for i in range(N):
        # too many for loops, `for` loop should be replaced by `torch` operation
        identity_points_i = torch.stack(identity_points[i], dim=0) # BxHxWx3
        identity_conf_i = torch.stack(identity_conf[i], dim=0).unsqueeze(-1) # BxHxWx1
        weighted_avg_points = torch.sum(identity_points_i * identity_conf_i, dim=0) / torch.sum(identity_conf_i, dim=0)
        # estimate the camera focal with the full-resolution prediction
        average_points.append(weighted_avg_points) # H,W,3
        average_confs.append(torch.mean(identity_conf_i, dim=0)) # H,W,1

    ### fix the first view, run the global 3D-3D alignment
    # build the optimizable parameters
    camera_pp=torch.tensor([W/2, H/2], device=p_device, dtype=p_dtype).expand(N, 2).clone()
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=p_dtype, device=p_device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(N)]
    trans = [nn.Parameter(torch.zeros(3, device=p_device, dtype=p_dtype)) for _ in range(N)]
    log_scales = [nn.Parameter(torch.zeros(1, device=p_device, dtype=p_dtype)) for _ in range(N)]
    # we fix the first frame's prediction as the reference to maintain the scale; and the first frame's pose is identity
    # do the initial alignment
    fixed_idx = 0
    for pair in correspondence:
        img_idx1, slice1, img_idx2, slice2 = pair #note pair is unique
        if img_idx1 == fixed_idx:
            tgt_pts = average_points[img_idx1][slice1[:, 0], slice1[:, 1]].reshape(-1, 3)
            src_pts = average_points[img_idx2][slice2[:, 0], slice2[:, 1]].reshape(-1, 3)
            confs = average_confs[img_idx2][slice2[:, 0], slice2[:, 1]].reshape(-1, 1)
            # estimate the camera pose
            s, R, T = rigid_points_registration(src_pts, tgt_pts, confs)
            quats[img_idx2].data = roma.rotmat_to_unitquat(R)
            trans[img_idx2].data = T
            log_scales[img_idx2].data = torch.log(torch.tensor([s], device=p_device, dtype=p_dtype))
        elif img_idx2 == fixed_idx:
            tgt_pts = average_points[img_idx2][slice2[:, 0], slice2[:, 1]].reshape(-1, 3)
            src_pts = average_points[img_idx1][slice1[:, 0], slice1[:, 1]].reshape(-1, 3)
            confs = average_confs[img_idx1][slice1[:, 0], slice1[:, 1]].reshape(-1, 1)
            # estimate the camera pose
            s, R, T = rigid_points_registration(src_pts, tgt_pts, confs)
            quats[img_idx1].data = roma.rotmat_to_unitquat(R)
            trans[img_idx1].data = T
            log_scales[img_idx1].data = torch.log(torch.tensor([s], device=p_device, dtype=p_dtype))
        else:
            continue

    # we do not do the gradient-based optimization, we only need to return the initial alignment directly.
    cam2ws = torch.eye(4, dtype=p_dtype, device=p_device)[None].expand(N, 4, 4).clone()
    cam2ws[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats, dim=0), dim=1))
    cam2ws[:, :3, 3] = torch.stack(trans, dim=0)
    scales = torch.cat(log_scales).exp()
    focals = torch.ones(N, device=p_device, dtype=p_dtype)
    depths = torch.zeros(N, H, W, device=p_device, dtype=p_dtype)
    for i in range(N):
        points = average_points[i] * scales[i]
        # recover the focal length
        focal = estimate_focal_knowing_depth(points[None], camera_pp[i].unsqueeze(0), focal_mode='weiszfeld')
        focals[i] = focal
        # from points to depths
        depths[i] = points[..., 2]
    return {
        'poses': cam2ws.detach(),
        'focals': focals.detach(),
        'depths': depths.detach()
    }

def NoGradParamDict(x):
    assert isinstance(x, dict)
    return nn.ParameterDict(x).requires_grad_(False)

def get_conf_trf(mode):
        if mode == 'log':
            def conf_trf(x): return x.log()
        elif mode == 'sqrt':
            def conf_trf(x): return x.sqrt()
        elif mode == 'm1':
            def conf_trf(x): return x-1
        elif mode in ('id', 'none'):
            def conf_trf(x): return x
        else:
            raise ValueError(f'bad mode for {mode=}')
        return conf_trf

def _ravel_hw(x):
    x = x.view((x.shape[0] * x.shape[1],) + x.shape[2:])
    return x

def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    params = [_ravel_hw(p) for p in params]
    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params

class LocalBA(nn.Module):
    def __init__(self, known_poses, known_depths, preds, intrinsics, device='cuda'):
        super().__init__()
        """ With the known poses and depths, optimizing the last frame's pose and depth """
        ### prepare the data
        assert len(known_poses) == len(known_depths)
        self.POSE_DIM = 7
        self.H, self.W = known_depths[0].shape
        self.known_cams = len(known_poses) - 1 # the last frame is the target
        self.opt_cams = 1
        self.device = device
        self.known_poses = known_poses
        self.known_inv_depths = 1.0/known_depths
        self.known_intrinsics = intrinsics
        self.dist = ALL_DISTS['l1']
        # TODO
        # symmetry = True if len(next(iter(preds.items()))[1]) > 2 else False
        self.symmetrized = False
        self.edges = [(int(i), int(j)) for i, j in preds.keys()]
        # optimizable parameters
        # model predicted data
        pred_pts = [pred for pred in preds.values()]
        pred1_pts = [pts[0]['pts3d'].squeeze(0) for pts in pred_pts]
        pred2_pts = [pts[1]['pts3d'].squeeze(0) for pts in pred_pts] # if symmetrize, then we have [3]and[4] to generate more pairs
        pred1_conf = [pts[0]['conf'].squeeze(0) for pts in pred_pts]
        pred2_conf = [pts[1]['conf'].squeeze(0) for pts in pred_pts]
        self.pred_pts_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
        self.pred_pts_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})
        self.pred_conf_i = NoGradParamDict({ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)})
        self.pred_conf_j = NoGradParamDict({ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)})
        self.im_conf = self._compute_img_conf(pred1_conf, pred2_conf)
        # utilized functions
        self.conf_trf = get_conf_trf(mode='log')
        # The pair-wise prediction parameters
        self.register_buffer('_weight_i', ParameterStack([self.conf_trf(self.pred_conf_i[i_j]) for i_j in self.str_edges])) # list to stack
        self.register_buffer('_weight_j', ParameterStack([self.conf_trf(self.pred_conf_j[i_j]) for i_j in self.str_edges]))
        self.register_buffer('_stacked_pred_pts_i', ParameterStack(self.pred_pts_i, self.str_edges))
        self.register_buffer('_stacked_pred_pts_j', ParameterStack(self.pred_pts_j, self.str_edges))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        im_shapes = [(self.H, self.W) for _ in range(self.known_cams + self.opt_cams)] # TODO: in fact, should align to the input image size
        im_areas = [H * W for H, W in im_shapes]
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

        # optimizable pair-wise alignment poses
        self.pw_poses = nn.Parameter(torch.randn((self.n_edges, 1+self.POSE_DIM))) # [rotation, translation, scale] <-> [x, y, z, w, tx, ty, tz, scale], scale is the log scale

        # initialize the last frame depth and pose for optimization
        # use the `log` function to avoid the negative depth
        self.opt_depths = nn.Parameter(torch.log(known_depths[-1].clone()).ravel())
        # known_poses: [translation, rotation] <-> [tx, ty, tz, x, y, z, w]; opt_pose: [rotation, translation] <-> [x, y, z, w, tx, ty, tz]
        last_frame_cam2w = SE3(known_poses[-1]).inv().data
        _opt_pose = last_frame_cam2w.clone()[[3, 4, 5, 6, 0, 1, 2]]
        _opt_pose[4:7] = signed_log1p(_opt_pose[4:7])
        self.opt_pose = nn.Parameter(_opt_pose.clone()) # opt_pose: [rotation, translation], and translation is the log scale

        # rerun inspection
        ### --------debug-------- ###
        """"
        self.rr_iter = 0
        rr.init("debug")
        rr.connect()
        """
        ### --------debug-------- ###

        # TODO: better organize the code
        self.initialize_local_map_params()

    def _set_pose(self, poses, i, R, T=None, scale=None, force=False):
        # all poses == cam-to-world
        pose = poses[i]

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    def initialize_local_map_params(self):
        """ Since gradient-based optimization method is sensitive to initial guess, we need to initialize the pw_poses instead of random values """
        reference = self.get_fixed_pts3d()[:self.known_cams] # BxNx3
        # align the pred_j to the reference to initialize the `pw_poses`
        for e, (i, j) in enumerate(self.edges):
            # TODO
            # Since currently we use the `ref-last_frame` prediction, the reference prediction is always the (j) in the (ij) pair
            pred_j = self._stacked_pred_pts_j[j]
            ref_j = reference[j]
            # estimate the camera pose
            R, T, s = roma.rigid_points_registration(pred_j, ref_j, weights=self._weight_j[j], compute_scaling=True)
            self._set_pose(self.pw_poses, e, R, T, s)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]

    def _get_poses(self, poses):
        # normalize rotation
        if poses.dim() < 2:
            poses = poses[None]
        Q = poses[:, :4]
        T = signed_expm1(poses[:, 4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT

    @torch.no_grad()
    def _compute_img_conf(self, pred1_conf, pred2_conf):
        im_conf = nn.ParameterList([torch.zeros(self.H, self.W, device=self.device) for i in range(self.known_cams + self.opt_cams)])
        for e, (i, j) in enumerate(self.edges):
            im_conf[i] = torch.maximum(im_conf[i], pred1_conf[e])
            im_conf[j] = torch.maximum(im_conf[j], pred2_conf[e])
        return im_conf

    def get_pw_scale(self):
        """ return the optimized scale of pair-wise prediction """
        # since we have the reference depth map, the just return the optimized scale;
        # due to the scale should always be positive, we use the exponential function
        return self.pw_poses[:, -1].exp()

    def get_pw_poses(self):
        """ get the pair-wise prediction poses"""
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)
        return scaled_RT

    def get_fixed_pts3d(self):
        points = droid_backends.iproj(SE3(self.known_poses).inv().data, self.known_inv_depths, self.known_intrinsics) # BxHxWx3
        return points.view(len(points), -1, 3)

    def get_opt_pose(self):
        RT = self._get_poses(self.opt_pose)
        scaled_RT = RT.clone()
        # NOTE: no scale here
        return scaled_RT

    @property
    def get_opt_depth(self):
        return self.opt_depths.exp()

    @property
    def get_opt_depth_droid(self):
        opt_depth = self.opt_depths.exp().detach().clone()
        # reshape
        opt_depth = opt_depth.view(self.H, self.W)
        return opt_depth

    @property
    def get_opt_pose_droid(self):
        """ get the optimized pose in the droid format """
        opt_pose = self.opt_pose.clone().detach()
        opt_pose.data[4:7] = signed_expm1(opt_pose.data[4:7]) # translation by exp
        # re-order to the droid format [tx, ty, tz, x, y, z, w]
        opt_pose.data[:] = opt_pose[[4, 5, 6, 0, 1, 2, 3]]
        return opt_pose

    def get_opt_points(self):
        """project the optimized depth to the 3d points in the local coordinate"""
        """ TODO: make get_points methods a unified way"""
        f, _, cx, cy = self.known_intrinsics
        pp = torch.tensor([cx, cy], device=self.device, dtype=self.opt_depths.dtype)
        depth = self.get_opt_depth.unsqueeze(1) # (HxW), one-dimensional, unsqueeze to match the grid(HxW,1)

        grid = xy_grid(self.W, self.H, device=self.device)
        grid = grid.view((grid.shape[0] * grid.shape[1],) + grid.shape[2:]) # HxWx2

        points = torch.cat((depth * (grid - pp) / f, depth), dim=-1)
        return points

    def get_opt_pts3d(self):
        opt_pose = self.get_opt_pose()
        opt_points = self.get_opt_points()[None]
        # NOTE
        opt_points = geotrf(opt_pose, opt_points) # (B, 4, 4) x (B, N, 3)

        """ visualize and compare the initialzed points
        known_poses = self.known_poses
        known_poses = SE3(known_poses).inv().data #identity
        known_points = droid_backends.iproj(known_poses, self.known_inv_depths, self.known_intrinsics)[-1]
        known_points = known_points.view(-1, 3)
        opt_points = opt_points.squeeze(0).view(-1, 3)

        pcd_known = o3d.geometry.PointCloud()
        pcd_known.points = o3d.utility.Vector3dVector(known_points.detach().cpu().numpy())
        pcd_opt = o3d.geometry.PointCloud()
        pcd_opt.points = o3d.utility.Vector3dVector(opt_points.detach().cpu().numpy())
        pcd_known.paint_uniform_color([1, 0, 0])
        pcd_opt.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([pcd_known, pcd_opt])
        """
        return opt_points

    def forward(self):
        """ The pair-wise global alignment """
        pw_poses = self.get_pw_poses()
        fixed_pts3d = self.get_fixed_pts3d()[:self.known_cams] # BxNx3
        opt_pts3d = self.get_opt_pts3d() # 1xNx3
        pts3d = torch.cat((fixed_pts3d, opt_pts3d), dim=0) # (B+1)xNx3

        aligned_pred_i = geotrf(pw_poses, self._stacked_pred_pts_i)
        aligned_pred_j = geotrf(pw_poses, self._stacked_pred_pts_j)

        #### --------debug-------- ####
        """
        points0 = fixed_pts3d[4].clone().detach().cpu().numpy()
        points1 = aligned_pred_j[4].clone().detach().cpu().numpy()
        rr.log("world/points0", rr.Points3D(points0))
        rr.log("world/points1", rr.Points3D(points1))
        rr.set_time_sequence("#frame", self.rr_iter)
        self.rr_iter += 1
        """
        #### --------debug-------- ####

        li = self.dist(pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j
        return li + lj

def mast3r_inference_init(images, model, device='cuda'):
    """
    The simple mast3r-based model prediction alignment, we only predict via the first-frame alignment.
    Namely, all other frames will be aligned to the first frame.
    """
    # make image pairs, pairs only need to get the one-forward pass (no need to symmetric pairs)
    pairs = make_pairs(images, scene_graph='oneref-0', prefilter=None, symmetrize=False) # use the 0 as the reference.
    N_images = len(images)

    # get the model prediction
    out = paris_symmetric_inference(pairs, model, device) # C_n^2 pairs, each with index of the image

    # run the group BA (with refinement)
    scene = mast3r_simple_align(out, N_images, device=device)

    # return the result
    return scene

@torch.cuda.amp.autocast(enabled=False)
def global_alignment_loop(scene, lr=0.01, niter=300, schedule='cosine', lr_min=1e-6):
    params = [p for p in scene.parameters() if p.requires_grad]
    if not params:
        return scene

    # TODO: verbose or not to print detailed information
    print("global alignment, optimizing for:")
    print([name for name, value in scene.named_parameters() if value.requires_grad])

    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))
    with tqdm(total=niter) as pbar:
        while pbar.n < pbar.total:
            loss = global_alignment_iter(scene, pbar.n, niter, lr_base, lr_min, optimizer, schedule)
            pbar.set_postfix_str(f'{lr=:g} loss={loss:g}')
            pbar.update()
    return loss

def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f'bad lr {schedule=}')
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()
    loss = net()
    loss.backward()
    optimizer.step()

    return float(loss)

def mast3r_inference(images, cam_poses, inv_depths, RES, intrinsics, model, device='cuda'):
    """
    The dust3r-like global alignment, use the preset camera poses and depths to predict the last frame's pose and depth.

    [Optional]: Only fixing the first frame
    """
    last_idx = len(images) - 1
    pairs = make_pairs(images, scene_graph=f'oneref-{last_idx}', prefilter=None, symmetrize=False) # use the last frame as the reference.
    N_images = len(images)
    # dust3r-based global alignment
    out = paris_symmetric_inference(pairs, model, device) # C_n^2 pairs, each with index of the image; the output will be the dict, two-pair will be the predicted 3d points

    # upsample the DROID depths to the target resolution
    depths = (1.0 / inv_depths)[None] # inverse of the depth
    depths = F.interpolate(depths, scale_factor=RES, mode='bilinear').squeeze().clone()
    intrinsics = intrinsics * RES

    #TODO: clean the point cloud, for those depths larger than 2xmedian, make them 2xmedian
    for i in range(N_images):
        depths[i] = depths[i].clamp(0, 2 * depths[i].median())

    """
    pcd = o3d.geometry.PointCloud()
    points = droid_backends.iproj(SE3(cam_poses).inv().data, 1/depths, intrinsics)[-1].detach().cpu().numpy().reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    """

    with torch.enable_grad():
        scene = LocalBA(cam_poses, depths, out, intrinsics, device=device).to(device)
        # construct the optimizer and do the optimization loop
        global_alignment_loop(scene)
    return scene