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
from mast3r.cloud_opt.sparse_ga import paris_asymmetric_inference, paris_symmetric_inference
from mast3r.fast_nn import fast_reciprocal_NNs_sample
from mast3r.usage import rigid_points_registration

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


def mast3r_inference(images, model, device='cuda'):
    # make image pairs, pairs only need to get the one-forward pass (no need to symmetric pairs)
    pairs = make_pairs(images, scene_graph='oneref-0', prefilter=None, symmetrize=False) # use the 0 as the reference.
    N_images = len(images)

    # get the model prediction
    out = paris_symmetric_inference(pairs, model, device) # C_n^2 pairs, each with index of the image

    # run the group BA (with refinement)
    scene = mast3r_simple_align(out, N_images, device=device)

    # return the result
    return scene