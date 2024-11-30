# Use the dust3r/mast3r model to initialize the scene.
# Use the first frame scale and camera pose and the initial scene.

import open3d as o3d
import torch
from dust3r.inference import load_model
from dust3r.dust3r_type import set_as_dust3r_image, dust3r_inference
from mast3r.inference import local_dust3r_ba
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import symmetric_inference
from dust3r.utils.image import format_images
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs as make_pairs_dust3r
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import torchvision.transforms as tvf
from tqdm import tqdm
import copy
import cv2

def single_img_pred(mast3r_img, model):
    """ Use the dust3r/mast3r model to predict the 3D points for a single image """
    pred = inference([(mast3r_img, mast3r_img)], model, device='cuda', batch_size=1, verbose=True)
    pts = pred['pred1']['pts3d']
    pose = torch.eye(4)
    return pts, pose

def intrinsic_preset_format(dpvo_intrinsics):
    """ Format the intrinsic matrix to the dust3r format """
    scale=4.0 # By default, the DPVO scale is 4.0
    preset_intrinsics = torch.eye(3).repeat(len(dpvo_intrinsics), 1, 1)
    preset_intrinsics[:, 0, 0] = dpvo_intrinsics[:, 0] * scale
    preset_intrinsics[:, 1, 1] = dpvo_intrinsics[:, 1] * scale
    preset_intrinsics[:, 0, 2] = dpvo_intrinsics[:, 2] * scale
    preset_intrinsics[:, 1, 2] = dpvo_intrinsics[:, 3] * scale
    return preset_intrinsics

def depth_preset_format(prior_pts):
    """ Convert the prior-achieved points to depth map"""
    n, H, W, _ = prior_pts.shape
    mast3r_depths = torch.zeros(n, H, W)
    mast3r_depths = prior_pts[:, :, :, 2]
    return mast3r_depths

@torch.no_grad()
def dpvo_mast3r_initialization(images, mast3r_model, intrinsics=None, device='cuda'):
    """ Use the mast3r model prediction to initialize the scene;
    The first frame will be used as the reference (both scale and camera pose)
    """
    mast3r_images = format_images(images)
    pts, pose = single_img_pred(mast3r_images[0], mast3r_model)

    pairs = make_pairs_dust3r(mast3r_images, scene_graph='complete', prefilter=None, symmetrize=False)
    res = inference(pairs, mast3r_model, device, batch_size=1,verbose=True)
    modular_scene = global_aligner(res, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, fx_and_fy=True)

    # Preset the known information during the optimization
    pose_msk = [True] + [False] * (len(images) - 1)
    pts_msk = [True] + [False] * (len(images) - 1)
    intrinsic_msk = [True] * len(images)
    modular_scene.preset_pose([pose], pose_msk)
    mast3r_intrinsics = intrinsic_preset_format(intrinsics)
    modular_scene.preset_intrinsics(mast3r_intrinsics, intrinsic_msk)
    mast3r_depth = depth_preset_format(pts)
    modular_scene.preset_depthmap(mast3r_depth, pts_msk)

    # conduct the scene optimization
    with torch.enable_grad():
        loss = modular_scene.compute_global_alignment(init="mst", niter=250, schedule='cosine', lr=0.01)
    opt_depthmaps = modular_scene.get_depthmaps()
    opt_poses = modular_scene.get_im_poses()
    return opt_depthmaps, opt_poses


def collect_mast3r_prediction(pairs, model, desc_conf='desc_conf', device='cuda', subsample=8):
    pts_11={}; conf_11={}
    for img1, img2 in tqdm(pairs):
        res = symmetric_inference(model, img1, img2, device=device)
        X11, X21, X22, X12 = [r['pts3d'][0] for r in res] #X21: the second image points in the first image coordinate
        C11, C21, C22, C12 = [r['conf'][0] for r in res]
        if img1['idx'] not in pts_11:
            pts_11[img1['idx']] = {}
        if img2['idx'] not in pts_11:
            pts_11[img2['idx']] = {}
        if img1['idx'] not in conf_11:
            conf_11[img1['idx']] = {}
        if img2['idx'] not in conf_11:
            conf_11[img2['idx']] = {}
        pts_11[img1['idx']][img2['idx']] = X11
        pts_11[img2['idx']][img1['idx']] = X22
        conf_11[img1['idx']][img2['idx']] = C11
        conf_11[img2['idx']][img1['idx']] = C22
    return pts_11, conf_11

def canonical_view(pts11, conf11):
    # weighted average of the pts11 by confidence
    cancon_pts = {}
    for kp, vp, cp in zip(pts11.keys(), pts11.values(), conf11.values()):
        pt = torch.stack(list(vp.values()), dim=0)
        conf = torch.stack(list(cp.values()), dim=0)
        conf = conf.unsqueeze(-1) - 0.999
        canon = (conf * pt).sum(dim=0) / conf.sum(dim=0)
        cancon_pts[kp] = canon
    return cancon_pts

@torch.no_grad()
def dpvo_mast3r_optimization(images, poses, pts, mast3r_model, fixed=6, intrinsics=None, device='cuda'):

    def visualize_color_point_cloud(images, poses, pts):
        pcd_all = o3d.geometry.PointCloud()
        for i in range(len(pts)):
            pt = pts[i].cpu().numpy().reshape(-1, 3)
            pose = poses[i].cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            color = cv2.cvtColor(images[i].cpu().numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
            pcd.points = o3d.utility.Vector3dVector(pt)
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcd.transform(pose)
            pcd_all += pcd
        o3d.visualization.draw_geometries([pcd_all])

    def dus3r_coarse_alignment(pairs):
        res = inference(pairs, mast3r_model, device, batch_size=1,verbose=True)
        modular_scene = global_aligner(res, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, fx_and_fy=True)
        # Preset the known information during the optimization
        pose_msk = [True] * fixed + [False] * (len(images) - fixed)
        pts_msk = [False] * len(images)
        intrinsic_msk = [True] * len(images)
        poses_c2w = [pose.inv().matrix() for pose in poses]
        modular_scene.preset_pose(poses_c2w, pose_msk) # we can also initialize scene poses from the DPVO setting and then set the pair-wise poses
        mast3r_intrinsics = intrinsic_preset_format(intrinsics)
        modular_scene.preset_intrinsics(mast3r_intrinsics, intrinsic_msk)
        # conduct the scene optimization
        with torch.enable_grad():
            loss = modular_scene.compute_global_alignment(init="mst", niter=250, schedule='cosine', lr=0.01)
        opt_depthmaps = modular_scene.get_depthmaps()
        modular_scene.show_modified()
        opt_poses = modular_scene.get_im_poses()
        return

    """ Use mast3r model to initially estimate the DPVO camera poses
    """
    mast3r_images = format_images(images)
    pairs = make_pairs(mast3r_images, scene_graph='complete', prefilter='seq5', symmetrize=False)
    pts_11, conf_11 = collect_mast3r_prediction(pairs, mast3r_model, desc_conf='desc_conf', device=device, subsample=8)
    cancon_pts = canonical_view(pts_11, conf_11)
    pose_c2w = [pose.inv().matrix() for pose in poses]


    return
