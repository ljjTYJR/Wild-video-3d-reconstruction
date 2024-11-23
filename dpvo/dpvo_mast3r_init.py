# Use the dust3r/mast3r model to initialize the scene.
# Use the first frame scale and camera pose and the initial scene.

import open3d as o3d
import torch
from dust3r.inference import load_model
from dust3r.dust3r_type import set_as_dust3r_image, dust3r_inference
from mast3r.inference import local_dust3r_ba
from dust3r.utils.image import format_images
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import torchvision.transforms as tvf

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

    pairs = make_pairs(mast3r_images, scene_graph='complete', prefilter=None, symmetrize=False)
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
