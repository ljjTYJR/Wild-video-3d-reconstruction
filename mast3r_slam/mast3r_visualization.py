import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl
import cv2
import open3d as o3d

def mast3r_visualize_matched_images(img1, img2, matches1, matches2, n_viz=20):
    """ visualize the matches in the two image """
    H0, W0 = img1.shape[1], img1.shape[2]
    valid_matches_im0 = (matches1[:, 0] >= 3) & (matches1[:, 0] < int(W0) - 3) & (
        matches1[:, 1] >= 3) & (matches1[:, 1] < int(H0) - 3)

    H1, W1 = img2.shape[1], img2.shape[2]
    valid_matches_im1 = (matches2[:, 0] >= 3) & (matches2[:, 0] < int(W1) - 3) & (
        matches2[:, 1] >= 3) & (matches2[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches1, matches2 = matches1[valid_matches], matches2[valid_matches]

    # n_viz = 20
    num_matches = matches1.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches1[match_idx_to_viz], matches2[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([img1, img2]):
        # rgb_tensor = view['img'] * image_std + image_mean
        # viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
        view = view[[2, 1, 0], :, :]
        viz_imgs.append(view.permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)

def mast3r_o3d_vis_group_points(pts_from_backend):
    """ visualize the points in the 3D space """
    N = len(pts_from_backend)
    pcd_all = o3d.geometry.PointCloud()
    for i in range(N):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_from_backend[i].detach().cpu().numpy().reshape(-1,3))
        pcd_all += pcd
    o3d.visualization.draw_geometries([pcd_all])