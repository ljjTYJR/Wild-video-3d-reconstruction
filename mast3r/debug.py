import open3d as o3d
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl

def draw_two_point_clouds(points1, points2, image1, image2):
    pts1 = o3d.geometry.PointCloud()
    pts2 = o3d.geometry.PointCloud()
    if points1.shape[1] != 3:
        points1 = points1.reshape(-1, 3)
    if points2.shape[1] != 3:
        points2 = points2.reshape(-1, 3)
    pts1.points = o3d.utility.Vector3dVector(points1)
    pts2.points = o3d.utility.Vector3dVector(points2)
    if image1.shape[0] == 1:
        image1 = image1[0]
    if image2.shape[0] == 1:
        image2 = image2[0]
    # image should have the shape of HxWx3, then to (-1,3)
    pts1.colors = o3d.utility.Vector3dVector(image1)
    pts2.colors = o3d.utility.Vector3dVector(image2)
    o3d.visualization.draw_geometries([pts1, pts2])

def draw_image_matching(f11, f21, image1, image2, n_viz=25, hw_order=True):
    """using the plt to draw the matching pixels
    f11: (N,2) pixel coordinate in image1
    f21: (N,2) pixel coordinate in image2
    image1: (1,3,H,W) tensor
    image2: (1,3,H,W) tensor
    """
    num_matches = f11.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = f11[match_idx_to_viz], f21[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs=[]
    for img in [image1, image2]:
        rgb_tensor = img * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    H0, W0, H1, W1 = viz_imgs[0].shape[0], viz_imgs[0].shape[1], viz_imgs[1].shape[0], viz_imgs[1].shape[1]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        if hw_order:
            (x0, y0), (x1, y1) = viz_matches_im0[i][::-1], viz_matches_im1[i][::-1] # if (hw) sequence
        else:
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T # if (w,h) sequence
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)