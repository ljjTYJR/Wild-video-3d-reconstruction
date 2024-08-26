# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Fast Nearest Neighbor
# --------------------------------------------------------
import torch
import numpy as np
import math
from scipy.spatial import KDTree

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy, todevice  # noqa


@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, device='cuda', block_size=None, dist='l2'):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim) # the dor-product is large, the cosine similarity is small
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    if block_size is None or len(A) * len(B) <= block_size**2: # this part can be re-written in the cuda?
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float('inf'), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size) # divide the data into many blocks for processing

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size:(i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size:(j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)  # A, B, 1
                # dists_blk = dists[i * block_size:(i+1)*block_size, j * block_size:(j+1)*block_size]
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1) # for every point in A, find the nearest point in B, second is the index
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0) # for every point in B, find the nearest point in A

                col_mask = min_A_i < dis_A[i * block_size:(i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size:(j + 1) * block_size]

                dis_A[i * block_size:(i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size:(j + 1) * block_size][line_mask] = min_B_j[line_mask] # update the new distance

                nn_A[i * block_size:(i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (j * block_size) # update the index for nearest neighbor
                nn_B[j * block_size:(j + 1) * block_size][line_mask] = argmin_B_j[line_mask] + (i * block_size)
    nn_A = nn_A.cpu().numpy() # now, we can get the nearest neighbor, index for A and B
    nn_B = nn_B.cpu().numpy()
    return nn_A, nn_B


class cdistMatcher:
    def __init__(self, db_pts, device='cuda'):
        self.db_pts = db_pts.to(device)
        self.device = device

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = bruteforce_reciprocal_nns(queries, self.db_pts, device=self.device, **kw) # queries: queried features
        dis = None
        return dis, nnA # nnA is the corresponding index in image B.


def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx1.dtype == idx2.dtype == np.int32

    # unique and sort along idx1 # Very nice code!
    corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index) # might convergence to the same point; combine them together is to avoid the same point.
    if ret_index:
        corres, indices = corres
    xy2, xy1 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy1 = np.unravel_index(xy1, shape1)
        xy2 = np.unravel_index(xy2, shape2)
        if ret_xy != 'y_x':
            xy1 = xy1[0].base[:, ::-1] # (x,y)
            xy2 = xy2[0].base[:, ::-1]
        else:
            xy1 = xy1[0].base #(yx) follow (h,w)
            xy2 = xy2[0].base

    if ret_index:
        return xy1, xy2, indices
    return xy1, xy2

def fast_reciprocal_NNs_debug(pts1, pts2, view1=None, view2=None, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False,
                        device='cuda', **matcher_kw):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
        S = subsample_or_initxy1
        y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1) # Not understanding this downsampling, maybe border has some problems; or this represents the center of a grid?
        max_iter = 10
    else:
        x1, y1 = subsample_or_initxy1
        if isinstance(x1, torch.Tensor):
            x1 = x1.cpu().numpy()
        if isinstance(y1, torch.Tensor):
            y1 = y1.cpu().numpy()
        max_iter = 1

    xy1 = np.int32(np.unique(x1 + W1 * y1))  # make sure there's no doublons
    xy2 = np.full_like(xy1, -1)
    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    if (isinstance(device, str) and device.startswith('cuda')) or (isinstance(device, torch.device) and device.type.startswith('cuda')):
        pts1 = pts1.to(device)
        pts2 = pts2.to(device)
        tree1 = cdistMatcher(pts1, device=device)
        tree2 = cdistMatcher(pts2, device=device)
    else:
        pts1, pts2 = to_numpy((pts1, pts2))
        tree1 = KDTree(pts1)
        tree2 = KDTree(pts2)

    notyet = np.ones(len(xy1), dtype=bool)
    basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

    niter = 0
    # n_notyet = [len(notyet)]
    while notyet.any():
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw)) # xy2 is the index corresponding in image2 to xy1
        if not ret_basin:
            notyet &= (old_xy2 != xy2)  # remove points that have converged

        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw)) # xy2 is the returned indices of matching from pts1
        if ret_basin:
            basin[old_xy1[notyet]] = xy1[notyet]
        notyet &= (old_xy1 != xy1)  # remove points that have converged

        # n_notyet.append(notyet.sum())
        niter += 1
        if niter >= max_iter:
            break

        old_xy2[:] = xy2
        old_xy1[:] = xy1
        # DEBUG: visualize the correspondence in the image; xy1 and xy2 are a pair of correspondence now.
        if view1 is not None and view1 is not None:
            matches_im0, matches_im1 = merge_corres(xy1, xy2, (H1, W1), (H2, W2), ret_xy=ret_xy) # view all matches
            # ignore small border around the edge
            H0, W0 = view1['true_shape'][0]
            valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

            H1, W1 = view2['true_shape'][0]
            valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

            valid_matches = valid_matches_im0 & valid_matches_im1
            matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

            # visualize a few matches
            import torchvision.transforms.functional
            from matplotlib import pyplot as pl

            n_viz = 20
            num_matches = matches_im0.shape[0]
            match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
            viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

            image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
            image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

            viz_imgs = []
            for i, view in enumerate([view1, view2]):
                rgb_tensor = view['img'] * image_std + image_mean
                viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

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

    # print('notyet_stats:', ' '.join(map(str, (n_notyet+[0]*10)[:max_iter])))

    if pixel_tol > 0:
        # in case we only want to match some specific points
        # and still have some way of checking reciprocity
        old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
        new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
        dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
        converged = dis < pixel_tol
        if not isinstance(subsample_or_initxy1, int):
            xy1 = old_xy1  # replace new points by old ones
    else:
        converged = ~notyet  # converged correspondences

    # keep only unique correspondences, and sort on xy1
    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2

def fast_reciprocal_NNs_sample(pts1, pts2, sample1, max_iter=10, ret_xy=True, pixel_tol=0, ret_basin=False, device='cuda', **matcher_kw):
    """similar to fast_reciprocal_NNs, but given the sample pixels of pts1
    samples1: (N,), the indices of the sample pixels in pts1
    """
    H1, W1, DIM1 = pts1.shape # for features, H,W,24
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    xy1 = np.int32(sample1.numpy()) # the sample pixels
    xy2 = np.full_like(xy1, -1) # the corresponding index in image2
    old_xy1 = xy1.copy() # for 1-D array, copy is deep copy
    old_xy2 = xy2.copy()

    # build the NN search object
    pts1 = pts1.to(device)
    pts2 = pts2.to(device)
    tree1 = cdistMatcher(pts1, device=device)
    tree2 = cdistMatcher(pts2, device=device)

    notyet = np.ones(len(xy1), dtype=bool) # record left pixels not stable already
    basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None
    niter = 0
    while notyet.any():
        # only query those unstable points and only update unstable points in xy2
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
        if not ret_basin:
            notyet &= (old_xy2 != xy2)
        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
        if ret_basin:
            basin[old_xy1[notyet]] = xy1[notyet]
        notyet &= (old_xy1 != xy1)

        niter += 1
        if niter >= max_iter:
            break
        old_xy1[:] = xy1
        old_xy2[:] = xy2

    xy1, xy2 = merge_corres(xy1[~notyet], xy2[~notyet], (H1, W1), (H2, W2), ret_xy=ret_xy)
    return xy1, xy2

def fast_simple_forward_search(query, fea0, fea1, device='cuda', valid_mask=False, **matcher_kw):
    """simple forward search for the nearest neighbor
    also, output the valid mask which represents the recipal nearest neighbor
    """
    H,W,D = fea0.shape
    query = query.to(device)
    fea0 = fea0.reshape(-1, D).to(device)
    fea1 = fea1.reshape(-1, D).to(device)
    tree = cdistMatcher(fea1, device=device)
    _, nn = to_numpy(tree.query(query, **matcher_kw))
    if valid_mask:
        # print the valid mask by recipal nearest neighbor verification
        tree_2 = cdistMatcher(fea0, device=device)
        query_2 = fea1[nn]
        _, nn_2 = to_numpy(tree_2.query(query_2, **matcher_kw))
        return nn, nn_2
    return nn

def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False,
                        device='cuda', **matcher_kw):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
        S = subsample_or_initxy1
        y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1) # Not understanding this downsampling, maybe border has some problems; or this represents the center of a grid?
        max_iter = 10
    else:
        x1, y1 = subsample_or_initxy1
        if isinstance(x1, torch.Tensor):
            x1 = x1.cpu().numpy()
        if isinstance(y1, torch.Tensor):
            y1 = y1.cpu().numpy()
        max_iter = 1

    xy1 = np.int32(np.unique(x1 + W1 * y1))  # make sure there's no doublons, the index
    xy2 = np.full_like(xy1, -1)
    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    if (isinstance(device, str) and device.startswith('cuda')) or (isinstance(device, torch.device) and device.type.startswith('cuda')):
        pts1 = pts1.to(device)
        pts2 = pts2.to(device)
        tree1 = cdistMatcher(pts1, device=device)
        tree2 = cdistMatcher(pts2, device=device)
    else:
        pts1, pts2 = to_numpy((pts1, pts2))
        tree1 = KDTree(pts1)
        tree2 = KDTree(pts2)

    notyet = np.ones(len(xy1), dtype=bool)
    basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

    niter = 0
    # n_notyet = [len(notyet)]
    while notyet.any():
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw)) # xy2 is the index corresponding in image2 to xy1
        if not ret_basin:
            notyet &= (old_xy2 != xy2)  # remove points that have converged

        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw)) # xy2 is the returned indices of matching from pts1
        if ret_basin:
            basin[old_xy1[notyet]] = xy1[notyet]
        notyet &= (old_xy1 != xy1)  # remove points that have converged

        # n_notyet.append(notyet.sum())
        niter += 1
        if niter >= max_iter:
            break

        old_xy2[:] = xy2
        old_xy1[:] = xy1

    # print('notyet_stats:', ' '.join(map(str, (n_notyet+[0]*10)[:max_iter])))

    if pixel_tol > 0:
        # in case we only want to match some specific points
        # and still have some way of checking reciprocity
        old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
        new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
        dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
        converged = dis < pixel_tol
        if not isinstance(subsample_or_initxy1, int):
            xy1 = old_xy1  # replace new points by old ones
    else:
        converged = ~notyet  # converged correspondences

    # keep only unique correspondences, and sort on xy1
    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2


def extract_correspondences_nonsym(A, B, confA, confB, subsample=8, device=None, ptmap_key='pred_desc', pixel_tol=0):
    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    # merge corres from opposite pairs
    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]
    if pixel_tol == 0:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)
    else:
        S = subsample
        yA, xA = np.mgrid[S // 2:HA:S, S // 2:WA:S].reshape(2, -1)
        yB, xB = np.mgrid[S // 2:HB:S, S // 2:WB:S].reshape(2, -1)

        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=(xA, yA), ret_xy=False, pixel_tol=pixel_tol, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=(xB, yB), ret_xy=False, pixel_tol=pixel_tol, **opt)

    idx1 = np.r_[nn1to2[0], nn2to1[1]]
    idx2 = np.r_[nn1to2[1], nn2to1[0]]

    c1 = confA.ravel()[idx1]
    c2 = confB.ravel()[idx2]

    xy1, xy2, idx = merge_corres(idx1, idx2, (HA, WA), (HB, WB), ret_xy=True, ret_index=True)
    conf = np.minimum(c1[idx], c2[idx])
    corres = (xy1.copy(), xy2.copy(), conf)
    return todevice(corres, device)
