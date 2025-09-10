import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .lietorch import SE3

all_times = []

class Timer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()

    def __exit__(self, type, value, traceback):
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end)
            all_times.append(elapsed)
            print(self.name, elapsed)


def coords_grid(b, n, h, w, **kwargs):
    """ coordinate grid """
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)
    coords = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    return coords[[1,0]].view(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)

def coords_grid_with_index(d, **kwargs):
    """ coordinate grid with frame index"""
    b, n, h, w = d.shape
    i = torch.ones_like(d)
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords = torch.stack([x, y, d], dim=2)
    index = torch.arange(0, n, dtype=torch.float, **kwargs)
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index

def patchify(x, patch_size=3):
    """ extract patches from video """
    b, n, c, h, w = x.shape
    x = x.view(b*n, c, h, w)
    y = F.unfold(x, patch_size)
    y = y.transpose(1,2)
    return y.reshape(b, -1, c, patch_size, patch_size)


def pyramidify(fmap, lvls=[1]):
    """ turn fmap into a pyramid """
    b, n, c, h, w = fmap.shape

    pyramid = []
    for lvl in lvls:
        gmap =  F.avg_pool2d(fmap.view(b*n, c, h, w), lvl, stride=lvl)
        pyramid += [ gmap.view(b, n, c, h//lvl, w//lvl) ]

    return pyramid

def all_pairs_exclusive(n, **kwargs):
    ii, jj = torch.meshgrid(torch.arange(n, **kwargs), torch.arange(n, **kwargs))
    k = ii != jj
    return ii[k].reshape(-1), jj[k].reshape(-1)

def set_depth(patches, depth):
    patches[...,2,:,:] = depth[...,None,None]
    return patches

def flatmeshgrid(*args, **kwargs):
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def evaluate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian

def measure_motion(img1, img2, threshold=1.0):
    # note: the image is now BGR mode, which is the default for OpenCV
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Compute optical flow (Farnebäck example)
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.linalg.norm(flow, axis=2)

    mean_flow = np.mean(magnitude)
    median_flow = np.median(magnitude)

    # Normalized values
    h, w = img1.shape
    # measure under 512 resolution
    scale = 512.0 / max(h, w)
    mean_flow_scaled = mean_flow * scale
    diag = np.sqrt(h**2 + w**2)
    mean_norm = mean_flow / diag
    median_norm = median_flow / diag

    # Motion ratio
    motion_ratio = np.mean(magnitude > threshold)

    return mean_flow, mean_flow_scaled, median_flow, mean_norm, median_norm, motion_ratio
