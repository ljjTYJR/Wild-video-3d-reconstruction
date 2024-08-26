# The helper functions for Dust3R implementation
import PIL
import numpy as np
import torchvision.transforms as tvf
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import torch

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def set_as_dust3r_image(image, idx, size=512):
    """Set the image as Dust3R image format;

    Args:
        image (torch.Tensor): 3XHXW image tensor.
    Returns:
        The Dust3R image format.
        1. image; [1 x 3 x H x W]
        2. image shape [H x W]
        3. index;
        4. instance
    """
    H,W = image.shape[1:]
    if W != size:
        image = _resize_pil_image(image, size)
    image = ImgNorm(image.permute(1,2,0).cpu().numpy())[None]
    return dict(
        img=image,
        true_shape=np.int32([image.shape[-2:]]),
        idx=idx,
        instance=str(idx)
    )


def dust3r_inference(images, model, device='cuda', batch_size=1, niter=300, schedule='cosine', lr=0.01):
    """Inference on the input images buffer using the Dust3R model

    """
    # make pairs
    pairs=make_pairs(images, scene_graph='complete', prefilter='seq3', symmetrize=True)
    # model prediction
    output=inference(pairs, model, device, batch_size=batch_size)
    # 3D BA for the alignment
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, same_focal=True)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    # scene.show()
    depths = torch.stack(scene.get_depthmaps()).detach()
    focal=scene.get_focals().cpu()[0] # If using the same focal, then [0] represents all focal
    return focal, depths