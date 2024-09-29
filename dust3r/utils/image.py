# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def _resize_pil_image_by_scale(img, scale=None):
    """ Resize the image by a scale factor of the original size."""
    interp = PIL.Image.LANCZOS # a kind of downsampling filter
    new_size = tuple(int(round(x * scale)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True, scale=None):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if scale is None:
            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W//2, H//2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if not (square_ok) and W == H:
                    halfh = 3*halfw/4
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        else:
            # use the scale to resize the image
            img = _resize_pil_image_by_scale(img, scale)

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)))) # there is 'true shape'

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def format_images(images):
    """ Convert the images array to the dust3r-needed format; By default, the images should be reszie to 512 """
    # to RGB order, to correspond to the code `img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')`
    if isinstance(images, list):
        images = torch.stack(images, dim=0)
    images = images[:, [2, 1, 0], :, :]
    imgs = []
    for i, image in enumerate(images):
        img = ImgNorm(image.permute(1,2,0).cpu().numpy())[None]
        imgs.append(dict(img=img, true_shape=np.int32([img.shape[2:]]), idx=i, instance=str(i)))
    return imgs

def format_mast3r_out(pairs, out):
    """ format the direct output prediction from the mast3r model for BundelAdjustment usage """
    res = dict()
    # view1 are the meta data from the first view; tensor/tensor/list/list
    res['view1'] = {
        'img': torch.cat([x[0]['img'] for x in pairs], dim=0),
        'true_shape': torch.cat([torch.from_numpy(x[0]['true_shape']) for x in pairs], dim=0),
        'idx' : [x[0]['idx'] for x in pairs],
        'instance' : [x[0]['instance'] for x in pairs]
    }
    res['view2'] = {
        'img': torch.cat([x[1]['img'] for x in pairs], dim=0),
        'true_shape': torch.cat([torch.from_numpy(x[1]['true_shape']) for x in pairs], dim=0),
        'idx' : [x[1]['idx'] for x in pairs],
        'instance' : [x[1]['instance'] for x in pairs]
    }
    res['pred1'] = {
        'pts3d': torch.cat([val[0]['pts3d'] for key, val in out.items()], dim=0),
        'conf': torch.cat([val[0]['conf'] for key, val in out.items()], dim=0),
    }
    res['pred2'] = {
        'pts3d_in_other_view': torch.cat([val[1]['pts3d'] for key, val in out.items()], dim=0),
        'conf': torch.cat([val[1]['conf'] for key, val in out.items()], dim=0),
    }
    return res