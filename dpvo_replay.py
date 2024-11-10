import pickle
import tyro
import torch
import random
import os
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue
from evo.core.trajectory import PoseTrajectory3D

from dpvo.dpvo import DPVO
from dpvo.plot_utils import save_output_for_COLMAP
from dpvo.stream import image_stream, video_stream, image_stream_limit

from mast3r.model import AsymmetricMASt3R
from mast3r.inference import local_ba_flexible

def fetch_dpvo_data(slam: DPVO):
    points, colors, (intrinsic, H, W) = slam.get_pts_clr_intri(inlier=True)
    return slam.terminate(), (points, colors, (*intrinsic, H, W))

def seed_all(seed=0):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(
    path: str,
    imagedir: str,
    calib: str,
    export_colmap: bool = True,
) -> None:
    saved_model = Path(path) / 'dpvo.pkl'
    slam = None
    queue = Queue(maxsize=8)
    init_intrinsic = None
    stride = 1
    skip = 0

    with open(saved_model, 'rb') as f:
        slam = pickle.load(f)
        f.close()
    assert isinstance(slam, DPVO)

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))
    reader.start()

    frames = np.array(list(slam.inlier_ratio_record.keys()))
    n = 0
    while 1:
        (t, image, intrinsics) = queue.get()
        if init_intrinsic is not None:
            intrinsics = init_intrinsic
        if t < 0: break
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if t == frames[n]:
            print(f"Processing frame {t}")
            inlier_ratio = slam.inlier_ratio_record[t]

            # buffer the input data
            slam.image_buffer_[n % slam.mem] = image

            if inlier_ratio < 0.8:
                # initialize the optimization window and raw data, the `n` is just current `n`
                print(f"frame{t} inlier ratio {inlier_ratio} < pre-set threshold")
                if slam.mast3r_model == None:
                    slam.mast3r_est = True
                    slam.mast3r_model = AsymmetricMASt3R.from_pretrained('checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth').to('cuda').eval()

                # use images to previous N (currently, set 10), estimate 2-2 correspondences
                N_buffer = 10
                img_buffer = []
                for i in range(1, N_buffer+1):
                    img_buffer.append(slam.image_buffer_[(n-N_buffer+i) % slam.mem]) # BGR format
                scene = local_ba_flexible(img_buffer, slam.mast3r_model)
            n += 1

    # (poses, tstamps), (points, colors, calib) = fetch_dpvo_data(slam)
    # trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)
    # if export_colmap:
    #     save_output_for_COLMAP(path, tstamps, trajectory, points, colors, True, *calib)

if __name__ == "__main__":
    seed_all()
    tyro.cli(main)