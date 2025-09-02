import atexit
import glob
import os
import signal
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch

# Set multiprocessing start method to avoid CUDA context issues
torch.multiprocessing.set_start_method('spawn', force=True)
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from tqdm import tqdm

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.dpvo_colmap_init import run_colmap_initialization
from dpvo.netvlad_retrieval import RetrievalNetVLADOffline
from dpvo.plot_utils import plot_trajectory
from dpvo.stream import image_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, viz=False, show_img=False, rerun=False, colmap_init=False):

    slam = None

    retrieval = None
    if cfg.loop_enabled:
        print("Extracting global descriptors...")
        retrieval = RetrievalNetVLADOffline(Path(imagedir), stride=stride)
        retrieval.insert_img_offline()
        retrieval.end_and_clean()

    if colmap_init:
        est_intrinsic = run_colmap_initialization(Path(imagedir), Path(imagedir).parent, skip=SKIP)
        calib = est_intrinsic

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, imagedir, None, None, calib, stride, 0))
    reader.start()

    with tqdm(desc=f"Processing {imagedir}", unit="frames") as pbar:
        while 1:
            (t, image, depth, mask, intrinsics) = queue.get()
            if t < 0: break

            image = torch.from_numpy(image).permute(2,0,1).cuda()
            intrinsics = torch.from_numpy(intrinsics).cuda()

            if show_img:
                show_image(image, 1)

            if slam is None:
                slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz, nvlad_db=retrieval, rerun=rerun)

            with Timer("SLAM", enabled=False):
                slam(t, image, depth, mask, intrinsics)

            pbar.update(1)

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    # Add cleanup handlers to prevent Arrow-related SIGTERM errors
    def cleanup_handler(*_):
        """Clean shutdown handler to prevent Arrow cleanup issues"""
        print("\nPerforming clean shutdown...")
        try:
            # Force garbage collection
            import gc
            gc.collect()
            # Clear CUDA cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Clean exit without triggering Arrow cleanup
            os._exit(0)
        except:
            os._exit(0)

    def signal_handler(sig, _):
        """Handle interrupt signals gracefully"""
        print(f"\nReceived signal {sig}, shutting down...")
        cleanup_handler()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register cleanup function for normal exit
    atexit.register(cleanup_handler)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='checkpoints/dpvo.pth')
    parser.add_argument('--config', default="dpvo_configs/default.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--eurocdir', default="datasets/EUROC")
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--rerun', action="store_true")
    parser.add_argument('--loop_enabled', action="store_true")
    parser.add_argument('--colmap_init', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.loop_enabled = args.loop_enabled
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(412)
    # Set CUDA memory management settings to prevent segfaults
    if torch.cuda.is_available():
        # Enable memory cleanup on exit
        torch.cuda.memory._record_memory_history(enabled=None)
        # Set conservative memory management
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    euroc_scenes = [
        # "MH_01_easy",
        # "MH_02_easy",
        "MH_03_medium",
        "MH_04_difficult",
        "MH_05_difficult",
        # "V1_01_easy",
        "V1_02_medium",
        "V1_03_difficult",
        # "V2_01_easy",
        "V2_02_medium",
        "V2_03_difficult",
    ]

    try:
        results = {}
        for scene in euroc_scenes:
            imagedir = os.path.join(args.eurocdir, scene, "mav0/cam0/data")
            groundtruth = "euroc_groundtruth/{}.txt".format(scene)

            scene_results = []
            for i in range(args.trials):
                traj_est, timestamps = run(cfg, args.network, imagedir, "calib/euroc.txt", args.stride, args.viz, args.show_img,
                                           args.rerun, args.colmap_init)

                images_list = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::args.stride]
                tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

                traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est[:,:3],
                    orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                    timestamps=np.array(tstamps))

                traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
                traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

                result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
                ate_score = result.stats["rmse"]

                if args.plot:
                    scene_name = '_'.join(scene.split('/')[1:]).title()
                    Path("trajectory_plots").mkdir(exist_ok=True)
                    plot_trajectory(traj_est, traj_ref, f"Euroc {scene} Trial #{i+1} (ATE: {ate_score:.03f})",
                                    f"trajectory_plots/Euroc_{scene}_Trial{i+1:02d}.pdf", align=True, correct_scale=True)

                if args.save_trajectory:
                    Path("saved_trajectories").mkdir(exist_ok=True)
                    file_interface.write_tum_trajectory_file(f"saved_trajectories/Euroc_{scene}_Trial{i+1:02d}.txt", traj_est)

                scene_results.append(ate_score)

            scene_mean = np.mean(scene_results)
            scene_std = np.std(scene_results)
            results[scene] = {'mean': scene_mean, 'std': scene_std}
            print(f"{scene}: mean={scene_mean:.4f}, std={scene_std:.4f}, values={sorted(scene_results)}")

        means = []
        stds = []
        for scene in results:
            scene_data = results[scene]
            print(f"{scene}: mean={scene_data['mean']:.4f}, std={scene_data['std']:.4f}")
            means.append(scene_data['mean'])
            stds.append(scene_data['std'])

        overall_mean = np.mean(means)
        overall_std = np.mean(stds)
        print(f"OVERALL: mean={overall_mean:.4f}, avg_std={overall_std:.4f}")
        print("EuRoC evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean exit to avoid Arrow cleanup issues
        import gc
        gc.collect()
        print("Exiting cleanly...")
        os._exit(0)
