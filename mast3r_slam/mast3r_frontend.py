##
# @ Desc:
#   The frontend module of the Mast3r-based SLAM
#   The frontend will receive the keyframe from the motion filter, and then do the local bundle adjustment based on local keyframes.
#   The frame information will be stored in the Mast3rVideo object.
# @ Author: Shuo Sun
##

from dust3r.utils.image import load_images, format_images, format_mast3r_out
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import paris_asymmetric_inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

class Mast3rFrontend:
    def __init__(self, mast3r_model, video):
        self.device='cuda:0'
        self.mast3r_model = mast3r_model
        self.video = video

        self.count = 0
        self.warmup = 8
        self.ba_window = 8

        # local ba setting
        self.niter = 200
        self.lr = 0.01
        self.schedule = 'cosine'
        self.batch_size = 1

        self.is_initialized = False

    def __initialize__(self):
        """ Initialize the frontend; After the initialization, we will fix the intrinsic for following tracking; also, the scale. """
        initial_images = self.video.images[:self.ba_window]
        images = format_images(initial_images)
        pairs = make_pairs(images, scene_graph='complete', prefilter='seq3', symmetrize=False)
        out = paris_asymmetric_inference(pairs, self.mast3r_model, self.device)
        res = format_mast3r_out(pairs, out)
        scene = global_aligner(res, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr)

        self.is_initialized = True

    def __track__(self):
        """ Video local Mast3r BA except the initialization; Use the initialized intrinsic """
        pass

    def __call__(self):
        """ The main thread of the frontend """
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize__()
        elif self.is_initialized and self.video.counter.value % self.ba_window == 0:
            self.__track__()
        else:
            pass

