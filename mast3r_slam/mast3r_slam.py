import torch
import lietorch
import numpy as np

import sys
sys.path.append('../droid_slam')
# sys.path.append('sea_raft')

from collections import OrderedDict
from torch.multiprocessing import Process

from mast3r.model import AsymmetricMASt3R
from mast3r_video import Mast3rVideo
from mast3r_motion_filter import Mast3rMotionFilter
from mast3r_frontend import Mast3rFrontend

# DROID_SLAM
from droid_net import DroidNet
import os
from lietorch import SE3
import droid_backends

# SEA-RAFT
from sea_raft.raft import RAFT
from sea_raft.utils.utils import load_ckpt

class Mast3rSlam:
    def __init__(self, args):
        super(Mast3rSlam, self).__init__()
        self.load_weights(args.mast3r_weights, args.droid_weights, args.sea_rafts_weights)
        self.args = args

        self.video = Mast3rVideo(args.image_size, args.buffer)

        self.filterx = Mast3rMotionFilter(self.droid_net, self.sea_raft, self.video, thresh=args.filter_thresh)

        self.mast3r_frontend = Mast3rFrontend(self.mast3r_net, self.video)

    def load_weights(self, mast3r_weights, droid_weights=None, sea_rafts=None):
        """
        load trained model weights,
        """
        self.mast3r_net = AsymmetricMASt3R.from_pretrained(mast3r_weights).to('cuda').eval()

        self.droid_net = None
        self.sea_raft = None
        if droid_weights is not None: # for optical flow estimation
            self.droid_net = DroidNet()
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in torch.load(droid_weights).items()])

            state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
            state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
            state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
            state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

            self.droid_net.load_state_dict(state_dict)
            self.droid_net.to("cuda:0").eval()
        if sea_rafts is not None:
            # load the sea-raft weights
            self.sea_raft = RAFT()
            load_ckpt(self.sea_raft, sea_rafts)
            self.sea_raft = self.sea_raft.to('cuda:0')
            self.sea_raft.eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ The Mast3r SLAM main thread """

        # the optical flow estimation to select the keyframe
        track = self.filterx.track(tstamp, image, depth, intrinsics)

        # do the local BA for local reconstruction
        if track:
            self.mast3r_frontend()

    def terminate(self, stream=None):
        pass

    def save_trajectory(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        # save the trajectory.
        counter = self.video.counter.value

        poses = self.video.poses[:counter].detach().cpu().numpy() # camera2world

        depths = self.video.disps[:counter].detach().cpu().numpy() # note that the inverse depths

        intrinsics = self.video.intrinsics[:counter].detach().cpu().numpy()

        np.savez(os.path.join(path, 'mast3r_trajectory.npz'), poses=poses, depths=depths, intrinsics=intrinsics)
