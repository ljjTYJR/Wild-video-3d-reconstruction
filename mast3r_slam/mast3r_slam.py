import torch
import lietorch
import numpy as np

import sys
sys.path.append('../droid_slam')

from collections import OrderedDict
from torch.multiprocessing import Process

from mast3r.model import AsymmetricMASt3R
from mast3r_video import Mast3rVideo
from mast3r_motion_filter import Mast3rMotionFilter

from droid_net import DroidNet

class Mast3rSlam:
    def __init__(self, args):
        super(Mast3rSlam, self).__init__()
        self.load_weights(args.mast3r_weights, args.droid_weights)
        self.args = args

        self.video = Mast3rVideo(args.image_size, args.buffer)

        self.filterx = Mast3rMotionFilter(self.droid_net, self.video, thresh=args.filter_thresh)

    def load_weights(self, mast3r_weights, droid_weights=None):
        """
        load trained model weights,
        """
        self.mast3r_net = AsymmetricMASt3R.from_pretrained(mast3r_weights).to('cuda').eval()

        self.droid_net = None
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

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ The Mast3r SLAM main thread """

        # the optical flow estimation to select the keyframe
        self.filterx.track(tstamp, image, depth, intrinsics)

        # do the local BA for local reconstruction
        # self.frontend()


    def terminate(self, stream=None):
        pass
