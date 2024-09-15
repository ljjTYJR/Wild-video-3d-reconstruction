import cv2
import torch
import lietorch

import sys
sys.path.append('../droid_slam')

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock

# SEA-RAFT related optical flow estimation
def sea_raft_optical_flow(image1, image2, sea_raft):
    # image from BGR to RGB
    image1 = image1[[2, 1, 0], :, :][None]
    image2 = image2[[2, 1, 0], :, :][None]
    if image1.device != image2.device:
        image1 = image1.to('cuda')
        image2 = image2.to('cuda')
    output = sea_raft(image1, image2, iters=4, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

class Mast3rMotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, sea_raft, video, thresh=2.5, mast3r_pred=False, device="cuda:0"):

        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        # whether to use the mast3r prediction
        self.mast3r_pred = mast3r_pred
        self.seq_raft = sea_raft

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            # The initialization, no need to add
            self.video.append(tstamp, image[0], Id, 1.0, intrinsics / 8.0)

        ### only add new frame if there is enough motion ###
        else:
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr) # I think it just compare the same pixel's flow, to remove static frames

            # sea-raft optical flow
            if self.seq_raft is not None:
                sea_raft_flow, _ = sea_raft_optical_flow(image[0], self.video.images[-1], self.seq_raft)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                # for the following frames, the intrinsic will be set as the first frame
                intrinsics = self.video.intrinsics[0].clone()
                self.video.append(tstamp, image[0], None, None, intrinsics)
            else:
                self.count += 1
