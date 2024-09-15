# The video management for the mast3r-based SLAM
import torch

from torch.multiprocessing import Process, Queue, Lock, Value

class Mast3rVideo:
    def __init__(self, image_size, buffer=512, devide='cuda:0'):
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float)
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8) # to cuda?
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float)
        # different from the DROID which down-sample with the factor of 8, we use the original size
        self.disps = torch.ones(buffer, ht, wd, device="cuda", dtype=torch.float)
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float)

        ### feature attributes ###
        c = 1
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda")
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda")
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda")

        # attributes of the video
        self.counter = Value('i', 0)

        # the bundle adjustment setting
        self.ba_window = 8 # After collecting 8 frames, do BA

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # append the items
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            self.intrinsics[index] = item[4]

    def __setitem__(self, index, item):
        self.__item_setter(index, item)

    def __getitem__(self, index):
        if isinstance(index, int) and index < 0:
            index = self.counter.value + index

        item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                )
        return item

    def append(self, *item):
        self.__item_setter(self.counter.value, item)