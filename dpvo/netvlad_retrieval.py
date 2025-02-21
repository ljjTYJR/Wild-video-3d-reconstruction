# Image retrieval using netvlad

import os
import shutil
import sys
import torch
import numpy as np
sys.path.append('../')
# from hloc import extract_features, pairs_from_retrieval
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import read_image
from hloc import extractors
from itertools import chain
from pathlib import Path
import tqdm

netvlad_confs={
    "output": "global-feats-netvlad",
    "model": {"name": "netvlad"},
    "preprocessing": {"resize_max": 1024},
}

class RetrievalNetVLAD:
    def __init__(self, buffer_size=1000, top_k=20):
        Model = dynamic_load(extractors, netvlad_confs["model"]["name"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.as_half = False
        self.netvlad_model=Model(netvlad_confs["model"]).eval().to(self.device)

        self.netvlad_db = torch.zeros((buffer_size, 4096), dtype=torch.float16 if self.as_half else torch.float32) # The length of the descriptor is 4096
        self.img_buffer = {}
        self.TOPK=top_k

    @torch.no_grad()
    def insert_img(self, idx, image):
        """
        Insert image into the netvlad database.
        image: np.array of shape (3, H, W); range [0, 1]
        """
        if image.ndim == 3:
            image = image.unsqueeze(0) # (1, 3, H, W)
        pred = self.netvlad_model({"image": image.to(self.device, non_blocking=True)})
        # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        if self.as_half:
            pred['global_descriptor'] = pred['global_descriptor'].to(torch.float16)
        self.netvlad_db[idx] = pred['global_descriptor']


    @torch.no_grad()
    def query(self, idx, skip_window=-1):
        query_desc = self.netvlad_db[idx]
        # query indices outside the local bundle adjustment window
        if skip_window < 0:
            skip_window = idx
        potential_indices=list(range(0, skip_window))
        sim = torch.matmul(self.netvlad_db[potential_indices], query_desc.unsqueeze(0).T)
        if len(sim) < self.TOPK:
            val, indices = torch.topk(sim, len(sim), dim=0)
        else:
            val, indices = torch.topk(sim, self.TOPK, dim=0)
        return val, indices


class RetrievalNetVLADOffline(RetrievalNetVLAD):
    def __init__(self, img_dir, skip, end, stride):
        self.img_dir = img_dir

        self.skip = skip
        self.end = end
        self.stride = stride

        img_exts = ["*.png", "*.jpeg", "*.jpg"]
        self.image_list = sorted(chain.from_iterable(Path(self.img_dir).glob(e) for e in img_exts))[self.skip:self.end:self.stride]
        self.netvlad_db_online = torch.zeros((len(self.image_list), 4096), dtype=torch.float32).contiguous()

        super().__init__(len(self.image_list))

    @property
    def nvlad_db(self):
        return self.netvlad_db
    @property
    def nvlad_db_online(self):
        return self.netvlad_db_online

    def query_online(self, idx, skip_window=-1, top_k=10):
        query_desc = self.netvlad_db_online[idx]
        if idx <= skip_window:
            return None, None
        if skip_window < 0:
            skip_window = idx
        q_0 = 0
        q_1 = idx - skip_window
        sim = torch.matmul(self.netvlad_db_online[q_0:q_1], query_desc.unsqueeze(0).T)
        if len(sim) < top_k:
            val, indices = torch.topk(sim, len(sim), dim=0)
        else:
            val, indices = torch.topk(sim, top_k, dim=0)
        return val, indices

    def insert_img_offline(self):
        # use tqdm to show the progress
        for i in tqdm.tqdm(range(len(self.image_list))):
            image = read_image(self.image_list[i]) # RGB format
            image = image.astype(np.float32).transpose((2, 0, 1)) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
            self.insert_img(i, image)

    def insert_desc(self, idx, desc):
        self.netvlad_db_online[idx] = desc

    def get_desc(self, idx):
        return self.netvlad_db_online[idx]

    def end_and_clean(self):
        self.netvlad_model = None
        torch.cuda.empty_cache()

    @classmethod
    def from_instance(cls, instance):
        if not isinstance(instance, cls):
            raise TypeError("Instance must be of type MyClass")
        return cls(instance.img_dir, instance.skip, instance.end, instance.stride)