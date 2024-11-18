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

netvlad_confs={
    "output": "global-feats-netvlad",
    "model": {"name": "netvlad"},
    "preprocessing": {"resize_max": 1024},
}

class RetrievalNetVLAD:
    def __init__(self, buffer_size=1000):
        Model = dynamic_load(extractors, netvlad_confs["model"]["name"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.as_half = False
        self.netvlad_model=Model(netvlad_confs["model"]).eval().to(self.device)

        self.netvlad_db = torch.zeros((buffer_size, 4096), dtype=torch.float16 if self.as_half else torch.float32)
        self.img_buffer = {}
        self.TOPK=20
        self.RETRIEVAL_WINDOW=10

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
    def extract_feature(self):
        # process each image and then return the reference index
        # for idx, img in enumerate(sorted(os.listdir(IMG_DIR))):
        #     self.img_buffer[idx] = img
        #     img_path = os.path.join(IMG_DIR, img)
        #     self.insert_img(idx, img_path)
        #     query_indices = self.query(idx)
        return

    @torch.no_grad()
    def query(self, idx):
        query_desc = self.netvlad_db[idx]
        # indices = []
        # potential_indices = list(range(idx-self.RETRIEVAL_WINDOW))
        potential_indices=list(range(idx)) # ignore the frame 0
        sim = torch.matmul(self.netvlad_db[potential_indices], query_desc.unsqueeze(0).T)
        if len(sim) < self.TOPK:
            val, indices = torch.topk(sim, len(sim), dim=0)
        else:
            val, indices = torch.topk(sim, self.TOPK, dim=0)
        # print the idx image and the topk images
        return val, indices