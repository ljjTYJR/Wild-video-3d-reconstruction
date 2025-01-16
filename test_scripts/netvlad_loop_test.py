""" Test using NetVLAD for the image retrieval """
import os
import shutil
import sys
import torch
import numpy as np
import tqdm
sys.path.append('.')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from hloc import extract_features, pairs_from_retrieval
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import read_image
from hloc import extractors

IMG_DIR = '/media/shuo/T7/duslam/video_images/china_classical_park_512/loop_test/bd_images'
QUERY_DIR = '/media/shuo/T7/duslam/video_images/china_classical_park_512/loop_test/query_images'
WINODW_SIZE = 30
retrieval_option = 'netvlad'
TOPK=20
netvlad_confs={
    "output": "global-feats-netvlad",
    "model": {"name": "netvlad"},
    "preprocessing": {"resize_max": 1024},
}

class RetrievalNetVLAD:

    def __init__(self):
        Model = dynamic_load(extractors, netvlad_confs["model"]["name"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.as_half = False
        self.netvlad_model=Model(netvlad_confs["model"]).eval().to(self.device)

        BUFFER_SIZE=2000
        # self.netvlad_db =np.zeros((BUFFER_SIZE, 4096), dtype=np.float16 if self.as_half else np.float32)
        self.netvlad_db = torch.zeros((BUFFER_SIZE, 4096), dtype=torch.float16 if self.as_half else torch.float32)
        self.img_buffer = {}

    @torch.no_grad()
    def insert_img(self, idx, img_path):
        image = read_image(img_path)
        # Refer to the `hloc` implementation, the image is normalized to [0, 1] (in the NeTVLAD, will be recovered to [0, 255])
        image = image.astype(np.float32).transpose((2, 0, 1)) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        pred = self.netvlad_model({"image": image.to(self.device, non_blocking=True)})
        # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        if self.as_half:
            pred['global_descriptor'] = pred['global_descriptor'].to(torch.float16)
        self.netvlad_db[idx] = pred['global_descriptor']

    @torch.no_grad()
    def extract_feature(self, img_dir):
        for idx, img in enumerate(sorted(os.listdir(img_dir))):
            self.img_buffer[idx] = img
            img_path = os.path.join(img_dir, img)
            self.insert_img(idx, img_path)
            # query_indices = self.query(idx)

    @torch.no_grad()
    def query(self, idx):
        # query index with prior poses by calculating cosine similarity
        query_desc = self.netvlad_db[idx]
        indices = []
        if idx <= WINODW_SIZE:
            print(idx, "Not enough images to query")
            return indices
        potential_indices = list(range(idx-WINODW_SIZE))
        sim = torch.matmul(self.netvlad_db[potential_indices], query_desc.unsqueeze(0).T)
        if len(sim) < TOPK:
            val, indices = torch.topk(sim, len(sim), dim=0)
        else:
            val, indices = torch.topk(sim, TOPK, dim=0)
        # print the idx image and the topk images
        print(self.img_buffer[idx])
        print(indices, val)
        # corrd_imgs = [self.img_buffer[potential_indices[i]] for i in indices]
        # print(corrd_imgs)

    @torch.no_grad()
    def query_by_img(self, query_img_path):
        # query the image by iterating the img in the path
        # use tqdm
        for idx, img in enumerate(sorted(os.listdir(query_img_path))):
            img_path = os.path.join(query_img_path, img)
            image = read_image(img_path)
            image = image.astype(np.float32).transpose((2, 0, 1)) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
            pred = self.netvlad_model({"image": image.to(self.device, non_blocking=True)})
            if self.as_half:
                pred['global_descriptor'] = pred['global_descriptor'].to(torch.float16)
            query_desc = pred['global_descriptor'].cpu()

            sim = torch.matmul(self.netvlad_db, query_desc.T)
            if len(sim) < TOPK:
                val, indices = torch.topk(sim, len(sim), dim=0)
            else:
                val, indices = torch.topk(sim, TOPK, dim=0)
            print(idx)
            print(val[0], indices[0])

def main():
    netvlad_bd = RetrievalNetVLAD()
    netvlad_bd.extract_feature(IMG_DIR)
    netvlad_bd.query_by_img(QUERY_DIR)

if __name__ == '__main__':
    main()