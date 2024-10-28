# Test the superpoint detection speed
import os
import shutil
import sys
import torch
import numpy as np
import time
from lightglue import SuperPoint
from lightglue.utils import load_image

extractor = SuperPoint(max_num_keypoints=96).eval().cuda()  # load the extractor
start = time.time()
for idx, img in enumerate(sorted(os.listdir('/media/shuo/T7/duslam/video_images/fpv3_10fps/failurecase1/images'))):
    image = load_image(os.path.join('/media/shuo/T7/duslam/video_images/fpv3_10fps/failurecase1/images', img)).cuda()
    with torch.no_grad():
        pred = extractor.extract(image)
        print(idx)
end = time.time()
print(f"Time cost: {end - start} seconds.")