""" Test using NetVLAD for the image retrieval """
import os
import shutil
import sys
sys.path.append('./')
from hloc import extract_features, pairs_from_retrieval

IMG_DIR = '/media/shuo/T7/duslam/video_images/fpv3_10fps/failurecase_2/images'
OUTS_DIR = '/media/shuo/T7/duslam/video_images/fpv3_10fps/failurecase_2/net_vlad_outs'
retrieval_option = 'netvlad'

retrieval_conf = extract_features.confs[retrieval_option]
retrieval_path = extract_features.main(retrieval_conf, IMG_DIR, OUTS_DIR)

sfm_pairs = os.path.join(OUTS_DIR, 'pairs-from-retrieval')
number_imgs=15
try:
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=number_imgs)
except Exception as e:
    print("retrieval_path", retrieval_path)
    print("sfm_pairs", sfm_pairs)
    print("number_imgs", number_imgs)
    print(f"Error: {e}")
    quit()