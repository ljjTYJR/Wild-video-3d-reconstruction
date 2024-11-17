""" Test the BoW model with ORB featurs, the input will be a sequence of images """

import os
import cv2
import time
import numpy as np
from einops import parse_shape
from multiprocessing import Process, Queue, Value

try:
    import dpretrieval
    dpretrieval.DPRetrieval
except:
    raise ModuleNotFoundError("Couldn't load dpretrieval. It may not be installed.")

NMS = 50 # Slow motion gets removed from keyframes anyway. So this is really the keyframe distance

RAD = 15 # The idex difference smaller than this will be considered as the loop, otherwise, just a local window


def _dbow_loop(in_queue, out_queue, vocab_path, ready):
    """ Simulate the multiple processes in the retrieval system """
    dbow = dpretrieval.DPRetrieval(vocab_path, RAD)
    ready.value = 1
    while True:
        n, image = in_queue.get()
        dbow.insert_image(image)
        q = dbow.query(n)
        out_queue.put((n, q))

class RetrievalDBOW:

    def __init__(self, vocab_path="ORBvoc.txt"):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError("""Missing the ORB vocabulary. Please download and un-tar it from """
                                  """https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Vocabulary/ORBvoc.txt.tar.gz""")
        else:
            print("loading orb vocabulary from", vocab_path)

        self.img_dir='/media/shuo/T7/duslam/video_images/temple/seq1/small_test/images'
        self.img_buffer = {}
        self.in_queue = Queue(maxsize=20)
        self.out_queue = Queue(maxsize=20)
        # ready = Value('i', 0) # mutex
        # self.proc = Process(target=_dbow_loop, args=(self.in_queue, self.out_queue, vocab_path, ready))
        # self.proc.start()
        # self.being_processed = 0
        # while not ready.value:
            # wait for the process to be ready
            # time.sleep(0.01)
        self.dbow = dpretrieval.DPRetrieval(vocab_path, RAD)

    def read_images(self, image_dir):
        if not os.path.exists(image_dir):
            raise FileNotFoundError("Image directory not found")

        img_names = sorted(os.listdir(image_dir))
        # sorted list of images
        for i, img in enumerate(sorted(os.listdir(image_dir))):
            img_i = cv2.imread(os.path.join(image_dir, img)) # (H, W, C)
            # assert the information
            assert isinstance(img_i, np.ndarray)
            assert img_i.dtype == np.uint8
            assert parse_shape(img_i, '_ _ RGB') == dict(RGB=3)
            self.dbow.insert_image(img_i)
            score, j, _ = self.dbow.query(i)
            print(i, img, img_names[j], score)

    def extract_features(image):
        pass

    def detect_loop(img):
        pass

    def main(self):
        self.read_images(self.img_dir)
        print(list(self.img_buffer))

if __name__ == '__main__':
    dow = RetrievalDBOW('../checkpoints/ORBvoc.txt')
    dow.main()