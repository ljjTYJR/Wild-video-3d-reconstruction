"""
We implement the NetVLAD descriptor for retrieval, note that we implement it offline.
"""
import os
import time
import torch
import numpy as np
from einops import parse_shape
import multiprocessing as mp
import queue
from multiprocessing import Process, Queue, Value
from dpvo.netvlad_retrieval import RetrievalNetVLADOffline

NMS = 50 # Slow motion gets removed from keyframes anyway. So this is really the keyframe distance
SKIP_WINDOW = 50

def _dvlad_loop(in_queue, out_queue, vlad_db, ready):

    """ Run vlad retrieval """
    ready.value = 1
    while True:
        n, desc = in_queue.get()
        vlad_db.insert_desc(n, desc)
        v, k = vlad_db.query_online(n, SKIP_WINDOW, top_k=1)
        if v is None or k is None:
            v = torch.tensor([0.0])
            out_queue.put((n, (v,k)))
        else:
            out_queue.put((n, (v.item(), k.item())))

class RetrievalNetVLAD:
    def __init__(self, nvlad_db):
        self.descriptor_buffer = {}
        self.stored_indices = np.zeros(100000, dtype=bool)

        # Keep track of detected and closed loops
        self.prev_loop_closes = []
        self.found = []

        self.nvlad = nvlad_db

        self.in_queue = Queue(maxsize=40)
        self.out_queue = Queue(maxsize=40)
        ready = Value('i', 0)
        self.proc = Process(target=_dvlad_loop, args=(self.in_queue, self.out_queue, self.nvlad, ready))
        self.proc.start()
        self.being_processed = 0
        while not ready.value:
            time.sleep(0.01)

    def keyframe(self, k):
        """ Once we keyframe an image, we can safely cache all images
         before & including it """
        tmp = dict(self.descriptor_buffer)
        self.descriptor_buffer.clear()
        for n, v in tmp.items():
            if n != k:
                key = (n-1) if (n > k) else n
                self.descriptor_buffer[key] = v

    def save_up_to(self, c):
        for n in list(self.descriptor_buffer):
            if n <= c:
                assert not self.stored_indices[n]
                desc = self.descriptor_buffer.pop(n)
                desc = desc.contiguous()
                self.in_queue.put((n, desc))
                self.stored_indices[n] = True
                self.being_processed += 1

    def confirm_loop(self, i, j):
        """ Record the loop closure so we don't have redundant edges"""
        assert i > j
        self.prev_loop_closes.append((i, j))

    def _repetition_check(self, idx, num_repeat):
        """ Check that we've retrieved <num_repeat> consecutive frames """
        if (len(self.found) < num_repeat):
            return
        latest = self.found[-num_repeat:]
        (b, _), (i, j), _ = latest
        if (1 + idx - b) == num_repeat:
            return (i, max(j,1)) # max(j,1) is to avoid centering the triplet on 0

    def detect_loop(self, thresh, num_repeat=1):
        """ Keep popping off the queue until the it is empty
         or we find a positive pair """
        while self.being_processed > 0:
            x = self._detect_loop(thresh, num_repeat)
            if x is not None:
                return x

    def _detect_loop(self, thresh, num_repeat=1):
        """ Pop retrived pairs off the queue. Return if they have non-trivial score """
        assert self.being_processed > 0
        i, (score, j) = self.out_queue.get()
        self.being_processed -= 1
        if score < thresh:
            return
        assert i > j

        # Ensure that this edge is not redundant
        dists_sq = [(np.square(i - a) + np.square(j - b)) for a,b in self.prev_loop_closes]
        if min(dists_sq, default=np.inf) < np.square(NMS):
            return

        # Add this frame pair to the list of retrieved matches
        self.found.append((i, j))

        # Check that we've retrieved <num_repeat> consecutive frames
        return self._repetition_check(i, num_repeat)

    def __call__(self, image, n, tstamp):
        """ Store the image into the frame buffer """
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert parse_shape(image, '_ _ RGB') == dict(RGB=3)
        self.descriptor_buffer[n] = self.nvlad.nvlad_db[tstamp]

    def close(self):
        self.proc.kill()
        self.proc.join()