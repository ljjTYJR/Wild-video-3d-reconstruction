import numpy as np
import torch
import droid_backends
from lietorch import SE3
import rerun as rr
import cv2

class RerunVisualizer:
    def __init__(self, video):
        self.video = video
        rr.init('DROID_Mast3R_SLAM')
        rr.connect()
        rr.set_time_sequence("#frame", 0)

        self.point_label = 'world/points'
        self.path_label = 'world/camera_path'
        self.camera_label = 'world/camera'
        self.image_label = 'world/image'

    def __call__(self, point, path, camera, image, frame_n=None):
        t0 = 0
        t1 = self.video.counter.value
        dirty_index = torch.arange(t0, t1).cuda().long()

        poses = torch.index_select(self.video.poses, 0, dirty_index)
        disps = torch.index_select(self.video.disps, 0, dirty_index)


        # TODO: prune the outlier points
        if frame_n is not None:
            rr.set_time_sequence("#frame", frame_n)
        else:
            rr.set_time_sequence("#frame", t1)
        if point is True:

            points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu()

            images_original = torch.index_select(self.video.images, 0, dirty_index).cpu()
            images = images_original[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = points.reshape(-1, 3)
            colors = images.reshape(-1, 3)
            rr.log(self.point_label,
                   rr.Points3D(points, colors=colors))

        poses = SE3(poses).inv().data.cpu().numpy()
        translations = poses[:, :3]
        rotations = poses[:, 3:]

        if path is True:
            rr.log(self.path_label,
                   rr.LineStrips3D([translations], colors=[[255, 0, 0]]))

        if camera is True:
            rr.log(self.camera_label,
               rr.Transform3D(translation=translations[-1], rotation=rr.Quaternion(xyzw=rotations[-1]), scale=0.0005)
            )

        if image is True:
            image_current = self.video.images[self.video.counter.value-1].cpu().numpy()
            image_current = np.transpose(image_current, (1, 2, 0))
            image_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2BGR)
            rr.log(self.image_label,
                   rr.Image(image_current))