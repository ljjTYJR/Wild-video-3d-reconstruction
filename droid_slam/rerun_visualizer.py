import numpy as np
import torch
import droid_backends
from lietorch import SE3
import rerun as rr
import cv2

def get_current_color_points(video):
    t0 = 0
    t1 = video.counter.value
    dirty_index = torch.arange(t0, t1).cuda().long()

    poses = torch.index_select(video.poses, 0, dirty_index)
    disps = torch.index_select(video.disps, 0, dirty_index)

    points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

    images_original = torch.index_select(video.images, 0, dirty_index).cpu()
    images = images_original[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
    points = points.reshape(-1, 3)
    colors = images.reshape(-1, 3)
    return points, colors

class RerunVisualizer:
    def __init__(self, video):
        self.video = video
        rr.init('droid_slam_visualizer')
        rr.connect()
        rr.set_time_sequence("#frame", 0)

        self.point_label = 'world/points'
        self.path_label = 'world/camera_path'
        self.camera_label = 'world/camera'
        self.image_label = 'world/image'

        self.filter_thresh = 0.005

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

            points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu() # it is the intrinsic after the RES

            images_original = torch.index_select(self.video.images, 0, dirty_index).cpu()
            images = images_original[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = points.reshape(-1, 3)
            colors = images.reshape(-1, 3)

            # filter out the outlier points
            thresh = self.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            count = droid_backends.depth_filter(
                self.video.poses, self.video.disps, self.video.intrinsics[0], dirty_index, thresh)
            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True))).reshape(-1)
            points = points[masks]
            colors = colors[masks]

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