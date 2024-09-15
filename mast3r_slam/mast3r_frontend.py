##
# @ Desc:
#   The frontend module of the Mast3r-based SLAM
#   The frontend will receive the keyframe from the motion filter, and then do the local bundle adjustment based on local keyframes.
#   The frame information will be stored in the Mast3rVideo object.
# @ Author: Shuo Sun
##

from dust3r.utils.image import load_images, format_images

class Mast3rFrontend:
    def __init__(self, mast3r_model, video):
        self.mast3r_model = mast3r_model
        self.video = video

        self.count = 0
        self.warmup = 8
        self.ba_window = 8
        # local ba setting


        self.is_initialized = False

    def __initialize__(self):
        """ Initialize the frontend; After the initialization, we will fix the intrinsic for following tracking; also, the scale. """
        initial_images = self.video.images[:self.ba_window]
        # make the image mast3r/dust3r-support format
        images = format_images(initial_images)

        # do the 3D BA

        # export the needed parameters
        self.is_initialized = True

    def __track__(self):
        """ Video local Mast3r BA except the initialization; Use the initialized intrinsic """
        pass

    def __call__(self):
        """ The main thread of the frontend """
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize__()
        elif self.is_initialized and self.video.counter.value % self.ba_window == 0:
            self.__track__()
        else:
            pass

