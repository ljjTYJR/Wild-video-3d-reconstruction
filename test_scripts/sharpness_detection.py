import cv2
import numpy as np
import os

def evaluate_sharpness(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian

if __name__ == "__main__":
    img_dir='/media/shuo/T7/rgbd_slam/tum_rgbd/rgbd_dataset_freiburg1_360/initialized/images'
    # iterate over all images in the directory
    for img in os.listdir(img_dir):
        sharpness = evaluate_sharpness(os.path.join(img_dir, img))
        print(f"{img}: {sharpness}")