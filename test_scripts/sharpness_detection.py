import cv2
import numpy as np
import os

def evaluate_sharpness(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian

if __name__ == "__main__":
    img_dir='/media/shuo/T7/duslam/video_images/fpv3_10fps/failurecase_2/images'
    log = open('sharpness.log', 'w')
    # iterate over all images in the directory
    for img in os.listdir(img_dir):
        sharpness = evaluate_sharpness(os.path.join(img_dir, img))
        log.write(f"{img}: {sharpness}\n")