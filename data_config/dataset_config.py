"""
Dataset configuration for automatic DROID-SLAM demo
Edit this file to add or modify dataset paths
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Dataset:
    imagedir: str
    calib: str
    depthdir: Optional[str] = None
    start_idx: int = 0
    end_idx: Optional[int] = None

    @property
    def exists(self) -> bool:
        """Check if the image directory exists"""
        return os.path.exists(self.imagedir)

    @property
    def calib_exists(self) -> bool:
        """Check if the calibration file exists"""
        return os.path.exists(self.calib)

    @property
    def is_valid(self) -> bool:
        """Check if both image directory and calibration file exist"""
        return self.exists and self.calib_exists

DATASETS = {
    # "china_park": Dataset(
    #     imagedir="/media/shuo/T7/duslam/video_images/china_classical_park_512/images",
    #     calib="calib/park_colmap_512.txt",
    #     start_idx=0,
    #     end_idx=None
    # ),

    "he_01": Dataset(
        imagedir="/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/images",
        calib="calib/helsingborgw_UZ_zslH3vY_512.txt",
        start_idx=0,
        end_idx=2700
    ),

    "he_02": Dataset(
        imagedir="/media/shuo/T7/duslam/video_images/youtube_HELSINGBORG_wUZ_zslH3vY/clip0/512/images",
        calib="calib/helsingborgw_UZ_zslH3vY_512.txt",
        start_idx=2700,
        end_idx=None
    ),

    "lund": Dataset(
        imagedir="/media/shuo/T7/duslam/video_images/youtube_lund_Nhc5BNlfDms/images/512/images",
        calib="calib/lund_Nhc5BNlfDms.txt",
        start_idx=0,
        end_idx=None
    ),

    "uppsala": Dataset(
        imagedir="/media/shuo/T7/duslam/video_images/youtube_upplasa_aVh_jTIP2cE/image/512/images",
        calib="calib/upplasa_aVh_jTIP2cE.txt",
        start_idx=0,
        end_idx=None
    ),
}