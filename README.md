# Visual SLAM System

A robust visual SLAM pipeline for real-time camera tracking and 3D reconstruction, featuring deep learning-based pose estimation and bundle adjustment.

## Overview

This project provides a complete visual SLAM system with:
- Real-time frame-by-frame camera tracking and mapping
- Deep learning-based pose and depth optimization on keyframe windows
- Optional local map maintenance for enhanced performance
- Support for monocular and depth-aided configurations

## Installation

### Prerequisites
- CUDA-capable GPU (compute capability 6.0+)
- CUDA Toolkit 11.x or 12.x
- Conda package manager

### Setup Steps

1. **Create and activate the conda environment:**
   ```bash
   conda create -n dpvo python=3.10.14
   conda activate dpvo
   ```

2. **Install PyTorch with CUDA support:** (specify the version according to your CUDA installation)
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

   # Or for CUDA 12.1
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. **Install glomap:**
   Refer to : https://github.com/colmap/glomap

4. **Install the lietorch library:**
   ```bash
   pip install thirdparty/lietorch
   ```

5. **Build the SLAM extensions:**
   ```bash
   pip install -e .
   ```
6. Mask and depths:
   1. We use `UniDepth` to generate dense depth maps;
   2. We use `MaskRCNN` to generate dynamic object masks.

## Usage

### Basic Demo

Run the SLAM system on an image sequence:

```bash
conda activate dpvo
python dpvo_demo.py \
  --imagedir=/path/to/images \
  --depthdir=/path/to/depths \
  --maskdir=/path/to/masks \
  # --calib=calib/calibration.txt \ (optional)
  --stride=1 \
  --buffer=2048 \
  --keyframe_thresh=2.0 \
  --rerun
```

**Arguments:**
- `--imagedir`: Path to input image directory (required)
- `--depthdir`: Path to depth maps (optional)
- `--maskdir`: Path to dynamic object masks (optional)
- `--calib`: Camera calibration file path (optional, will be estimated if not provided)
- `--stride`: Frame sampling stride (default: 1)
- `--buffer`: Maximum buffer size for keyframes (default: 2048)
- `--keyframe_thresh`: Threshold for keyframe selection (default: 2.0)
- `--export_colmap`: Export results in COLMAP format
- `--rerun`: Enable Rerun visualization
- `--disable_vis`: Disable visualization during processing

### Camera Calibration Format

Calibration files should contain camera intrinsics in the following format:
```
fx fy cx cy [k1 k2 p1 p2 k3]
```
where `fx, fy, cx, cy` are the focal lengths and principal point, and the distortion coefficients are optional.


## References

- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [DPT](https://github.com/isl-org/DPT)