# 3D reconstruction from in-the-wild videos

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

3. **Install other dependencies:**
   - Install COLMAP & GLOMAP: Refer to https://github.com/colmap/glomap
   - Install `hloc` (for loop closure detection):
     ```bash
     pip install git+https://github.com/cvg/Hierarchical-Localization.git@v1.4
     ```

4. **Install the lietorch library:**
   ```bash
   pip install thirdparty/lietorch
   ```

5. **Build the SLAM extensions:**
   ```bash
   pip install -e .
   ```

6. **Mask and depth generation tools:**
   - We use `UniDepth` to generate dense depth maps
   - We use `MaskRCNN` to generate dynamic object masks

## Usage

### Running the Demo

Run the SLAM system on an image sequence:

```bash
conda activate dpvo
python dpvo_demo.py \
  --imagedir=/path/to/images \
  --depthdir=/path/to/depths \
  --calib=calib/calibration.txt \
  --stride=1 \
  --skip=0 \
  --buffer=2048 \
  --export_colmap
```

**Required Arguments:**
- `--imagedir`: Path to input image directory

**Optional Arguments:**
- `--depthdir`: Path to depth maps (enables depth-aided tracking)
- `--maskdir`: Path to dynamic object masks (filters dynamic objects)
- `--calib`: Camera calibration file (format: `fx fy cx cy [k1 k2 p1 p2 k3]`)
- `--stride`: Frame sampling stride (default: 1)
- `--skip`: Number of initial frames to skip (default: 0)
- `--buffer`: Maximum buffer size for keyframes (default: 2048)
- `--export_colmap`: Export results in COLMAP format for further processing
- `--rerun`: Enable Rerun visualization (https://rerun.io/)
- `--loop_enabled`: Enable loop closure detection


## Data

### Videos used in the paper:

| Name                | Link                                                       | Description |
| ------------------- | ---------------------------------------------------------- | ----------- |
| Yanshan Park, China | [Link](https://www.youtube.com/watch?v=D8B30GIX)           |             |
| Taicang Park, China | [Link](https://www.youtube.com/watch?v=LJf7LKLvmUc)        |             |
| Helsingborg, Sweden | [Link](https://www.youtube.com/watch?v=wUZ_zslH3vY&t=300s) |             |

### Extract frames from videos
1. Install the youtube video download tool:
   https://github.com/yt-dlp/yt-dlp

2. Use ffmpeg to extract frames. We extract with resolution with 512*384 and with 5 FPS with 15 minutes each video clip.