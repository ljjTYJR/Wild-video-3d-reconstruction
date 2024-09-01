# Mast3R-DROID-SLAM

Use the [Mast3R model](https://github.com/naver/mast3r) and [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) to make a more robust SLAM pipeline.


# Process descrption
- After the initialization, we run the `Mast3R` based SLAM pipeline. The initialization of the Mast3R SLAM comes from the DROID-SLAM initialization by upsamplng (interpolation)
  - We select the latest `window` keyframes from the DROID buffer, then do the dust3r-based SLAM to predict the latest frame parameters.
  - [Optional] When doing `mast3r` SLAM, we can only fix one frame in the window. Now we only free the optimized frame (namely, the latest frame).
  - If the memory is not enough, we might not need the symmetric inference, just one-direction
- [Optional] We can also maintain an extra Mast3R local map for SLAM.


# Tips
- When visualizing the point cloud in the `rerun`, we would better to visualize based on the camera paths.
- When feeding to the mast3r model, the depths larger than 2*`median` values are clamped to 2*`median`, which might incur problems.