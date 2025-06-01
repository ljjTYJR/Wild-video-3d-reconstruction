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

# Todo
- [ ] visualize the Mast3R local map: pass the optimized pose and depth back to DROID and update it.
- [x] Visualize the optimization process, the `opt_points` change process
- [x] Try without DOIRD initialization to optimize the Mast3R point cloud, will the result be better?
- [x] Initialize the `pw_poses` with the initial registration result (assuming that the gradient-based optimization is sensitive to the initialization)
  - The creteria should be that the initial guess align well (though not perfectly).
  - Currently we use the `log` of confidence when registration, test whether we need it or not.
  - *Bug reason*: the pose in the DROID is world-to-camera while the pose in the dust3r is camera to world.
- [x] Use the `dust3r` initialized map to feed the DROID and run optimization again.
- [x] Test the new motion-only keyframe detection and initialize with the dust3r
  - [x] initialize the new keyframe in the dust3r by the pair-wise prediction.
- [ ] Use the `COLMAP` to refine the initialized camera intrinsic.
  - For a rotation-only scene, it is difficult to estimate or refine the camera intrinsic. It also has some requirements for the recorded scene.
- [ ] Use the `GeoCalib` to estimate the camera intrinsic parameters.
- [ ] Run the local SFM based on the Mast3R and align them using the Two-view geometry.
- [ ] Run the batch BA to refine the pose and the scene when merge two separate models;
- [ ] Add the dynamic object detection

# Not very important
- [ ] Compare the dust3r-based optimization and the mast3r-based optimization (seems that not all points will be used to optimized or just patch-wise optimization)
- [ ] Understand the optical flow update operator
- [ ] In addition to the dense optical flow, we can also use the sparse correspondence to estimate the optical flow (like the superpoint, etc.)

# Notes
- In the Mast3R optimized point cloud, there is distoration in the optimized points (which might be a problem). This can come from the incorrect intrinsics
- The distoration of the camera model really affects the reconstruction quality. If the estimated camera model is not good enough, we need to fix it.
- To reconstruct the wild video, this paper `ego-slam` looks good

# Bug reports
- [ ] Currently, there is a bug in the Mast3R SLAM optimization process, the final optimized result is not correct. Figure out why! The expected result at least should re-produce the result as in the `reproduce_bug.py` in the `duslam` project.

# Difficulties
- The unknown camera intrinsic parameters.
- The dynamic objects in the wild video.
- The rotation-only camera motions during the process.

# Features in the future
- [ ] For `mast3r-slam`, the front-end and the back-end is implemented in two separate processes.

# Install
- `pip install thirdparty/lietorch` // we remove the lietorch installation in the parent directory.
-