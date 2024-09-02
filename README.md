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
- [ ] Use the `dust3r` initialized map to feed the DROID and run optimization again.

# Notes
- In the Mast3R optimized point cloud, there is distoration in the optimized points (which might be a problem). This can come from the incorrect intrinsics

# Bug reports
- [ ] Currently, there is a bug in the Mast3R SLAM optimization process, the final optimized result is not correct. Figure out why! The expected result at least should re-produce the result as in the `reproduce_bug.py` in the `duslam` project.