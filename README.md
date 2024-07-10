# gtsam_points

This is a collection of [GTSAM](https://gtsam.org/) factors and optimizers for range-based SLAM.

Tested on Ubuntu 22.04 / 24.04 and CUDA 12.2, and NVIDIA Jetson Orin with **GTSAM 4.2a9**.


[![Doc](https://img.shields.io/badge/API_list-Doxygen-blue)](https://koide3.github.io/gtsam_points/doc_cpp/index.html) [![Build](https://github.com/koide3/gtsam_points/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/gtsam_points/actions/workflows/build.yml)

## Factors

### Scan Matching Factors

- **IntegratedICPFactor & IntegratedPointToPlaneICPFactor**  
    The conventional point-to-point and point-to-plane ICP [[1]](#ICP).
- **IntegratedGICPFactor**  
    Generalized ICP based on the distribution-to-distribution distance [[2]](#GICP).
- **IntegratedVGICPFactor**  
    GICP with voxel-based data association and multi-distribution-correspondence [[3]](#VGICP1)[[4]](#VGICP2).
- **IntegratedVGICPFactorGPU**  
    GPU implementation of VGICP [[3]](#VGICP1)[[4]](#VGICP2).  
    To enable this factor, set ```-DBUILD_WITH_CUDA=ON```.
- **IntegratedLOAMFactor**  
    Matching cost factor based on the combination of point-to-plane and point-to-edge distances [[5]](#LOAM)[[6]](#LEGO).


### Colored Scan Matching Factors

- **IntegratedColorConsistencyFactor**  
    Photometric ICP error [[7]](#COLORED).
- **IntegratedColoredGICPFactor**  
    Photometric ICP error + GICP geometric error [[2]](#GICP)[[7]](#COLORED).


### Continuous-time ICP Factors

- **IntegratedCT_ICPFactor**  
    Continuous Time ICP Factor [[8]](#CTICP).
- **IntegratedCT_GICPFactor**  
    Continuous Time ICP with GICP's D2D matching cost [[2]](#GICP)[[8]](#CTICP).


### Bundle Adjustment Factors

- **PlaneEVMFactor and EdgeEVMFactor**  
    Bundle adjustment factor based on Eigenvalue minimization [[9]](#BA_EVM).
- **LsqBundleAdjustmentFactor**  
    Bundle adjustment factor based on EVM and EF optimal condition satisfaction [[10]](#BA_LSQ).


## Optimizers for GPU Factors

All the following optimizers were derived from the implementations in GTSAM.

- **LevenbergMarquardtOptimizerExt**
- **ISAM2Ext**
- **IncrementalFixedLagSmootherExt**


## Nearest Neighbor Search
- **KdTree**
    KdTree with parallel tree construction. Derived from [nanoflann](https://github.com/jlblancoc/nanoflann).
- **IncrementalVoxelMap**
    Incremental voxel-based nearest neighbor search (iVox) [[11]](#IVOX).
- **IncrementalCovarianceVoxelMap**
    Incremental voxelmap with online normal and covariance estimation.


## Continuous-Time Trajectory (Under development)
- **B-Spline**
    Cubic B-Spline-based interpolation and linear acceleration and angular velocity expressions [[12]](#BSPLINE_D).
- **ContinuousTrajectory**
    Cubic B-Spline-based continuous trajectory representation for offline batch optimization.


## Installation

```bash
# Install gtsam
git clone https://github.com/borglab/gtsam
cd gtsam
git checkout 4.2a9

mkdir build && cd build
cmake .. \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF

make -j$(nproc)
sudo make install

# [optional] Install iridescence visualization library
# This is required for only demo programs
sudo apt install -y libglm-dev libglfw3-dev libpng-dev
git clone https://github.com/koide3/iridescence --recursive
mkdir iridescence/build && cd iridescence/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

## Build gtsam_points
git clone https://github.com/koide3/gtsam_points
mkdir gtsam_points/build && cd gtsam_points/build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Optional cmake arguments
# cmake .. \
#   -DBUILD_DEMO=OFF \
#   -DBUILD_TESTS=OFF \
#   -DBUILD_WITH_CUDA=OFF \
#   -DBUILD_WITH_MARCH_NATIVE=OFF

make -j$(nproc)
sudo make install
```

## Demo

```bash
cd gtsam_points
./build/demo_matching_cost_factors
./build/demo_bundle_adjustment
./build/demo_continuous_time
./build/demo_continuous_trajectory
./build/demo_colored_registration
```

## Videos

- [Multi-scan registration of 5 frames (= A graph with 10 registration factors)](https://youtu.be/HCXCWlx_VOM)
- [Bundle adjustment factor](https://youtu.be/tuDV0GCOZXg)
- [Continuous-time ICP factor](https://youtu.be/Xv2-qDlzQYM)
- [Colored ICP factor](https://youtu.be/xEQmiFV79LU)
- [Incremental voxel mapping and normal estimation](https://youtu.be/gDiKqQDc7yo)
- [SE3 BSpline interpolation](https://youtu.be/etAI8go3b8U)

## License

This library is released under the MIT license.

## Dependencies
- [Eigen](https://eigen.tuxfamily.org/index.php)
- [nanoflann](https://github.com/jlblancoc/nanoflann)
- [GTSAM](https://gtsam.org/)
- [optional] [OpenMP](https://www.openmp.org/)
- [optional] [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [optional] [iridescence](https://github.com/koide3/iridescence)


## Disclaimer

The test data in ```data``` directory are generated from [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/) and [The Newer College Dataset](https://ori-drs.github.io/newer-college-dataset/). Because they employ ```Creative Commons BY-NC-SA License 3.0 and 4.0```, the test data must not be used for commercial purposes.

## References
<a name="ICP"></a> [1] Zhang, "Iterative Point Matching for Registration of Free-Form Curve", IJCV1994  
<a name="GICP"></a> [2] Segal et al., "Generalized-ICP", RSS2005  
<a name="VGICP1"></a> [3] Koide et al., "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration", ICRA2021  
<a name="VGICP2"></a> [4] Koide et al., "Globally Consistent 3D LiDAR Mapping with GPU-accelerated GICP Matching Cost Factors", RA-L2021  
<a name="LOAM"></a> [5] Zhang and Singh, "Low-drift and real-time lidar odometry and mapping", Autonomous Robots, 2017  
<a name="LEGO"></a> [6] Tixiao and Brendan, "LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain", IROS2018  
<a name="COLORED"></a> [7] Park et al., "Colored Point Cloud Registration Revisited", ICCV2017  
<a name="CTICP"></a> [8] Bellenbach et al., "CT-ICP: Real-time Elastic LiDAR Odometry with Loop Closure", 2021  
<a name="BA_EVM"></a> [9] Liu and Zhang, "BALM: Bundle Adjustment for Lidar Mapping", IEEE RA-L, 2021  
<a name="BA_LSQ"></a> [10] Huang et al, "On Bundle Adjustment for Multiview Point Cloud Registration", IEEE RA-L, 2021  
<a name="IVOX"></a> [11] Bai et al., "Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels", IEEE RA-L, 2022  
<a name="BSPLINE_D"></a> [12] Sommer et al., "Efficient Derivative Computation for Cumulative B-Splines on Lie Groups", CVPR2020
