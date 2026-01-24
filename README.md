# gtsam_points

This is a collection of [GTSAM](https://gtsam.org/) factors and optimizers for range-based SLAM.

Tested on Ubuntu 22.04 / 24.04 and CUDA 12.2 / 12.6 / 13.1, and NVIDIA Jetson Orin with **GTSAM 4.2a9 and 4.3a0**.

[![DOI](https://zenodo.org/badge/819211095.svg)](https://zenodo.org/doi/10.5281/zenodo.13378351) [![Doc](https://img.shields.io/badge/API_list-Doxygen-blue)](https://koide3.github.io/gtsam_points/doc_cpp/index.html) [![Build](https://github.com/koide3/gtsam_points/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/gtsam_points/actions/workflows/build.yml)

## Updates

- 2026/01/24 : v1.2.0 released. Support for both GTSAM 4.2a9 and GTSAM 4.3a0, and CUDA 13.1. Intensity support for Gaussian voxelmap.
- 2025/06/15 : The base GTSAM version has been changed. Make sure you have rebuilt and installed **GTSAM 4.3a0**.

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
- **FastOccupancyGrid**
    Binary occupancy grid with bit blocks and flat hashing for efficient point cloud overlap estimation.

## Point Features and Global Point Cloud Registration
- **Point Feature Histogram (PFH)** [[13]](#PFH)
- **Fast Point Feature Histogram (FPFH)** [[14]](#FPFH)
- **RANSAC**
    RANSAC-based global point cloud registration. Supports 6DoF and 4DoF (XYZ + Yaw) estimation [[15]](#RANSAC).
- **Graduated Non-Convexity**
    Graduated Non-Contexity-based global point cloud registration. Supports 6DoF and 4DoF (XYZ + Yaw) estimation [[16]](#GNC).

## Object Segmentation
- **Region Growing Segmentation** [[17]](#RegionGrowing)
- **Min-Cut Segmentation** [[18]](#MinCut)

## Continuous-Time Trajectory (Under development)
- **B-Spline**
    Cubic B-Spline-based interpolation and linear acceleration and angular velocity expressions [[12]](#BSPLINE_D).
- **ContinuousTrajectory**
    Cubic B-Spline-based continuous trajectory representation for offline batch optimization.


## Installation

### Install from source

```bash
# Install gtsam
git clone https://github.com/borglab/gtsam
cd gtsam
git checkout 4.3a0

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
#   -DBUILD_DEMO=OFF \                # Set ON to build demo programs
#   -DBUILD_TESTS=OFF \               # Set ON to build unit tests
#   -DBUILD_TOOLS=OFF \               # Set ON to build tools
#   -DBUILD_WITH_TBB=OFF \            # Set ON to enable TBB
#   -DBUILD_WITH_OPENMP=OFF \         # Set ON to enable OpenMP
#   -DBUILD_WITH_CUDA=OFF \           # Set ON to enable CUDA support
#   -DBUILD_WITH_CUDA_MULTIARCH=OFF \ # Set ON to enable multi-arch CUDA support
#   -DCMAKE_CUDA_ARCHITECTURES=89 \   # If not specified, "native" architecture is used
#   -DBUILD_WITH_MARCH_NATIVE=OFF     # Set ON to enable -march=native (recommended to keep it OFF)

make -j$(nproc)
sudo make install
```

### Install from [PPA](https://github.com/koide3/ppa) [Ubuntu 22.04, 24.04] [AMD64, ARM64]

#### Setup PPA

```bash
curl -s https://koide3.github.io/ppa/setup_ppa.sh | sudo bash
```

#### Install GTSAM and gtsam_points

```bash
# Without CUDA
sudo apt install -y libgtsam-points-dev

# with CUDA 12.2 (for only Ubuntu 22.04)
sudo apt install -y libgtsam-points-cuda12.2-dev

# with CUDA 12.5
sudo apt install -y libgtsam-points-cuda12.5-dev

# with CUDA 13.1
sudo apt install -y libgtsam-points-cuda13.1-dev
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

## Citation

```
@software{gtsam_points,
  author       = {Kenji Koide},
  title        = {gtsam_points : A collection of GTSAM factors and optimizers for point cloud SLAM},
  month        = Aug,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {1.0.4},
  doi          = {10.5281/zenodo.13378352},
  url          = {https://github.com/koide3/gtsam_points)}}
}
```

## Dependencies
- [Eigen](https://eigen.tuxfamily.org/index.php)
- [nanoflann](https://github.com/jlblancoc/nanoflann)
- [GTSAM](https://gtsam.org/)
- [optional] [PCL]([https://www.openmp.org/](https://pointclouds.org/))
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
<a name="PFH"></a> [13] Rusu et al., "ligning Point Cloud Views using Persistent Feature Histograms", IROS2008  
<a name="FPFH"></a> [14] Rusu et al., "Fast Point Feature Histograms (FPFH) for 3D Registration", ICRA2009  
<a name="RANSAC"></a> [15] Buch et al., "Pose Estimation using Local Structure-Specific Shape and Appearance Context", ICRA2013  
<a name="GNC"></a> [16] Zhou et al., "Fast Global Registration", ECCV2016  
<a name="RegionGrowing"></a> [17] Rabbani et al., "Segmentation of Point Clouds Using Smoothness Constraint", Remote Sensing and Spatial Information Sciences, 2006  
<a name="MinCut"></a> [18] Golovinskiy et al., "Min-Cut Based Segmentation of Point Clouds", S3DV-WS@ICCV2009  


