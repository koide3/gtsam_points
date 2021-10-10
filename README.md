# gtsam_ext

This is a collection of GTSAM factors and optimizers that would be useful for range-based SLAM.

Tested on Ubuntu 20.04 and CUDA 11.1.

## Factors

### Scan Matching Factors

- **IntegratedICPFactor & IntegratedPointToPlaneICPFactor**  
    The conventional point-to-point and point-to-plane ICP [[1]](#ICP)  
- **IntegratedGICPFactor**  
    Generalized ICP based on the distribution-to-distribution distance [[2]](#GICP)    
- **IntegratedVGICPFactor**  
    GICP with voxel-based data association and multi-distribution-correspondence [[3]](#VGICP1)[[4]](#VGICP2)    
- **IntegratedVGICPFactorGPU**  
    GPU implementation of VGICP [[3]](#VGICP1)[[4]](#VGICP2)  
    Need to set ```BUILD_WITH_CUDA``` cmake option to ```ON```
- **IntegratedLOAMFactor**  
    Matching cost based on the combination of point-to-plane and point-to-edge distances [[5]](#LOAM)[[6]](#LEGO)  
    
### Continuous-time ICP Factors

- **IntegratedCT_ICPFactor**  
    Continuous Time ICP Factor [[7]](#CTICP)  
- **IntegratedCT_GICPFactor**  
    Continuous Time ICP with GICP's D2D cost [[2]](#GICP)[[7]](#CTICP)

### Bundle Adjustment Factors

- **PlaneEVMFactor and EdgeEVMFactor**  
    Bundle adjustment factor based on Eigenvalue minimization [[8]](#BA_EVM)
- **LsqBundleAdjustmentFactor**  
    Bundle adjustment factor based on EVM and EF optimal condition satisfaction [[9]](#BA_LSQ)


## Optimizers for GPU-based Factors
All the following optimizers were derived from the implementations in GTSAM

- **LevenbergMarquardtOptimizerExt**
- **ISAM2Ext**
- **IncrementalFixedLagSmootherExt**


## Nearest Neighbor Search
- **KdTree**  
    Standard KdTree search using [nanoflann](https://github.com/jlblancoc/nanoflann)
- **VoxelSearch** (to be implemented)
- **ProjectiveSearch** (to be implemented)


## Installation

```bash
# Install gtsam
git clone https://github.com/borglab/gtsam
mkdir gtsam/build && cd gtsam/build
cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF

# If you face segfaults, try the following configuration
# cmake .. \
#   -DGTSAM_WITH_TBB=OFF \
#   -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF

make -j$(nproc)
sudo make install

# [optional] Install visualization library
# This is required for only demo programs
sudo apt install -y libglm-dev libglfw3-dev libpng-dev
git clone https://github.com/koide3/iridescence --recursive
mkdir iridescence/build && cd iridescence/build
cmake ..
make -j$(nproc)
sudo make install

## Install gtsam_ext
git clone https://github.com/koide3/gtsam_ext --recursive
mkdir gtsam_ext/build && cd gtsam/build
cmake ..

# Optional cmake arguments
# cmake .. \
#   -DBUILD_DEMO=OFF \
#   -DBUILD_TESTS=OFF \
#   -DBUILD_WITH_CUDA=OFF \
#   -DBUILD_WITH_MARCH_NATIVE=OFF \
#   -DBUILD_WITH_SYSTEM_EIGEN=OFF

make -j$(nproc)
```

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
<a name="CTICP"></a> [7] Bellenbach et al., "CT-ICP: Real-time Elastic LiDAR Odometry with Loop Closure", 2021  
<a name="BA_EVM"></a> [8] Liu and Zhang, "BALM: Bundle Adjustment for Lidar Mapping", IEEE RA-L, 2021  
<a name="BA_LSQ"></a> [9] Huang et al, "On Bundle Adjustment for Multiview Point Cloud Registration", IEEE RA-L, 2021