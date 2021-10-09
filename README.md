# gtsam_ext

## Scan Matching Factors

- **IntegratedICPFactor & IntegratedPointToPlaneICPFactor**  
    The conventional point-to-point and point-to-plane ICP [[1]](#ICP)  
- **IntegratedGICPFactor**  
    Generalized ICP based on distribution-to-distribution distance [[2]](#GICP)    
- **IntegratedVGICPFactor**  
    GICP with voxel-based data association and multi-distribution-correspondence [[3]](#VGICP1)[[4]](#VGICP2)    
- **IntegratedVGICPFactorGPU**  
    GPU implementation of VGICP [[3]](#VGICP1)[[4]](#VGICP2)
- **IntegratedLOAMFactor**  
    Matching cost based on the combination of point-to-plane and point-to-edge distances [[5]](#LOAM)[[6]](#LEGO)  
    
## Continuous-time ICP Factors

- **IntegratedCT_ICPFactor**  
    Continuous Time ICP Factor [[7]](#CTICP)  
- **IntegratedCT_GICPFactor**  
    Continuous Time ICP with GICP's D2D cost [[2]](#GICP)[[7]](#CTICP)



## Bundle Adjustment Factors

- **PlaneEVMFactor and EdgeEVMFactor**  
    Bundle adjustment factor based on Eigenvalue minimization [[8]](#BA_EVM)
- LsqBundleAdjustmentFactor  
    Bundle adjustment factor based on EVM and EF optimal condition satisfaction [[9]](#BA_LSQ)


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