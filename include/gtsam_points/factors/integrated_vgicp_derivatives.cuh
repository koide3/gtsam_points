// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

struct CUstream_st;

namespace gtsam_points {

class LinearizedSystem6;
class TempBufferManager;

class IntegratedVGICPDerivatives {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IntegratedVGICPDerivatives(
    const GaussianVoxelMapGPU::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    CUstream_st* ext_stream,
    std::shared_ptr<TempBufferManager> temp_buffer);
  ~IntegratedVGICPDerivatives();

  void set_inlier_update_thresh(double trans, double angle) {
    inlier_update_thresh_trans = trans;
    inlier_update_thresh_angle = angle;
  }

  void set_enable_surface_validation(bool enable) { enable_surface_validation = enable; }

  int get_num_inliers() const { return num_inliers; }

  // synchronized interface
  LinearizedSystem6 linearize(const Eigen::Isometry3f& x);
  double compute_error(const Eigen::Isometry3f& xl, const Eigen::Isometry3f& xe);

  void reset_inliers(const Eigen::Isometry3f& x, const Eigen::Isometry3f* d_x, bool force_update = false);
  void update_inliers(int num_inliers);

  // async interface
  void sync_stream();
  void issue_linearize(const Eigen::Isometry3f* d_x, LinearizedSystem6* d_output);
  void issue_compute_error(const Eigen::Isometry3f* d_xl, const Eigen::Isometry3f* d_xe, float* d_output);

private:
  bool enable_surface_validation;
  double inlier_update_thresh_trans;
  double inlier_update_thresh_angle;

  bool external_stream;
  CUstream_st* stream;
  std::shared_ptr<TempBufferManager> temp_buffer;

  GaussianVoxelMapGPU::ConstPtr target;
  PointCloud::ConstPtr source;

  Eigen::Isometry3f inlier_evaluation_point;
  const Eigen::Isometry3f* inlier_evaluation_point_gpu;

  int num_inliers;
  int* num_inliers_gpu;
  int* source_inliers;
};
}  // namespace gtsam_points