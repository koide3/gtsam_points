// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/ann/kdtree_gpu.hpp>
#include <gtsam_points/types/point_cloud_gpu.hpp>

struct CUstream_st;

namespace gtsam_points {

class LinearizedSystem6;
class TempBufferManager;

class IntegratedGICPDerivatives {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IntegratedGICPDerivatives(
    const PointCloud::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    const KdTreeGPU::ConstPtr& target_tree,
    CUstream_st* ext_stream,
    std::shared_ptr<TempBufferManager> temp_buffer);
  ~IntegratedGICPDerivatives();

  void set_inlier_update_thresh(double trans, double angle) {
    inlier_update_thresh_trans = trans;
    inlier_update_thresh_angle = angle;
  }

  void set_enable_offloading(bool enable) { enable_offloading = enable; }

  void set_enable_surface_validation(bool enable) { enable_surface_validation = enable; }

  int get_num_inliers() const { return num_inliers; }

  void touch_points();

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
  bool enable_offloading;

  bool enable_surface_validation;
  double inlier_update_thresh_trans;
  double inlier_update_thresh_angle;

  bool external_stream;
  CUstream_st* stream;
  std::shared_ptr<TempBufferManager> temp_buffer;

  PointCloud::ConstPtr target;
  PointCloud::ConstPtr source;
  KdTreeGPU::ConstPtr target_tree;

  Eigen::Isometry3f inlier_evaluation_point;
  const Eigen::Isometry3f* inlier_evaluation_point_gpu;

  int num_inliers;
  int* num_inliers_gpu;
  std::pair<int, int>* source_target_correspondences;
};
}  // namespace gtsam_points