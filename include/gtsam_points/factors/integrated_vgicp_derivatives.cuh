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
  virtual ~IntegratedVGICPDerivatives();

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
  virtual void issue_linearize(const Eigen::Isometry3f* d_x, LinearizedSystem6* d_output);
  virtual void issue_compute_error(const Eigen::Isometry3f* d_xl, const Eigen::Isometry3f* d_xe, float* d_output);

protected:
  /// @brief Get the CUDA stream used by this derivatives instance.
  CUstream_st* cuda_stream() const { return stream; }

  /**
   * @brief Apply a custom device transform to each valid VGICP correspondence after standard linearization and reduce the results.
   *
   * Transform must be device-callable with the following signature:
   * @code
   * Result operator()(const thrust::pair<int, int>& correspondence, const LinearizedSystem6& linearized) const;
   * @endcode
   * Invalid correspondences are mapped directly to identity and are never passed to Transform.
   * Reduction must be device-callable and identity must be its neutral element.
   * Transform must preserve LinearizedSystem6::num_inliers when its result contains the system used by
   * IntegratedVGICPFactorGPU, because the count is also used to maintain the inlier index buffer.
   */
  template <typename Result, typename Transform, typename Reduction>
  void issue_linearize_transform_reduce(
    const Eigen::Isometry3f* d_x,
    Result* d_output,
    Transform transform,
    Reduction reduction,
    const Result& identity);

  /**
   * @brief Additive convenience overload of issue_linearize_transform_reduce().
   */
  template <typename Result, typename Transform>
  void issue_linearize_transform_reduce(
    const Eigen::Isometry3f* d_x,
    Result* d_output,
    Transform transform,
    const Result& identity);

  /**
   * @brief Apply a custom device transform to each valid VGICP correspondence error and reduce the results.
   *
   * Transform must be device-callable with the following signature:
   * @code
   * Result operator()(const thrust::pair<int, int>& correspondence, float error) const;
   * @endcode
   * Invalid correspondences are mapped directly to identity and are never passed to Transform.
   */
  template <typename Result, typename Transform, typename Reduction>
  void issue_compute_error_transform_reduce(
    const Eigen::Isometry3f* d_xl,
    const Eigen::Isometry3f* d_xe,
    Result* d_output,
    Transform transform,
    Reduction reduction,
    const Result& identity);

  /**
   * @brief Additive convenience overload of issue_compute_error_transform_reduce().
   */
  template <typename Result, typename Transform>
  void issue_compute_error_transform_reduce(
    const Eigen::Isometry3f* d_xl,
    const Eigen::Isometry3f* d_xe,
    Result* d_output,
    Transform transform,
    const Result& identity);

private:
  bool enable_offloading;

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

#if defined(__CUDACC__)
#include <gtsam_points/factors/impl/integrated_vgicp_derivatives_impl.cuh>
#endif
