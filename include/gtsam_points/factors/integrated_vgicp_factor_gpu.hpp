// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/factors/nonlinear_factor_gpu.hpp>

struct CUstream_st;

namespace gtsam {
class Pose3;
}

namespace gtsam_points {

class LinearizedSystem6;
class IntegratedVGICPDerivatives;
class TempBufferManager;

/**
 * @brief GPU-accelerated Voxelized GICP matching cost factor
 *        Koide et al., "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration", ICRA2021
 *        Koide et al., "Globally Consistent 3D LiDAR Mapping with GPU-accelerated GICP Matching Cost Factors", RA-L2021
 */
class IntegratedVGICPFactorGPU : public gtsam_points::NonlinearFactorGPU {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedVGICPFactorGPU>;

  /**
   * @brief Create a binary VGICP_GPU factor between target and source poses.
   * @param target_key    Target key
   * @param source_key    Source key
   * @param target        Target voxelmap
   * @param source        Source frame
   * @param stream        CUDA stream
   * @param temp_buffer   CUDA temporary buffer manager
   */
  IntegratedVGICPFactorGPU(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const GaussianVoxelMap::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    CUstream_st* stream = nullptr,
    std::shared_ptr<TempBufferManager> temp_buffer = nullptr);

  /**
   * @brief Create a unary VGICP_GPU factor between a fixed target pose and an active source pose.
   * @param targfixed_target_pose  Fixed target pose
   * @param source_key             Source key
   * @param target                 Target voxelmap
   * @param source                 Source frame
   * @param stream                 CUDA stream
   * @param temp_buffer            CUDA temporary buffer manager
   */
  IntegratedVGICPFactorGPU(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const GaussianVoxelMap::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    CUstream_st* stream = nullptr,
    std::shared_ptr<TempBufferManager> temp_buffer = nullptr);

  virtual ~IntegratedVGICPFactorGPU() override;

  /// @brief Print the factor information.
  virtual void print(const std::string& s = "", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override;

  /// @brief Enable or disable surface orientation validation for correspondence search
  /// @note  To enable surface orientation validation, source frame must have point normals
  void set_enable_surface_validation(bool enable);

  /// @brief Set the threshold values to trigger inlier points update.
  ///        Setting larger values reduces GPU sync but may affect the registration accuracy.
  void set_inlier_update_thresh(double trans, double angle);

  /// @brief  Get the number of inlier points.
  /// @note   This function must be called after the factor is linearized.
  int num_inliers() const;

  /// @brief Get the fraction of inlier points.
  /// @note  This function must be called after the factor is linearized.
  double inlier_fraction() const;

  /// @brief Get the target voxelmap.
  GaussianVoxelMapGPU::ConstPtr get_target() const { return target; }

  /// @brief Get the pose of the fixed target. This function is only valid for unary factors.
  Eigen::Isometry3f get_fixed_target_pose() const { return fixed_target_pose; }

  // forbid copy
  IntegratedVGICPFactorGPU(const IntegratedVGICPFactorGPU&) = delete;
  IntegratedVGICPFactorGPU& operator=(const IntegratedVGICPFactorGPU&) = delete;

  virtual gtsam::NonlinearFactor::shared_ptr clone() const override;

  virtual size_t dim() const override { return 6; }
  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

  virtual size_t linearization_input_size() const override;
  virtual size_t linearization_output_size() const override;
  virtual size_t evaluation_input_size() const override;
  virtual size_t evaluation_output_size() const override;

  virtual void set_linearization_point(const gtsam::Values& values, void* lin_input_cpu) override;
  virtual void issue_linearize(const void* lin_input_cpu, const void* lin_input_gpu, void* lin_output_gpu) override;
  virtual void store_linearized(const void* lin_output_cpu) override;

  virtual void set_evaluation_point(const gtsam::Values& values, void* eval_input_cpu) override;
  virtual void issue_compute_error(
    const void* lin_input_cpu,
    const void* eval_input_cpu,
    const void* lin_input_gpu,
    const void* eval_input_gpu,
    void* eval_output_gpu) override;
  virtual void store_computed_error(const void* eval_output_cpu) override;

  virtual void sync() override;

private:
  Eigen::Isometry3f calc_delta(const gtsam::Values& values) const;

private:
  bool is_binary;
  Eigen::Isometry3f fixed_target_pose;

  GaussianVoxelMapGPU::ConstPtr target;
  PointCloud::ConstPtr source;

  std::unique_ptr<IntegratedVGICPDerivatives> derivatives;

  mutable bool linearized;
  mutable Eigen::Isometry3f linearization_point;

  mutable boost::optional<float> evaluation_result;
  mutable std::unique_ptr<LinearizedSystem6> linearization_result;
};

}  // namespace gtsam_points