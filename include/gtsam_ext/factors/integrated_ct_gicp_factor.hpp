// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_ext/factors/integrated_ct_icp_factor.hpp>

namespace gtsam_ext {

/**
 * @brief Continuous Time ICP with GICP's D2D cost
 * @ref Bellenbach et al., "CT-ICP: Real-time Elastic LiDAR Odometry with Loop Closure", 2021
 * @ref Segal et al., "Generalized-ICP", RSS2005
 */
class IntegratedCT_GICPFactor : public IntegratedCT_ICPFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedCT_GICPFactor>;

  /**
   * @brief Constructor
   * @param source_t0_key   Key of the source pose at the beginning of the scan
   * @param source_t1_key   Key of the source pose at the end of the scan
   * @param target          Target point cloud
   * @param source          Source point cloud
   * @param target_tree     NN search for the target point cloud
   */
  IntegratedCT_GICPFactor(
    gtsam::Key source_t0_key,
    gtsam::Key source_t1_key,
    const gtsam_ext::Frame::ConstPtr& target,
    const gtsam_ext::Frame::ConstPtr& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree);

  /**
   * @brief Constructor
   * @param source_t0_key   Key of the source pose at the beginning of the scan
   * @param source_t1_key   Key of the source pose at the end of the scan
   * @param target          Target point cloud
   * @param source          Source point cloud
   */
  IntegratedCT_GICPFactor(gtsam::Key source_t0_key, gtsam::Key source_t1_key, const gtsam_ext::Frame::ConstPtr& target, const gtsam_ext::Frame::ConstPtr& source);

  virtual ~IntegratedCT_GICPFactor() override;

  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

  void set_num_threads(int n) { num_threads = n; }

protected:
  int num_threads;

  virtual void update_correspondences() const override;

  mutable std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis;
};

}  // namespace gtsam_ext
