#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/types/voxelized_frame.hpp>
#include <gtsam_ext/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_ext {

struct GaussianVoxel;

/**
 * @brief Voxelized GICP matching cost factor
 * @ref Koide et al., "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration", ICRA2021
 * @ref Koide et al., "Globally Consistent 3D LiDAR Mapping with GPU-accelerated GICP Matching Cost Factors", RA-L2021
 */
class IntegratedVGICPFactor : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedVGICPFactor>;

  IntegratedVGICPFactor(gtsam::Key target_key, gtsam::Key source_key, const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source);
  virtual ~IntegratedVGICPFactor() override;

  // note: If your GTSAM is built with TBB, linearization is already multi-threaded
  //     : and setting n>1 can rather affect the processing speed
  void set_num_threads(int n) { num_threads = n; }

private:
  virtual void update_correspondences(const Eigen::Isometry3d& delta) const override;

  virtual double evaluate(
    const Eigen::Isometry3d& delta,
    Eigen::Matrix<double, 6, 6>* H_target = nullptr,
    Eigen::Matrix<double, 6, 6>* H_source = nullptr,
    Eigen::Matrix<double, 6, 6>* H_target_source = nullptr,
    Eigen::Matrix<double, 6, 1>* b_target = nullptr,
    Eigen::Matrix<double, 6, 1>* b_source = nullptr) const override;

private:
  int num_threads;

  // I'm unhappy to have mutable members...
  mutable std::vector<std::shared_ptr<const GaussianVoxel>> correspondences;
  mutable std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis;

  std::shared_ptr<const VoxelizedFrame> target;
  std::shared_ptr<const Frame> source;
};

}  // namespace gtsam_ext