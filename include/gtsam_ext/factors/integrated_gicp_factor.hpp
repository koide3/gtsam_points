#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_ext {

struct NearestNeighborSearch;

/**
 * @brief Generalized ICP matching cost factor
 * @ref Segal et al., "Generalized-ICP", RSS2005
 */
class IntegratedGICPFactor : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedGICPFactor>;

  IntegratedGICPFactor(gtsam::Key target_key, gtsam::Key source_key, const Frame::ConstPtr& target, const Frame::ConstPtr& source);
  IntegratedGICPFactor(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const Frame::ConstPtr& target,
    const Frame::ConstPtr& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree);

  IntegratedGICPFactor(const gtsam::Pose3& fixed_target_pose, gtsam::Key source_key, const Frame::ConstPtr& target, const Frame::ConstPtr& source);
  IntegratedGICPFactor(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const Frame::ConstPtr& target,
    const Frame::ConstPtr& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree);

  virtual ~IntegratedGICPFactor() override;

  // note: If your GTSAM is built with TBB, linearization is already multi-threaded
  //     : and setting n>1 can rather affect the processing speed
  void set_num_threads(int n) { num_threads = n; }
  void set_max_corresponding_distance(double dist) { max_correspondence_distance_sq = dist * dist; }
  void set_correspondence_update_tolerance(double angle, double trans) {
    correspondence_update_tolerance_rot = angle;
    correspondence_update_tolerance_trans = trans;
  }

  double inlier_fraction() const {
    const int outliers = std::count(correspondences.begin(), correspondences.end(), -1);
    const int inliers = correspondences.size() - outliers;
    return static_cast<double>(inliers) / correspondences.size();
  }

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
  double max_correspondence_distance_sq;

  std::shared_ptr<NearestNeighborSearch> target_tree;

  // I'm unhappy to have mutable members...
  double correspondence_update_tolerance_rot;
  double correspondence_update_tolerance_trans;
  mutable Eigen::Isometry3d last_correspondence_point;
  mutable std::vector<int> correspondences;
  mutable std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis;

  std::shared_ptr<const Frame> target;
  std::shared_ptr<const Frame> source;
};

}  // namespace gtsam_ext