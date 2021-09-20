#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_matching_cost_factor.hpp>

namespace gtsam_ext {

class IntegratedGICPFactor : public gtsam_ext::IntegratedMatchingCostFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedGICPFactor>;

  IntegratedGICPFactor(gtsam::Key target_key, gtsam::Key source_key, const Frame::ConstPtr& target, const Frame::ConstPtr& source);
  virtual ~IntegratedGICPFactor() override;

  // note: If your GTSAM is built with TBB, linearization is already multi-threaded
  //     : and setting n>1 can rather affect the processing speed  void set_num_threads(int n) { num_threads = n; }
  void set_num_threads(int n) { num_threads = n; }
  void set_max_corresponding_distance(double dist) { max_correspondence_distance_sq = dist * dist; }

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

  struct KdTree;
  std::unique_ptr<KdTree> target_tree;

  // I'm unhappy to have mutable members...
  mutable std::vector<int> correspondences;
  mutable std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis;

  std::shared_ptr<const Frame> target;
  std::shared_ptr<const Frame> source;
};

}  // namespace gtsam_ext