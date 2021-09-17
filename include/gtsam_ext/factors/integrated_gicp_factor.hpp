#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

class IntegratedGICPFactor : public gtsam::NonlinearFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedGICPFactor>;

  IntegratedGICPFactor(gtsam::Key target_key, gtsam::Key source_key, const Frame::ConstPtr& target, const Frame::ConstPtr& source);
  ~IntegratedGICPFactor();

  void set_max_corresponding_distance(double dist) { max_correspondence_distance_sq = dist * dist; }

  virtual size_t dim() const override { return 6; }

  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

  const Eigen::Isometry3d& get_target_pose() const { return target_pose; }

private:
  Eigen::Isometry3d calc_delta(const gtsam::Values& values) const;
  void find_correspondences(const Eigen::Isometry3d& delta) const;

  double evaluate(
    const Eigen::Isometry3d& delta,
    Eigen::Matrix<double, 6, 6>* H_target = nullptr,
    Eigen::Matrix<double, 6, 6>* H_source = nullptr,
    Eigen::Matrix<double, 6, 6>* H_target_source = nullptr,
    Eigen::Matrix<double, 6, 1>* b_target = nullptr,
    Eigen::Matrix<double, 6, 1>* b_source = nullptr) const;

private:
  bool is_binary;
  Eigen::Isometry3d target_pose;

  double max_correspondence_distance_sq;

  struct KdTree;
  std::unique_ptr<KdTree> target_tree;

  // I'm unhappy for having a mutable member...
  mutable std::vector<int> correspondences;
  mutable std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis;

  std::shared_ptr<const Frame> target;
  std::shared_ptr<const Frame> source;
};

}  // namespace gtsam_ext