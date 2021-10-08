#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

struct NearestNeighborSearch;

class IntegratedCT_ICPFactor : public gtsam::NonlinearFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedCT_ICPFactor>;

  IntegratedCT_ICPFactor(
    gtsam::Key source_t0_key,
    gtsam::Key source_t1_key,
    const gtsam_ext::Frame::ConstPtr& target,
    const gtsam_ext::Frame::ConstPtr& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree);

  IntegratedCT_ICPFactor(gtsam::Key source_t0_key, gtsam::Key source_t1_key, const gtsam_ext::Frame::ConstPtr& target, const gtsam_ext::Frame::ConstPtr& source);

  virtual ~IntegratedCT_ICPFactor() override;

  virtual size_t dim() const override { return 6; }

  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> deskewed_source_points(const gtsam::Values& values);

protected:
  virtual void update_poses(const gtsam::Values& values) const;
  virtual void update_correspondences() const;

protected:
  int num_threads;
  double max_correspondence_distance_sq;

  std::shared_ptr<NearestNeighborSearch> target_tree;

  std::vector<double> time_table;
  mutable std::vector<gtsam::Pose3, Eigen::aligned_allocator<gtsam::Pose3>> source_poses;
  mutable std::vector<gtsam::Matrix6, Eigen::aligned_allocator<gtsam::Matrix6>> pose_derivatives_t0;
  mutable std::vector<gtsam::Matrix6, Eigen::aligned_allocator<gtsam::Matrix6>> pose_derivatives_t1;

  std::vector<int> time_indices;
  mutable std::vector<int> correspondences;

  std::shared_ptr<const Frame> target;
  std::shared_ptr<const Frame> source;
};

}  // namespace gtsam_ext
