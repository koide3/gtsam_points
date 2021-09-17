#include <gtsam_ext/factors/integrated_gicp_factor.hpp>

#include <nanoflann.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>

namespace gtsam_ext {

struct IntegratedGICPFactor::KdTree {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 3>;

  KdTree(int num_points, const Eigen::Vector4d* points) : num_points(num_points), points(points), index(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    index.buildIndex();
  }

  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

public:
  const int num_points;
  const Eigen::Vector4d* points;

  Index index;
};

IntegratedGICPFactor::IntegratedGICPFactor(gtsam::Key target_key, gtsam::Key source_key, const Frame::ConstPtr& target, const Frame::ConstPtr& source)
: gtsam::NonlinearFactor(gtsam::cref_list_of<2>(target_key)(source_key)),
  is_binary(true),
  target_pose(Eigen::Isometry3d::Identity()),
  target(target),
  source(source),
  max_correspondence_distance_sq(1.0) {
  //
  if (!target->points || !source->points) {
    std::cerr << "error: target or source points has not been allocated!!" << std::endl;
    abort();
  }

  target_tree.reset(new KdTree(target->num_points, target->points));
}

IntegratedGICPFactor::~IntegratedGICPFactor() {}

double IntegratedGICPFactor::error(const gtsam::Values& values) const {
  Eigen::Isometry3d delta = calc_delta(values);

  if (correspondences.size() != source->size()) {
    find_correspondences(delta);
  }

  double error = evaluate(delta);
  return error;
}

boost::shared_ptr<gtsam::GaussianFactor> IntegratedGICPFactor::linearize(const gtsam::Values& values) const {
  Eigen::Isometry3d delta = calc_delta(values);
  find_correspondences(delta);

  Eigen::Matrix<double, 6, 6> H_target, H_source, H_target_source;
  Eigen::Matrix<double, 6, 1> b_target, b_source;
  double error = evaluate(delta, &H_target, &H_source, &H_target_source, &b_target, &b_source);

  gtsam::HessianFactor::shared_ptr factor;

  if (is_binary) {
    factor.reset(new gtsam::HessianFactor(keys()[0], keys()[1], H_target, H_target_source, -b_target, H_source, -b_source, 0.0));
  } else {
    factor.reset(new gtsam::HessianFactor(keys()[0], H_source, -b_source, 0.0));
  }

  return factor;
}

Eigen::Isometry3d IntegratedGICPFactor::calc_delta(const gtsam::Values& values) const {
  if (is_binary) {
    gtsam::Pose3 target_pose = values.at<gtsam::Pose3>(keys()[0]);
    gtsam::Pose3 source_pose = values.at<gtsam::Pose3>(keys()[1]);
    Eigen::Isometry3d delta((target_pose.inverse() * source_pose).matrix());
    return delta;
  } else {
    gtsam::Pose3 target_pose(this->target_pose.matrix());
    gtsam::Pose3 source_pose = values.at<gtsam::Pose3>(keys()[1]);
    Eigen::Isometry3d delta((target_pose.inverse() * source_pose).matrix());
    return delta;
  }
}

void IntegratedGICPFactor::find_correspondences(const Eigen::Isometry3d& delta) const {
  correspondences.resize(source->size());
  mahalanobis.resize(source->size());

  for (int i = 0; i < source->size(); i++) {
    Eigen::Vector4d pt = delta * source->points[i];

    size_t k_index = -1;
    double k_sq_dist = -1;
    target_tree->index.knnSearch(pt.data(), 1, &k_index, &k_sq_dist);

    if (k_sq_dist > max_correspondence_distance_sq) {
      correspondences[i] = -1;
      mahalanobis[i].setIdentity();
    } else {
      correspondences[i] = k_index;

      Eigen::Matrix4d RCR = (target->covs[k_index] + delta.matrix() * source->covs[i] * delta.matrix().transpose());
      RCR(3, 3) = 1.0;
      mahalanobis[i] = RCR.inverse();
      mahalanobis[i](3, 3) = 0.0;
    }
  }
}

double IntegratedGICPFactor::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //
  double sum_errors = 0.0;
  if (H_target && H_source && H_target_source && b_target && b_source) {
    H_target->setZero();
    H_source->setZero();
    H_target_source->setZero();
    b_target->setZero();
    b_source->setZero();
  }

  for (int i = 0; i < source->size(); i++) {
    int target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    const auto& mean_A = source->points[i];
    const auto& cov_A = source->covs[i];

    const auto& mean_B = target->points[target_index];
    const auto& cov_B = target->covs[target_index];

    Eigen::Vector4d transed_mean_A = delta * mean_A;
    Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += 0.5 * error.transpose() * mahalanobis[i] * error;

    if (!H_target || !H_source || !H_target_source || !b_target || !b_source) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> J_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_target.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_mean_A.head<3>());
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_source.block<3, 3>(0, 0) = delta.linear() * gtsam::SO3::Hat(mean_A.head<3>());
    J_source.block<3, 3>(0, 3) = -delta.linear();

    (*H_target) += J_target.transpose() * mahalanobis[i] * J_target;
    (*H_source) += J_source.transpose() * mahalanobis[i] * J_source;
    (*H_target_source) += J_target.transpose() * mahalanobis[i] * J_source;
    (*b_target) += J_target.transpose() * mahalanobis[i] * error;
    (*b_source) += J_source.transpose() * mahalanobis[i] * error;
  }

  return sum_errors;
}

}  // namespace gtsam_ext
