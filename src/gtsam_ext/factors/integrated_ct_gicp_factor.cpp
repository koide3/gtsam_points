#include <gtsam_ext/factors/integrated_ct_gicp_factor.hpp>

#include <gtsam/linear/HessianFactor.h>

namespace gtsam_ext {

IntegratedCT_GICPFactor::IntegratedCT_GICPFactor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const gtsam_ext::Frame::ConstPtr& target,
  const gtsam_ext::Frame::ConstPtr& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree)
: IntegratedCT_ICPFactor(source_t0_key, source_t1_key, target, source, target_tree) {
  //
  if (!target->covs || !source->covs) {
    std::cerr << "error: target or source doesn't have covs!!" << std::endl;
    abort();
  }
}

IntegratedCT_GICPFactor::IntegratedCT_GICPFactor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const gtsam_ext::Frame::ConstPtr& target,
  const gtsam_ext::Frame::ConstPtr& source)
: IntegratedCT_GICPFactor(source_t0_key, source_t1_key, target, source, nullptr) {}

IntegratedCT_GICPFactor::~IntegratedCT_GICPFactor() {}

double IntegratedCT_GICPFactor::error(const gtsam::Values& values) const {
  update_poses(values);
  if (correspondences.size() != source->size()) {
    update_correspondences();
  }

  double sum_errors = 0.0;
  for (int i = 0; i < source->size(); i++) {
    const int target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    const int time_index = time_indices[i];
    const auto& pose = source_poses[time_index];

    const auto& source_pt = source->points[i];
    const auto& target_pt = target->points[target_index];
    const auto& target_normal = target->normals[i];

    Eigen::Vector4d transed_source_pt = pose.matrix() * source_pt;
    Eigen::Vector4d error = transed_source_pt - target_pt;

    sum_errors += 0.5 * error.transpose() * mahalanobis[i] * error;
  }

  return sum_errors;
}

boost::shared_ptr<gtsam::GaussianFactor> IntegratedCT_GICPFactor::linearize(const gtsam::Values& values) const {
  update_poses(values);
  update_correspondences();

  double sum_errors = 0.0;
  gtsam::Matrix6 H_00 = gtsam::Matrix6::Zero();
  gtsam::Matrix6 H_01 = gtsam::Matrix6::Zero();
  gtsam::Matrix6 H_11 = gtsam::Matrix6::Zero();
  gtsam::Vector6 b_0 = gtsam::Vector6::Zero();
  gtsam::Vector6 b_1 = gtsam::Vector6::Zero();

  for (int i = 0; i < source->size(); i++) {
    const int target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    const int time_index = time_indices[i];

    const auto& pose = source_poses[time_index];
    const auto& H_pose_0 = pose_derivatives_t0[time_index];
    const auto& H_pose_1 = pose_derivatives_t1[time_index];

    const auto& source_pt = source->points[i];
    const auto& target_pt = target->points[target_index];
    const auto& target_normal = target->normals[i];

    gtsam::Matrix36 H_transed_pose;
    gtsam::Point3 transed_source_pt = pose.transformFrom(source_pt.head<3>(), H_transed_pose);

    Eigen::Vector4d error = Eigen::Vector4d::Zero();
    error.head<3>() = transed_source_pt - target_pt.head<3>();

    gtsam::Matrix46 H_error_pose = gtsam::Matrix46::Zero();
    H_error_pose.block<3, 6>(0, 0) = H_transed_pose;

    gtsam::Matrix46 H_0 = H_error_pose * H_pose_0;
    gtsam::Matrix46 H_1 = H_error_pose * H_pose_1;

    sum_errors += 0.5 * error.transpose() * mahalanobis[i] * error;
    H_00 += H_0.transpose() * mahalanobis[i] * H_0;
    H_11 += H_1.transpose() * mahalanobis[i] * H_1;
    H_01 += H_0.transpose() * mahalanobis[i] * H_1;
    b_0 += H_0.transpose() * mahalanobis[i] * error;
    b_1 += H_1.transpose() * mahalanobis[i] * error;
  }

  auto factor = gtsam::HessianFactor::shared_ptr(new gtsam::HessianFactor(keys_[0], keys_[1], H_00, H_01, -b_0, H_11, -b_1, sum_errors));
  return factor;
}

void IntegratedCT_GICPFactor::update_correspondences() const {
  IntegratedCT_ICPFactor::update_correspondences();

  mahalanobis.resize(source->size());
  for (int i = 0; i < source->size(); i++) {
    if (correspondences[i] < 0) {
      mahalanobis[i].setZero();
    } else {
      const int time_index = time_indices[i];
      const Eigen::Matrix4d pose = source_poses[time_index].matrix();

      const int target_index = correspondences[i];
      const auto& cov_A = source->covs[i];
      const auto& cov_B = target->covs[target_index];
      Eigen::Matrix4d RCR = (cov_B + pose * cov_A * pose.transpose());
      RCR(3, 3) = 1.0;

      mahalanobis[i] = RCR.inverse();
      mahalanobis[i](3, 3) = 0.0;
    }
  }
}
}  // namespace gtsam_ext