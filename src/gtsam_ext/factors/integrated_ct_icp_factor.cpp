#include <gtsam_ext/factors/integrated_ct_icp_factor.hpp>

#include <gtsam/linear/HessianFactor.h>
#include <gtsam_ext/ann/kdtree.hpp>

#include <gtsam_ext/util/expressions.hpp>

namespace gtsam_ext {

IntegratedCT_ICPFactor::IntegratedCT_ICPFactor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const gtsam_ext::Frame::ConstPtr& target,
  const gtsam_ext::Frame::ConstPtr& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree)
: gtsam::NonlinearFactor(gtsam::cref_list_of<2>(source_t0_key)(source_t1_key)),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  target(target),
  source(source) {
  //
  if (!target->points || !source->points) {
    std::cerr << "error: target or source points has not been allocated!!" << std::endl;
    abort();
  }

  if (!source->times) {
    std::cerr << "error: source cloud doesn't have timestamps!!" << std::endl;
    abort();
  }

  time_table.reserve(source->size() / 10);
  time_indices.reserve(source->size());

  const double time_eps = 1e-3;
  for (int i = 0; i < source->size(); i++) {
    const double t = source->times[i];
    if (time_table.empty() || t - time_table.back() > time_eps) {
      time_table.push_back(t);
    }
    time_indices.push_back(time_table.size() - 1);
  }

  for (auto& t : time_table) {
    t = t / time_table.back();
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    this->target_tree.reset(new gtsam_ext::KdTree(target->points, target->size()));
  }
}

IntegratedCT_ICPFactor::IntegratedCT_ICPFactor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const gtsam_ext::Frame::ConstPtr& target,
  const gtsam_ext::Frame::ConstPtr& source)
: IntegratedCT_ICPFactor(source_t0_key, source_t1_key, target, source, nullptr) {}

IntegratedCT_ICPFactor::~IntegratedCT_ICPFactor() {}

double IntegratedCT_ICPFactor::error(const gtsam::Values& values) const {
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

    gtsam::Point3 transed_source_pt = pose.transformFrom(source_pt.head<3>());
    gtsam::Point3 residual = transed_source_pt - target_pt.head<3>();
    double error = gtsam::dot(residual, target_normal.head<3>());

    sum_errors += 0.5 * error * error;
  }

  return sum_errors;
}

boost::shared_ptr<gtsam::GaussianFactor> IntegratedCT_ICPFactor::linearize(const gtsam::Values& values) const {
  if (!target->normals) {
    std::cerr << "error: target cloud doesn't have normals!!" << std::endl;
    abort();
  }

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

    gtsam::Point3 residual = transed_source_pt - target_pt.head<3>();

    gtsam::Matrix13 H_error_transed;
    double error = gtsam::dot(residual, target_normal.head<3>(), H_error_transed);

    gtsam::Matrix16 H_error_pose = H_error_transed * H_transed_pose;
    gtsam::Matrix16 H_0 = H_error_pose * H_pose_0;
    gtsam::Matrix16 H_1 = H_error_pose * H_pose_1;

    sum_errors += 0.5 * error * error;
    H_00 += H_0.transpose() * H_0;
    H_11 += H_1.transpose() * H_1;
    H_01 += H_0.transpose() * H_1;
    b_0 += H_0.transpose() * error;
    b_1 += H_1.transpose() * error;
  }

  auto factor = gtsam::HessianFactor::shared_ptr(new gtsam::HessianFactor(keys_[0], keys_[1], H_00, H_01, -b_0, H_11, -b_1, sum_errors));
  return factor;
}

void IntegratedCT_ICPFactor::update_poses(const gtsam::Values& values) const {
  gtsam::Pose3 pose0 = values.at<gtsam::Pose3>(keys_[0]);
  gtsam::Pose3 pose1 = values.at<gtsam::Pose3>(keys_[1]);

  gtsam::Matrix6 H_delta_0, H_delta_1;
  gtsam::Pose3 delta = pose0.between(pose1, H_delta_0, H_delta_1);

  gtsam::Matrix6 H_vel_delta;
  gtsam::Vector6 vel = gtsam::Pose3::Logmap(delta, H_vel_delta);

  source_poses.resize(time_table.size());
  pose_derivatives_t0.resize(time_table.size());
  pose_derivatives_t1.resize(time_table.size());

  for (int i = 0; i < time_table.size(); i++) {
    const double t = time_table[i];

    gtsam::Matrix6 H_inc_vel;
    gtsam::Pose3 inc = gtsam::Pose3::Expmap(t * vel, H_inc_vel);

    gtsam::Matrix6 H_pose_0_a, H_pose_inc;
    source_poses[i] = pose0.compose(inc, H_pose_0_a, H_pose_inc);

    gtsam::Matrix6 H_pose_delta = H_pose_inc * H_inc_vel * t * H_vel_delta;

    pose_derivatives_t0[i] = H_pose_0_a + H_pose_delta * H_delta_0;
    pose_derivatives_t1[i] = H_pose_delta * H_delta_1;
  }
}

void IntegratedCT_ICPFactor::update_correspondences() const {
  correspondences.resize(source->size());

  for (int i = 0; i < source->size(); i++) {
    const int time_index = time_indices[i];

    const auto& pt = source->points[i];
    gtsam::Point3 transed_pt = source_poses[time_index] * pt.head<3>();

    size_t k_index;
    double k_sq_dist;
    size_t num_found = target_tree->knn_search(transed_pt.data(), 1, &k_index, &k_sq_dist);

    if (num_found == 0 || k_sq_dist > max_correspondence_distance_sq) {
      correspondences[i] = -1;
    } else {
      correspondences[i] = k_index;
    }
  }
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> IntegratedCT_ICPFactor::deskewed_source_points(const gtsam::Values& values) {
  update_poses(values);

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> deskewed(source->size());
  for (int i = 0; i < source->size(); i++) {
    const int time_index = time_indices[i];
    const auto& pose = source_poses[time_index];
    deskewed[i] = pose * source->points[i].head<3>();
  }

  return deskewed;
}

}  // namespace gtsam_ext