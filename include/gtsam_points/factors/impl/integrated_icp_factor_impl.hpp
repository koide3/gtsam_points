// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_icp_factor.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/ann/kdtree2.hpp>

#include <gtsam_points/types/frame_traits.hpp>

namespace gtsam_points {

template <typename TargetFrame, typename SourceFrame>
IntegratedICPFactor_<TargetFrame, SourceFrame>::IntegratedICPFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree,
  bool use_point_to_plane)
: gtsam_points::IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  use_point_to_plane(use_point_to_plane),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source) {
  //
  if (!frame::has_points(*target) || (use_point_to_plane && !frame::has_normals(*target))) {
    std::cerr << "error: target frame doesn't have required attributes for icp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for icp" << std::endl;
    abort();
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    this->target_tree.reset(new KdTree2<TargetFrame>(target));
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedICPFactor_<TargetFrame, SourceFrame>::IntegratedICPFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  bool use_point_to_plane)
: gtsam_points::IntegratedICPFactor_<TargetFrame, SourceFrame>(target_key, source_key, target, source, nullptr, use_point_to_plane) {}

template <typename TargetFrame, typename SourceFrame>
IntegratedICPFactor_<TargetFrame, SourceFrame>::IntegratedICPFactor_(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree,
  bool use_point_to_plane)
: gtsam_points::IntegratedMatchingCostFactor(fixed_target_pose, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  use_point_to_plane(use_point_to_plane),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source) {
  //
  if (!frame::has_points(*target) || (use_point_to_plane && !frame::has_normals(*target))) {
    std::cerr << "error: target frame doesn't have required attributes for icp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for icp" << std::endl;
    abort();
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    this->target_tree.reset(new KdTree2<TargetFrame>(target));
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedICPFactor_<TargetFrame, SourceFrame>::IntegratedICPFactor_(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  bool use_point_to_plane)
: gtsam_points::IntegratedICPFactor_<TargetFrame, SourceFrame>(fixed_target_pose, source_key, target, source, nullptr, use_point_to_plane) {}

template <typename TargetFrame, typename SourceFrame>
IntegratedICPFactor_<TargetFrame, SourceFrame>::~IntegratedICPFactor_() {}

template <typename TargetFrame, typename SourceFrame>
void IntegratedICPFactor_<TargetFrame, SourceFrame>::update_correspondences(const Eigen::Isometry3d& delta) const {
  if (correspondences.size() == frame::size(*source) && (correspondence_update_tolerance_trans > 0.0 || correspondence_update_tolerance_rot > 0.0)) {
    Eigen::Isometry3d diff = delta.inverse() * last_correspondence_point;
    double diff_rot = Eigen::AngleAxisd(diff.linear()).angle();
    double diff_trans = diff.translation().norm();
    if (diff_rot < correspondence_update_tolerance_rot && diff_trans < correspondence_update_tolerance_trans) {
      return;
    }
  }

  correspondences.resize(frame::size(*source));

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < frame::size(*source); i++) {
    Eigen::Vector4d pt = delta * frame::point(*source, i);

    size_t k_index = -1;
    double k_sq_dist = -1;
    size_t num_found = target_tree->knn_search(pt.data(), 1, &k_index, &k_sq_dist);

    if (num_found == 0 || k_sq_dist > max_correspondence_distance_sq) {
      correspondences[i] = -1;
    } else {
      correspondences[i] = k_index;
    }
  }

  last_correspondence_point = delta;
}

template <typename TargetFrame, typename SourceFrame>
double IntegratedICPFactor_<TargetFrame, SourceFrame>::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //
  if (correspondences.size() != frame::size(*source)) {
    update_correspondences(delta);
  }

  //
  double sum_errors = 0.0;

  std::vector<Eigen::Matrix<double, 6, 6>> Hs_target;
  std::vector<Eigen::Matrix<double, 6, 6>> Hs_source;
  std::vector<Eigen::Matrix<double, 6, 6>> Hs_target_source;
  std::vector<Eigen::Matrix<double, 6, 1>> bs_target;
  std::vector<Eigen::Matrix<double, 6, 1>> bs_source;

  if (H_target && H_source && H_target_source && b_target && b_source) {
    Hs_target.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_target_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    bs_target.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
    bs_source.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
  }

#pragma omp parallel for num_threads(num_threads) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < frame::size(*source); i++) {
    long target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    const auto& mean_A = frame::point(*source, i);
    const auto& mean_B = frame::point(*target, target_index);

    Eigen::Vector4d transed_mean_A = delta * mean_A;
    Eigen::Vector4d error = mean_B - transed_mean_A;

    if (use_point_to_plane) {
      const auto& normal_B = frame::normal(*target, target_index);
      error = normal_B.array() * error.array();
    }

    sum_errors += 0.5 * error.transpose() * error;

    if (Hs_target.empty()) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> J_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_target.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_mean_A.head<3>());
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_source.block<3, 3>(0, 0) = delta.linear() * gtsam::SO3::Hat(mean_A.template head<3>());
    J_source.block<3, 3>(0, 3) = -delta.linear();

    if (use_point_to_plane) {
      const auto& normal_B = frame::normal(*target, target_index);
      J_target = normal_B.asDiagonal() * J_target;
      J_source = normal_B.asDiagonal() * J_source;
    }

    int thread_num = 0;
#ifdef _OPENMP
    thread_num = omp_get_thread_num();
#endif

    Hs_target[thread_num] += J_target.transpose() * J_target;
    Hs_source[thread_num] += J_source.transpose() * J_source;
    Hs_target_source[thread_num] += J_target.transpose() * J_source;
    bs_target[thread_num] += J_target.transpose() * error;
    bs_source[thread_num] += J_source.transpose() * error;
  }

  if (H_target && H_source && H_target_source && b_target && b_source) {
    H_target->setZero();
    H_source->setZero();
    H_target_source->setZero();
    b_target->setZero();
    b_source->setZero();

    for (int i = 0; i < num_threads; i++) {
      (*H_target) += Hs_target[i];
      (*H_source) += Hs_source[i];
      (*H_target_source) += Hs_target_source[i];
      (*b_target) += bs_target[i];
      (*b_source) += bs_source[i];
    }
  }

  return sum_errors;
}

}  // namespace gtsam_points
