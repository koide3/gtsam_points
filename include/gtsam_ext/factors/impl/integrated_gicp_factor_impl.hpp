// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/factors/integrated_gicp_factor.hpp>

#include <nanoflann.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_ext/ann/kdtree.hpp>
#include <gtsam_ext/types/frame_traits.hpp>

namespace gtsam_ext {

template <typename Frame>
IntegratedGICPFactor<Frame>::IntegratedGICPFactor(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const Frame>& target,
  const std::shared_ptr<const Frame>& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree)
: gtsam_ext::IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source) {
  //
  if (!frame::has_points(*target) || !frame::has_covs(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for gicp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_covs(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for gicp" << std::endl;
    abort();
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    const Eigen::Vector4d* target_points = frame::points_ptr(*target);
    if (!target_points) {
      std::cerr << "error: failed to create target kdtree because the target frame doesn't provide points ptr!!" << std::endl;
      abort();
    }
    this->target_tree.reset(new KdTree(target_points, frame::size(*target)));
  }
}

template <typename Frame>
IntegratedGICPFactor<Frame>::IntegratedGICPFactor(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const Frame>& target,
  const std::shared_ptr<const Frame>& source)
: IntegratedGICPFactor(target_key, source_key, target, source, nullptr) {}

template <typename Frame>
IntegratedGICPFactor<Frame>::IntegratedGICPFactor(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const std::shared_ptr<const Frame>& target,
  const std::shared_ptr<const Frame>& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree)
: gtsam_ext::IntegratedMatchingCostFactor(fixed_target_pose, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source) {
  //
  if (!frame::has_points(*target) || !frame::has_covs(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for gicp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_covs(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for gicp" << std::endl;
    abort();
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    const Eigen::Vector4d* target_points = frame::points_ptr(*target);
    if (!target_points) {
      std::cerr << "error: failed to create target kdtree because the target frame doesn't provide points ptr!!" << std::endl;
      abort();
    }
    this->target_tree.reset(new KdTree(target_points, frame::size(*target)));
  }
}

template <typename Frame>
IntegratedGICPFactor<Frame>::IntegratedGICPFactor(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const std::shared_ptr<const Frame>& target,
  const std::shared_ptr<const Frame>& source)
: IntegratedGICPFactor(fixed_target_pose, source_key, target, source, nullptr) {}

template <typename Frame>
IntegratedGICPFactor<Frame>::~IntegratedGICPFactor() {}

template <typename Frame>
void IntegratedGICPFactor<Frame>::update_correspondences(const Eigen::Isometry3d& delta) const {
  bool do_update = true;
  if (correspondences.size() == frame::size(*source) && (correspondence_update_tolerance_trans > 0.0 || correspondence_update_tolerance_rot > 0.0)) {
    Eigen::Isometry3d diff = delta.inverse() * last_correspondence_point;
    double diff_rot = Eigen::AngleAxisd(diff.linear()).angle();
    double diff_trans = diff.translation().norm();
    if (diff_rot < correspondence_update_tolerance_rot && diff_trans < correspondence_update_tolerance_trans) {
      do_update = false;
    }
  }

  correspondences.resize(frame::size(*source));
  mahalanobis.resize(frame::size(*source));

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < frame::size(*source); i++) {
    if (do_update) {
      Eigen::Vector4d pt = delta * frame::point(*source, i);

      size_t k_index = -1;
      double k_sq_dist = -1;
      size_t num_found = target_tree->knn_search(pt.data(), 1, &k_index, &k_sq_dist);
      correspondences[i] = k_sq_dist < max_correspondence_distance_sq ? k_index : -1;
    }

    if (correspondences[i] < 0) {
      mahalanobis[i].setIdentity();
    } else {
      const auto& target_cov = frame::cov(*target, correspondences[i]);
      Eigen::Matrix4d RCR = (target_cov + delta.matrix() * frame::cov(*source, i) * delta.matrix().transpose());
      RCR(3, 3) = 1.0;
      mahalanobis[i] = RCR.inverse();
      mahalanobis[i](3, 3) = 0.0;
    }
  }

  last_correspondence_point = delta;
}

template <typename Frame>
double IntegratedGICPFactor<Frame>::evaluate(
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

  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs_target;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs_source;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs_target_source;
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs_target;
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs_source;

  if (H_target && H_source && H_target_source && b_target && b_source) {
    Hs_target.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_target_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    bs_target.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
    bs_source.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
  }

#pragma omp parallel for num_threads(num_threads) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < frame::size(*source); i++) {
    const int target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    const auto& mean_A = frame::point(*source, i);
    const auto& cov_A = frame::cov(*source, i);

    const auto& mean_B = frame::point(*target, target_index);
    const auto& cov_B = frame::cov(*target, target_index);

    Eigen::Vector4d transed_mean_A = delta * mean_A;
    Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += 0.5 * error.transpose() * mahalanobis[i] * error;

    if (Hs_target.empty()) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> J_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_target.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_mean_A.head<3>());
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_source.block<3, 3>(0, 0) = delta.linear() * gtsam::SO3::Hat(mean_A.template head<3>());
    J_source.block<3, 3>(0, 3) = -delta.linear();

    int thread_num = 0;
#ifdef _OPENMP
    thread_num = omp_get_thread_num();
#endif

    Hs_target[thread_num] += J_target.transpose() * mahalanobis[i] * J_target;
    Hs_source[thread_num] += J_source.transpose() * mahalanobis[i] * J_source;
    Hs_target_source[thread_num] += J_target.transpose() * mahalanobis[i] * J_source;
    bs_target[thread_num] += J_target.transpose() * mahalanobis[i] * error;
    bs_source[thread_num] += J_source.transpose() * mahalanobis[i] * error;
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

}  // namespace gtsam_ext
