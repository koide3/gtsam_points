// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/factors/integrated_ct_gicp_factor.hpp>

#include <gtsam/linear/HessianFactor.h>
#include <gtsam_ext/ann/nearest_neighbor_search.hpp>

namespace gtsam_ext {

IntegratedCT_GICPFactor::IntegratedCT_GICPFactor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const gtsam_ext::Frame::ConstPtr& target,
  const gtsam_ext::Frame::ConstPtr& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree)
: IntegratedCT_ICPFactor(source_t0_key, source_t1_key, target, source, target_tree),
  num_threads(1) {
  //
  if (!target->has_points() || !target->has_covs()) {
    std::cerr << "error: target frame doesn't have required attributes for ct_gicp" << std::endl;
    abort();
  }

  if (!source->has_points() || !source->has_covs()) {
    std::cerr << "error: source frame doesn't have required attributes for ct_gicp" << std::endl;
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
#pragma omp parallel for reduction(+ : sum_errors) schedule(guided, 8) num_threads(num_threads)
  for (int i = 0; i < source->size(); i++) {
    const int target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    const int time_index = time_indices[i];
    const Eigen::Isometry3d pose(source_poses[time_index].matrix());

    const auto& source_pt = source->points[i];
    const auto& target_pt = target->points[target_index];
    const auto& target_normal = target->normals[i];

    Eigen::Vector4d transed_source_pt = pose * source_pt;
    Eigen::Vector4d error = transed_source_pt - target_pt;

    sum_errors += 0.5 * error.transpose() * mahalanobis[i] * error;
  }

  return sum_errors;
}

boost::shared_ptr<gtsam::GaussianFactor> IntegratedCT_GICPFactor::linearize(const gtsam::Values& values) const {
  update_poses(values);
  update_correspondences();

  double sum_errors = 0.0;
  std::vector<gtsam::Matrix6, Eigen::aligned_allocator<gtsam::Matrix6>> Hs_00(num_threads, gtsam::Matrix6::Zero());
  std::vector<gtsam::Matrix6, Eigen::aligned_allocator<gtsam::Matrix6>> Hs_01(num_threads, gtsam::Matrix6::Zero());
  std::vector<gtsam::Matrix6, Eigen::aligned_allocator<gtsam::Matrix6>> Hs_11(num_threads, gtsam::Matrix6::Zero());
  std::vector<gtsam::Vector6, Eigen::aligned_allocator<gtsam::Vector6>> bs_0(num_threads, gtsam::Vector6::Zero());
  std::vector<gtsam::Vector6, Eigen::aligned_allocator<gtsam::Vector6>> bs_1(num_threads, gtsam::Vector6::Zero());

  gtsam::Matrix6 H_00 = gtsam::Matrix6::Zero();
  gtsam::Matrix6 H_01 = gtsam::Matrix6::Zero();
  gtsam::Matrix6 H_11 = gtsam::Matrix6::Zero();
  gtsam::Vector6 b_0 = gtsam::Vector6::Zero();
  gtsam::Vector6 b_1 = gtsam::Vector6::Zero();

#pragma omp parallel for reduction(+ : sum_errors) schedule(guided, 8) num_threads(num_threads)
  for (int i = 0; i < source->size(); i++) {
    const int target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    const int time_index = time_indices[i];

    const Eigen::Isometry3d pose(source_poses[time_index].matrix());
    const auto& H_pose_0 = pose_derivatives_t0[time_index];
    const auto& H_pose_1 = pose_derivatives_t1[time_index];

    const auto& source_pt = source->points[i];
    const auto& target_pt = target->points[target_index];
    const auto& target_normal = target->normals[i];

    gtsam::Matrix46 H_transed_pose = gtsam::Matrix46::Zero();
    H_transed_pose.block<3, 3>(0, 0) = pose.linear() * -gtsam::SO3::Hat(source_pt.head<3>());
    H_transed_pose.block<3, 3>(0, 3) = pose.linear();
    const Eigen::Vector4d transed_source_pt = pose * source_pt;

    const auto& H_error_pose = H_transed_pose;
    const Eigen::Vector4d error = transed_source_pt - target_pt;

    const gtsam::Matrix46 H_0 = H_error_pose * H_pose_0;
    const gtsam::Matrix46 H_1 = H_error_pose * H_pose_1;

    int thread_num = 0;
#ifdef _OPENMP
    thread_num = omp_get_thread_num();
#endif

    const gtsam::Vector4 mahalanobis_error = mahalanobis[i] * error;
    const gtsam::Matrix64 H_0_mahalanobis = H_0.transpose() * mahalanobis[i];
    const gtsam::Matrix64 H_1_mahalanobis = H_1.transpose() * mahalanobis[i];

    sum_errors += 0.5 * error.transpose() * mahalanobis_error;
    Hs_00[thread_num] += H_0_mahalanobis * H_0;
    Hs_11[thread_num] += H_1_mahalanobis * H_1;
    Hs_01[thread_num] += H_0_mahalanobis * H_1;
    bs_0[thread_num] += H_0.transpose() * mahalanobis_error;
    bs_1[thread_num] += H_1.transpose() * mahalanobis_error;
  }

  for (int i = 1; i < Hs_00.size(); i++) {
    Hs_00[0] += Hs_00[i];
    Hs_11[0] += Hs_11[i];
    Hs_01[0] += Hs_01[i];
    bs_0[0] += bs_0[i];
    bs_1[0] += bs_1[i];
  }

  auto factor = gtsam::HessianFactor::shared_ptr(new gtsam::HessianFactor(keys_[0], keys_[1], Hs_00[0], Hs_01[0], -bs_0[0], Hs_11[0], -bs_1[0], sum_errors));
  return factor;
}

void IntegratedCT_GICPFactor::update_correspondences() const {
  correspondences.resize(source->size());
  mahalanobis.resize(source->size());

#pragma omp parallel for schedule(guided, 8) num_threads(num_threads)
  for (int i = 0; i < source->size(); i++) {
    const int time_index = time_indices[i];
    const Eigen::Matrix4d pose = source_poses[time_index].matrix();

    const auto& pt = source->points[i];
    const Eigen::Vector4d transed_pt = pose * pt;

    size_t k_index;
    double k_sq_dist;
    size_t num_found = target_tree->knn_search(transed_pt.data(), 1, &k_index, &k_sq_dist);

    if (num_found == 0 || k_sq_dist > max_correspondence_distance_sq) {
      correspondences[i] = -1;
      mahalanobis[i].setZero();
    } else {
      correspondences[i] = k_index;

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