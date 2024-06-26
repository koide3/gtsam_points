// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_factor.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_points {

template <typename SourceFrame>
IntegratedVGICPFactor_<SourceFrame>::IntegratedVGICPFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const GaussianVoxelMap::ConstPtr& target_voxels,
  const std::shared_ptr<const SourceFrame>& source)
: gtsam_points::IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  target_voxels(std::dynamic_pointer_cast<const GaussianVoxelMapCPU>(target_voxels)),
  source(source) {
  //
  if (!frame::has_points(*source)) {
    std::cerr << "error: source points have not been allocated!!" << std::endl;
    abort();
  }

  if (!frame::has_covs(*source)) {
    std::cerr << "error: source don't have covs!!" << std::endl;
    abort();
  }

  if (!this->target_voxels) {
    std::cerr << "error: target voxelmap has not been created!!" << std::endl;
    abort();
  }
}

template <typename SourceFrame>
IntegratedVGICPFactor_<SourceFrame>::IntegratedVGICPFactor_(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const GaussianVoxelMap::ConstPtr& target_voxels,
  const std::shared_ptr<const SourceFrame>& source)
: gtsam_points::IntegratedMatchingCostFactor(fixed_target_pose, source_key),
  num_threads(1),
  target_voxels(std::dynamic_pointer_cast<const GaussianVoxelMapCPU>(target_voxels)),
  source(source) {
  //
  if (!frame::has_points(*source)) {
    std::cerr << "error: source points have not been allocated!!" << std::endl;
    abort();
  }

  if (!frame::has_covs(*source)) {
    std::cerr << "error: source don't have covs!!" << std::endl;
    abort();
  }

  if (!this->target_voxels) {
    std::cerr << "error: target voxelmap has not been created!!" << std::endl;
    abort();
  }
}

template <typename SourceFrame>
IntegratedVGICPFactor_<SourceFrame>::~IntegratedVGICPFactor_() {}

template <typename SourceFrame>
void IntegratedVGICPFactor_<SourceFrame>::update_correspondences(const Eigen::Isometry3d& delta) const {
  correspondences.resize(frame::size(*source));
  mahalanobis.resize(frame::size(*source));

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < frame::size(*source); i++) {
    Eigen::Vector4d pt = delta * frame::point(*source, i);
    Eigen::Vector3i coord = target_voxels->voxel_coord(pt);
    const auto voxel_id = target_voxels->lookup_voxel_index(coord);

    if (voxel_id < 0) {
      correspondences[i] = nullptr;
      mahalanobis[i].setIdentity();
    } else {
      const auto voxel = &target_voxels->lookup_voxel(voxel_id);

      correspondences[i] = voxel;

      Eigen::Matrix4d RCR = (voxel->cov + delta.matrix() * frame::cov(*source, i) * delta.matrix().transpose());
      RCR(3, 3) = 1.0;
      mahalanobis[i] = RCR.inverse();
      mahalanobis[i](3, 3) = 0.0;
    }
  }
}

template <typename SourceFrame>
double IntegratedVGICPFactor_<SourceFrame>::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
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
    const auto& target_voxel = correspondences[i];
    if (target_voxel == nullptr) {
      continue;
    }

    const auto& mean_A = frame::point(*source, i);
    const auto& cov_A = frame::cov(*source, i);

    const auto& mean_B = target_voxel->mean;
    const auto& cov_B = target_voxel->cov;

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

    Eigen::Matrix<double, 6, 4> J_target_mahalanobis = J_target.transpose() * mahalanobis[i];
    Eigen::Matrix<double, 6, 4> J_source_mahalanobis = J_source.transpose() * mahalanobis[i];

    Hs_target[thread_num] += J_target_mahalanobis * J_target;
    Hs_source[thread_num] += J_source_mahalanobis * J_source;
    Hs_target_source[thread_num] += J_target_mahalanobis * J_source;
    bs_target[thread_num] += J_target_mahalanobis * error;
    bs_source[thread_num] += J_source_mahalanobis * error;
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
