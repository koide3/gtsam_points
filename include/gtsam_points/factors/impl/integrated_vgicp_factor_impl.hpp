// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_factor.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/util/compact.hpp>
#include <gtsam_points/factors/impl/scan_matching_reduction.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

template <typename SourceFrame>
IntegratedVGICPFactor_<SourceFrame>::IntegratedVGICPFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const GaussianVoxelMap::ConstPtr& target_voxels,
  const std::shared_ptr<const SourceFrame>& source)
: gtsam_points::IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  mahalanobis_cache_mode(FusedCovCacheMode::FULL),
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
  mahalanobis_cache_mode(FusedCovCacheMode::FULL),
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
void IntegratedVGICPFactor_<SourceFrame>::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedVGICPFactor";
  if (is_binary) {
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;
  } else {
    std::cout << "(fixed, " << keyFormatter(this->keys()[0]) << ")" << std::endl;
  }

  std::cout << "target_resolusion=" << target_voxels->voxel_resolution() << ", |source|=" << frame::size(*source) << "pts" << std::endl;
  std::cout << "num_threads=" << num_threads << std::endl;
}

template <typename SourceFrame>
void IntegratedVGICPFactor_<SourceFrame>::update_correspondences(const Eigen::Isometry3d& delta) const {
  linearization_point = delta;
  correspondences.resize(frame::size(*source));

  switch (mahalanobis_cache_mode) {
    case FusedCovCacheMode::FULL:
      mahalanobis_full.resize(frame::size(*source));
      break;
    case FusedCovCacheMode::COMPACT:
      mahalanobis_compact.resize(frame::size(*source));
      break;
    case FusedCovCacheMode::NONE:
      break;
  }

  const auto perpoint_task = [&](int i) {
    Eigen::Vector4d pt = delta * frame::point(*source, i);
    Eigen::Vector3i coord = target_voxels->voxel_coord(pt);
    const auto voxel_id = target_voxels->lookup_voxel_index(coord);

    if (voxel_id < 0) {
      correspondences[i] = nullptr;

      switch (mahalanobis_cache_mode) {
        case FusedCovCacheMode::FULL:
          mahalanobis_full[i].setZero();
          break;
        case FusedCovCacheMode::COMPACT:
          mahalanobis_compact[i].setZero();
          break;
        case FusedCovCacheMode::NONE:
          break;
      }
    } else {
      const auto voxel = &target_voxels->lookup_voxel(voxel_id);
      correspondences[i] = voxel;

      switch (mahalanobis_cache_mode) {
        case FusedCovCacheMode::FULL: {
          const Eigen::Matrix4d RCR = (voxel->cov + delta.matrix() * frame::cov(*source, i) * delta.matrix().transpose());
          mahalanobis_full[i].setZero();
          mahalanobis_full[i].topLeftCorner<3, 3>() = RCR.topLeftCorner<3, 3>().inverse();
          break;
        }
        case FusedCovCacheMode::COMPACT: {
          const Eigen::Matrix4d RCR = (voxel->cov + delta.matrix() * frame::cov(*source, i) * delta.matrix().transpose());
          const Eigen::Matrix3d maha = RCR.topLeftCorner<3, 3>().inverse();
          mahalanobis_compact[i] = compact_cov(maha);
          break;
        }
        case FusedCovCacheMode::NONE:
          break;
      }
    }
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < frame::size(*source); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame::size(*source), 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
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
  if (correspondences.size() != frame::size(*source)) {
    update_correspondences(delta);
  }

  double sum_errors = 0.0;

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_target,
                               Eigen::Matrix<double, 6, 6>* H_source,
                               Eigen::Matrix<double, 6, 6>* H_target_source,
                               Eigen::Matrix<double, 6, 1>* b_target,
                               Eigen::Matrix<double, 6, 1>* b_source) {
    const auto& target_voxel = correspondences[i];
    if (target_voxel == nullptr) {
      return 0.0;
    }

    const auto& mean_A = frame::point(*source, i);
    const auto& cov_A = frame::cov(*source, i);

    const auto& mean_B = target_voxel->mean;
    const auto& cov_B = target_voxel->cov;

    Eigen::Vector4d transed_mean_A = delta * mean_A;
    Eigen::Vector4d residual = mean_B - transed_mean_A;

    Eigen::Matrix4d mahalanobis;
    switch (mahalanobis_cache_mode) {
      case FusedCovCacheMode::FULL:
        mahalanobis = mahalanobis_full[i];
        break;
      case FusedCovCacheMode::COMPACT:
        mahalanobis = uncompact_cov(mahalanobis_compact[i]);
        break;
      case FusedCovCacheMode::NONE: {
        const auto& delta_l = linearization_point;  // Delta at the linearization point
        const Eigen::Matrix4d RCR = (cov_B + delta_l.matrix() * cov_A * delta_l.matrix().transpose());
        mahalanobis.setZero();
        mahalanobis.topLeftCorner<3, 3>() = RCR.topLeftCorner<3, 3>().inverse();
      } break;
    }

    const double error = residual.transpose() * mahalanobis * residual;

    if (!H_target) {
      return error;
    }

    Eigen::Matrix<double, 4, 6> J_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_target.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_mean_A.head<3>());
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_source.block<3, 3>(0, 0) = delta.linear() * gtsam::SO3::Hat(mean_A.template head<3>());
    J_source.block<3, 3>(0, 3) = -delta.linear();

    Eigen::Matrix<double, 6, 4> J_target_mahalanobis = J_target.transpose() * mahalanobis;
    Eigen::Matrix<double, 6, 4> J_source_mahalanobis = J_source.transpose() * mahalanobis;

    *H_target += J_target_mahalanobis * J_target;
    *H_source += J_source_mahalanobis * J_source;
    *H_target_source += J_target_mahalanobis * J_source;
    *b_target += J_target_mahalanobis * residual;
    *b_source += J_source_mahalanobis * residual;

    return error;
  };

  if (is_omp_default() || num_threads == 1) {
    return scan_matching_reduce_omp(perpoint_task, frame::size(*source), num_threads, H_target, H_source, H_target_source, b_target, b_source);
  } else {
    return scan_matching_reduce_tbb(perpoint_task, frame::size(*source), H_target, H_source, H_target_source, b_target, b_source);
  }
}

}  // namespace gtsam_points
