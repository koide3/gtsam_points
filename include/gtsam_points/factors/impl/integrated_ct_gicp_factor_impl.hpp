// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>

#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/factors/impl/scan_matching_reduction.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

template <typename TargetFrame, typename SourceFrame>
IntegratedCT_GICPFactor_<TargetFrame, SourceFrame>::IntegratedCT_GICPFactor_(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree)
: IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>(source_t0_key, source_t1_key, target, source, target_tree) {
  //
  if (!frame::has_points(*target) || !frame::has_covs(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for ct_gicp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_covs(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for ct_gicp" << std::endl;
    abort();
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedCT_GICPFactor_<TargetFrame, SourceFrame>::IntegratedCT_GICPFactor_(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source)
: IntegratedCT_GICPFactor_(source_t0_key, source_t1_key, target, source, nullptr) {}

template <typename TargetFrame, typename SourceFrame>
IntegratedCT_GICPFactor_<TargetFrame, SourceFrame>::~IntegratedCT_GICPFactor_() {}

template <typename TargetFrame, typename SourceFrame>
void IntegratedCT_GICPFactor_<TargetFrame, SourceFrame>::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedCT_GICPFactor";
  std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;

  std::cout << "|target|=" << frame::size(*this->target) << "pts, |source|=" << frame::size(*this->source) << "pts" << std::endl;
  std::cout << "num_threads=" << this->num_threads << ", max_corr_dist=" << std::sqrt(this->max_correspondence_distance_sq) << std::endl;
}

template <typename TargetFrame, typename SourceFrame>
double IntegratedCT_GICPFactor_<TargetFrame, SourceFrame>::error(const gtsam::Values& values) const {
  this->update_poses(values);
  if (this->correspondences.size() != frame::size(*this->source)) {
    this->update_correspondences();
  }

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_target,
                               Eigen::Matrix<double, 6, 6>* H_source,
                               Eigen::Matrix<double, 6, 6>* H_target_source,
                               Eigen::Matrix<double, 6, 1>* b_target,
                               Eigen::Matrix<double, 6, 1>* b_source) {
    const long target_index = this->correspondences[i];
    if (target_index < 0) {
      return 0.0;
    }

    const int time_index = this->time_indices[i];
    const Eigen::Isometry3d pose(this->source_poses[time_index].matrix());

    const auto& source_pt = frame::point(*this->source, i);
    const auto& target_pt = frame::point(*this->target, target_index);

    const Eigen::Vector4d transed_source_pt = pose * source_pt;
    const Eigen::Vector4d residual = transed_source_pt - target_pt;
    const double error = residual.transpose() * mahalanobis[i] * residual;

    return error;
  };

  if (is_omp_default() || this->num_threads == 1) {
    return scan_matching_reduce_omp(perpoint_task, frame::size(*this->source), this->num_threads, nullptr, nullptr, nullptr, nullptr, nullptr);
  } else {
    return scan_matching_reduce_tbb(perpoint_task, frame::size(*this->source), nullptr, nullptr, nullptr, nullptr, nullptr);
  }
}

template <typename TargetFrame, typename SourceFrame>
boost::shared_ptr<gtsam::GaussianFactor> IntegratedCT_GICPFactor_<TargetFrame, SourceFrame>::linearize(const gtsam::Values& values) const {
  this->update_poses(values);
  this->update_correspondences();

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_00,
                               Eigen::Matrix<double, 6, 6>* H_11,
                               Eigen::Matrix<double, 6, 6>* H_01,
                               Eigen::Matrix<double, 6, 1>* b_0,
                               Eigen::Matrix<double, 6, 1>* b_1) {
    const long target_index = this->correspondences[i];
    if (target_index < 0) {
      return 0.0;
    }

    const int time_index = this->time_indices[i];

    const Eigen::Isometry3d pose(this->source_poses[time_index].matrix());
    const auto& H_pose_0 = this->pose_derivatives_t0[time_index];
    const auto& H_pose_1 = this->pose_derivatives_t1[time_index];

    const auto& source_pt = frame::point(*this->source, i);
    const auto& target_pt = frame::point(*this->target, target_index);

    gtsam::Matrix46 H_transed_pose = gtsam::Matrix46::Zero();
    H_transed_pose.block<3, 3>(0, 0) = pose.linear() * -gtsam::SO3::Hat(source_pt.template head<3>());
    H_transed_pose.block<3, 3>(0, 3) = pose.linear();
    const Eigen::Vector4d transed_source_pt = pose * source_pt;

    const auto& H_residual_pose = H_transed_pose;
    const Eigen::Vector4d residual = transed_source_pt - target_pt;

    const gtsam::Matrix46 H_0 = H_residual_pose * H_pose_0;
    const gtsam::Matrix46 H_1 = H_residual_pose * H_pose_1;

    const gtsam::Vector4 mahalanobis_residual = mahalanobis[i] * residual;
    const gtsam::Matrix64 H_0_mahalanobis = H_0.transpose() * mahalanobis[i];
    const gtsam::Matrix64 H_1_mahalanobis = H_1.transpose() * mahalanobis[i];

    const double error = residual.transpose() * mahalanobis_residual;
    *H_00 += H_0_mahalanobis * H_0;
    *H_11 += H_1_mahalanobis * H_1;
    *H_01 += H_0_mahalanobis * H_1;
    *b_0 += H_0.transpose() * mahalanobis_residual;
    *b_1 += H_1.transpose() * mahalanobis_residual;

    return error;
  };

  double error = 0.0;
  gtsam::Matrix6 H_00;
  gtsam::Matrix6 H_01;
  gtsam::Matrix6 H_11;
  gtsam::Vector6 b_0;
  gtsam::Vector6 b_1;

  if (is_omp_default() || this->num_threads == 1) {
    error = scan_matching_reduce_omp(perpoint_task, frame::size(*this->source), this->num_threads, &H_00, &H_11, &H_01, &b_0, &b_1);
  } else {
    error = scan_matching_reduce_tbb(perpoint_task, frame::size(*this->source), &H_00, &H_11, &H_01, &b_0, &b_1);
  }

  auto factor = gtsam::HessianFactor::shared_ptr(new gtsam::HessianFactor(this->keys_[0], this->keys_[1], H_00, H_01, -b_0, H_11, -b_1, error));
  return factor;
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedCT_GICPFactor_<TargetFrame, SourceFrame>::update_correspondences() const {
  this->correspondences.resize(frame::size(*this->source));
  this->mahalanobis.resize(frame::size(*this->source));

  const auto perpoint_task = [&](int i) {
    const int time_index = this->time_indices[i];
    const Eigen::Matrix4d pose = this->source_poses[time_index].matrix();

    const auto& pt = frame::point(*this->source, i);
    const Eigen::Vector4d transed_pt = pose * pt;

    size_t k_index = -1;
    double k_sq_dist = std::numeric_limits<double>::max();
    size_t num_found = this->target_tree->knn_search(transed_pt.data(), 1, &k_index, &k_sq_dist, this->max_correspondence_distance_sq);

    if (num_found == 0 || k_sq_dist > this->max_correspondence_distance_sq) {
      this->correspondences[i] = -1;
      this->mahalanobis[i].setZero();
    } else {
      this->correspondences[i] = k_index;

      const long target_index = this->correspondences[i];
      const auto& cov_A = frame::cov(*this->source, i);
      const auto& cov_B = frame::cov(*this->target, target_index);
      const Eigen::Matrix4d RCR = (cov_B + pose * cov_A * pose.transpose());

      mahalanobis[i].setZero();
      mahalanobis[i].block<3, 3>(0, 0) = RCR.block<3, 3>(0, 0).inverse();
    }
  };

  if (is_omp_default() || this->num_threads == 1) {
#pragma omp parallel for num_threads(this->num_threads) schedule(guided, 8)
    for (int i = 0; i < frame::size(*this->source); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame::size(*this->source), 8), [&](const tbb::blocked_range<int>& range) {
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
}  // namespace gtsam_points