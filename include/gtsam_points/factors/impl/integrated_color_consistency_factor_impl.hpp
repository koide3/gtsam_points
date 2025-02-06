// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_color_consistency_factor.hpp>

#include <vector>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/factors/impl/scan_matching_reduction.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::IntegratedColorConsistencyFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree,
  const std::shared_ptr<const IntensityGradients>& target_gradients)
: IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  photometric_term_weight(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source),
  target_tree(target_tree),
  target_gradients(target_gradients) {
  //
  if (!frame::has_points(*target) || !frame::has_normals(*target) || !frame::has_intensities(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_intensities(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }
}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::IntegratedColorConsistencyFactor_(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree,
  const std::shared_ptr<const IntensityGradients>& target_gradients)
: IntegratedMatchingCostFactor(fixed_target_pose, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  photometric_term_weight(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source),
  target_tree(target_tree),
  target_gradients(target_gradients) {
  //
  if (!frame::has_points(*target) || !frame::has_normals(*target) || !frame::has_intensities(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_intensities(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }
}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::~IntegratedColorConsistencyFactor_() {}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
void IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::print(
  const std::string& s,
  const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedColorConsistencyFactor";
  if (is_binary) {
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;
  } else {
    std::cout << "(fixed, " << keyFormatter(this->keys()[0]) << ")" << std::endl;
  }

  std::cout << "|target|=" << frame::size(*target) << "pts, |source|=" << frame::size(*source) << "pts" << std::endl;
  std::cout << "num_threads=" << num_threads << ", max_corr_dist=" << std::sqrt(max_correspondence_distance_sq) << std::endl;
}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
void IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::update_correspondences(const Eigen::Isometry3d& delta) const {
  bool do_update = true;
  if (correspondences.size() == frame::size(*source) && (correspondence_update_tolerance_trans > 0.0 || correspondence_update_tolerance_rot > 0.0)) {
    Eigen::Isometry3d diff = delta.inverse() * last_correspondence_point;
    double diff_rot = Eigen::AngleAxisd(diff.linear()).angle();
    double diff_trans = diff.translation().norm();
    if (diff_rot < correspondence_update_tolerance_rot && diff_trans < correspondence_update_tolerance_trans) {
      do_update = false;
    }
  }

  if (!do_update) {
    return;
  }

  last_correspondence_point = delta;
  correspondences.resize(frame::size(*source));

  const auto perpoint_task = [&](int i) {
    Eigen::Vector4d pt = delta * frame::point(*source, i);
    pt[3] = frame::intensity(*source, i);

    size_t k_index = -1;
    double k_sq_dist = -1;

    size_t num_found = target_tree->knn_search(pt.data(), 1, &k_index, &k_sq_dist, max_correspondence_distance_sq);
    correspondences[i] = (num_found && k_sq_dist < max_correspondence_distance_sq) ? k_index : -1;
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

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
double IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::evaluate(
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

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_target,
                               Eigen::Matrix<double, 6, 6>* H_source,
                               Eigen::Matrix<double, 6, 6>* H_target_source,
                               Eigen::Matrix<double, 6, 1>* b_target,
                               Eigen::Matrix<double, 6, 1>* b_source) {
    const long target_index = correspondences[i];
    if (target_index < 0) {
      return 0.0;
    }

    // source atributes
    const auto& mean_A = frame::point(*source, i);
    const double intensity_A = frame::intensity(*source, i);

    // target attributes
    const auto& mean_B = frame::point(*target, target_index);
    const auto& normal_B = frame::normal(*target, target_index);
    const auto& gradient_B = frame::intensity_gradient(*target_gradients, target_index);
    const double intensity_B = frame::intensity(*target, target_index);

    const Eigen::Vector4d transed_A = delta * mean_A;

    // photometric error
    const Eigen::Vector4d projected = transed_A - (transed_A - mean_B).dot(normal_B) * normal_B;
    const Eigen::Vector4d offset = projected - mean_B;
    const double error_photo = intensity_B + gradient_B.dot(offset) - intensity_A;

    const double err = error_photo * photometric_term_weight * error_photo;

    if (!H_target) {
      return err;
    }

    Eigen::Matrix<double, 4, 6> J_transed_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_transed_target.block<3, 3>(0, 0) = gtsam::SO3::Hat(transed_A.head<3>());
    J_transed_target.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_transed_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_transed_source.block<3, 3>(0, 0) = -delta.linear() * gtsam::SO3::Hat(mean_A.template head<3>());
    J_transed_source.block<3, 3>(0, 3) = delta.linear();

    // photometric error derivatives
    Eigen::Matrix<double, 4, 4> J_projected_transed = Eigen::Matrix4d::Identity() - normal_B * normal_B.transpose();
    J_projected_transed(3, 3) = 0.0;
    const auto& J_offset_transed = J_projected_transed;

    Eigen::Matrix<double, 1, 4> J_ephoto_offset = gradient_B;
    Eigen::Matrix<double, 1, 4> J_ephoto_transed = J_ephoto_offset * J_offset_transed;

    Eigen::Matrix<double, 1, 6> J_ephoto_target = J_ephoto_transed * J_transed_target;
    Eigen::Matrix<double, 1, 6> J_ephoto_source = J_ephoto_transed * J_transed_source;

    *H_target += J_ephoto_target.transpose() * photometric_term_weight * J_ephoto_target;
    *H_source += J_ephoto_source.transpose() * photometric_term_weight * J_ephoto_source;
    *H_target_source += J_ephoto_target.transpose() * photometric_term_weight * J_ephoto_source;
    *b_target += J_ephoto_target.transpose() * photometric_term_weight * error_photo;
    *b_source += J_ephoto_source.transpose() * photometric_term_weight * error_photo;

    return err;
  };

  double sum_errors_photo = 0.0;

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

  if (is_omp_default() || num_threads == 1) {
    return scan_matching_reduce_omp(perpoint_task, frame::size(*source), num_threads, H_target, H_source, H_target_source, b_target, b_source);
  } else {
    return scan_matching_reduce_tbb(perpoint_task, frame::size(*source), H_target, H_source, H_target_source, b_target, b_source);
  }
}

}  // namespace gtsam_points