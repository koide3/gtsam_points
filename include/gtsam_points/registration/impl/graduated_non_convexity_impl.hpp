// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/registration/graduated_non_convexity.hpp>

#include <algorithm>
#include <gtsam/geometry/Pose3.h>
#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/registration/alignment.hpp>

namespace gtsam_points {

template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_gnc_(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const NearestNeighborSearch& source_features_tree,
  const GNCParams& params) {
  // Find correspondences
  if (params.verbose) {
    std::cout << "Finding correspondences |target|=" << frame::size(target) << " |source|=" << frame::size(source) << std::endl;
  }

  const auto find_correspondences = [&params](
                                      const auto& target,
                                      const auto& source,
                                      const auto& target_features,
                                      const auto& source_features,
                                      const auto& target_features_tree,
                                      const auto& source_features_tree) {  //
    std::vector<int> source_indices(frame::size(source));
    std::iota(source_indices.begin(), source_indices.end(), 0);
    if (source_indices.size() > params.max_init_samples) {
      std::shuffle(source_indices.begin(), source_indices.end(), std::mt19937(params.seed));
      source_indices.resize(params.max_init_samples);
      std::sort(source_indices.begin(), source_indices.end());
    }

    if (params.verbose) {
      std::cout << "|source_indices|=" << source_indices.size() << std::endl;
    }

    std::vector<std::pair<int, int>> correspondences(source_indices.size(), std::make_pair(-1, -1));
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 4)
    for (size_t i = 0; i < source_indices.size(); i++) {
      const size_t source_index = source_indices[i];
      size_t target_index;
      double sq_dist;
      if (!target_features_tree.knn_search(source_features[source_index].data(), 1, &target_index, &sq_dist)) {
        continue;
      }

      if (!params.reciprocal_check) {
        correspondences[i] = {target_index, source_index};
        continue;
      }

      size_t source_index_recp;
      if (!source_features_tree.knn_search(target_features[target_index].data(), 1, &source_index_recp, &sq_dist)) {
        continue;
      }

      if (source_index == source_index_recp) {
        correspondences[i] = {target_index, source_index};
      }
    }

    return correspondences;
  };

  std::vector<std::pair<int, int>> correspondences;
  if (frame::size(source) < frame::size(target)) {
    correspondences = find_correspondences(target, source, target_features, source_features, target_features_tree, source_features_tree);
  } else {
    correspondences = find_correspondences(source, target, source_features, target_features, source_features_tree, target_features_tree);
    std::for_each(correspondences.begin(), correspondences.end(), [](auto& c) { std::swap(c.first, c.second); });
  }

  correspondences.erase(
    std::remove_if(correspondences.begin(), correspondences.end(), [](const auto& c) { return c.first == -1 || c.second == -1; }),
    correspondences.end());

  if (params.verbose) {
    std::cout << "|correspondences|=" << correspondences.size() << " (initial samples)" << std::endl;
  }

  // Edge length similarity check
  if (params.tuple_check && correspondences.size() > params.max_num_tuples * 3) {
    std::vector<std::pair<int, int>> tuples(params.max_num_tuples * 3, std::make_pair(-1, -1));  // (target, source)

    std::mt19937 mt(params.seed);

    for (int i = 0; i < params.max_num_tuples; i++) {
      std::uniform_int_distribution<> udist(0, correspondences.size() - 1);

      const std::array<int, 3> indices = {udist(mt), udist(mt), udist(mt)};

      bool valid = true;
      for (int k = 0; k < 3; k++) {
        const auto& c1 = correspondences[indices[k]];
        const auto& c2 = correspondences[indices[(k + 1) % 3]];
        const double d1 = (frame::point(target, c1.first) - frame::point(target, c2.first)).matrix().norm();
        const double d2 = (frame::point(source, c1.second) - frame::point(source, c2.second)).matrix().norm();
        const double ratio = d1 / d2;

        if (ratio < params.tuple_thresh || ratio > 1.0 / params.tuple_thresh) {
          valid = false;
          break;
        }
      }

      if (valid) {
        tuples[i * 3 + 0] = correspondences[indices[0]];
        tuples[i * 3 + 1] = correspondences[indices[1]];
        tuples[i * 3 + 2] = correspondences[indices[2]];
      }
    }

    std::sort(tuples.begin(), tuples.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    tuples.erase(std::unique(tuples.begin(), tuples.end(), [](const auto& a, const auto& b) { return a.second == b.second; }), tuples.end());
    tuples.erase(std::remove_if(tuples.begin(), tuples.end(), [](const auto& c) { return c.first == -1 || c.second == -1; }), tuples.end());

    correspondences = tuples;

    if (params.verbose) {
      std::cout << "|correspondences|=" << correspondences.size() << " (after tuple check)" << std::endl;
    }
  }

  // Rough estimate of the diameter of the target point cloud
  Eigen::Array4d min_pt = frame::point(target, 0);
  Eigen::Array4d max_pt = frame::point(target, 0);
  for (size_t i = 1; i < frame::size(target); i++) {
    min_pt = min_pt.min(frame::point(target, i).array());
    max_pt = max_pt.max(frame::point(target, i).array());
  }
  double mu = (max_pt - min_pt).matrix().norm();

  // GNC loop
  const double inlier_thresh_sq = std::pow(2.0 * params.max_corr_dist, 2);

  RegistrationResult result;
  result.inlier_rate = 0.0;
  result.T_target_source.setIdentity();

  for (int i = 0; i < params.max_iterations; i++) {
    for (int j = 0; j < params.innter_iterations; j++) {
      std::vector<double> weights(correspondences.size(), 1.0);
      std::vector<Eigen::Vector4d> target_points(correspondences.size());
      std::vector<Eigen::Vector4d> source_points(correspondences.size());

      double sum_weights = 0.0;
      double sum_errors = 0.0;
      size_t num_inliers = 0;
      for (int j = 0; j < correspondences.size(); j++) {
        const auto& target_pt = frame::point(target, correspondences[j].first);
        const auto& source_pt = frame::point(source, correspondences[j].second);
        const Eigen::Vector4d transformed = result.T_target_source * source_pt;
        const Eigen::Vector4d residual = target_pt - transformed;

        const double error = residual.squaredNorm();
        const double weight = (i == 0 && j == 0) ? 1.0 : std::pow(mu / (mu + error), 2);

        sum_errors += error;
        sum_weights += weight;
        num_inliers += error < inlier_thresh_sq;

        weights[j] = weight;
        target_points[j] = target_pt;
        source_points[j] = source_pt;
      }

      switch(params.dof) {
        case 6:
          result.T_target_source = align_points_se3(target_points.data(), source_points.data(), weights.data(), weights.size());
          break;
        case 4:
          result.T_target_source = align_points_4dof(target_points.data(), source_points.data(), weights.data(), weights.size());
          break;
        default:
          std::cerr << "error: invalid dof " << params.dof << std::endl;
          abort();
      }
      result.inlier_rate = static_cast<double>(num_inliers) / static_cast<double>(correspondences.size());

      if (params.verbose) {
        std::cout << i << "/" << j << " : mu=" << mu << " sum_weights=" << sum_weights << " sum_errors=" << sum_errors << " num_inliers=" << num_inliers << std::endl;
      }
    }

    mu /= params.div_factor;
    if (mu < params.max_corr_dist) {
      break;
    }
  }

  return result;
}

}  // namespace gtsam_points
