// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/registration/graduated_non_convexity.hpp>

#include <algorithm>
#include <gtsam/geometry/Pose3.h>
#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/util/easy_profiler.hpp>

namespace gtsam_points {

template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_gnc(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  const NearestNeighborSearch& source_features_tree,
  const GNCParams& params) {
  gtsam_points::EasyProfiler prof("gnc");

  // Find correspondences
  if (params.verbose) {
    std::cout << "Finding correspondences |target|=" << frame::size(target) << " |source|=" << frame::size(source) << std::endl;
  }

  const auto find_correspondences = [&params, &prof](
                                      const auto& target,
                                      const auto& source,
                                      const auto& target_features,
                                      const auto& source_features,
                                      const auto& target_features_tree,
                                      const auto& source_features_tree) {  //
    prof.push("select samples");
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

    prof.push("find correspondences");
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

  prof.push("correspondences");
  std::vector<std::pair<int, int>> correspondences;
  if (frame::size(source) < frame::size(target)) {
    correspondences = find_correspondences(target, source, target_features, source_features, target_features_tree, source_features_tree);
  } else {
    correspondences = find_correspondences(source, target, source_features, target_features, source_features_tree, target_features_tree);
    std::for_each(correspondences.begin(), correspondences.end(), [](auto& c) { std::swap(c.first, c.second); });
  }

  prof.push("erase");
  correspondences.erase(
    std::remove_if(correspondences.begin(), correspondences.end(), [](const auto& c) { return c.first == -1 || c.second == -1; }),
    correspondences.end());

  if (params.verbose) {
    std::cout << "|correspondences|=" << correspondences.size() << std::endl;
  }

  prof.push("prepare");

  //
  Eigen::Array4d min_pt = frame::point(target, 0);
  Eigen::Array4d max_pt = frame::point(target, 0);
  for (size_t i = 1; i < frame::size(target); i++) {
    min_pt = min_pt.min(frame::point(target, i).array());
    max_pt = max_pt.max(frame::point(target, i).array());
  }
  double mu = (max_pt - min_pt).matrix().norm();

  Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();

  std::vector<double> weights(correspondences.size(), 1.0);

  for (int i = 0; i < params.max_iterations; i++) {
    for (int j = 0; j < params.innter_iterations; j++) {
      prof.push("itr_" + std::to_string(i));

      Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
      for (int j = 0; j < correspondences.size(); j++) {
        const auto& target_pt = frame::point(target, correspondences[j].first);
        const auto& source_pt = frame::point(source, correspondences[j].second);
        const Eigen::Vector4d transformed = T_target_source * source_pt;
        const Eigen::Vector4d residual = target_pt - transformed;

        Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
        J.block<3, 3>(0, 0) = T_target_source.linear() * gtsam::SO3::Hat(source_pt.template head<3>());
        J.block<3, 3>(0, 3) = -T_target_source.linear();

        H += weights[j] * J.transpose() * J;
        b += weights[j] * J.transpose() * residual;
      }

      const Eigen::Matrix<double, 6, 1> delta = (H + Eigen::Matrix<double, 6, 6>::Identity() * 1e-6).ldlt().solve(-b);
      T_target_source = T_target_source * Eigen::Isometry3d(gtsam::Pose3::Expmap(delta).matrix());

      double sum_errors = 0.0;
      for (int j = 0; j < correspondences.size(); j++) {
        const auto& target_pt = frame::point(target, correspondences[j].first);
        const auto& source_pt = frame::point(source, correspondences[j].second);
        const Eigen::Vector4d transformed = T_target_source * source_pt;
        const Eigen::Vector4d residual = target_pt - transformed;

        const double error = residual.squaredNorm();
        sum_errors += error;
        weights[j] = std::pow(mu / (mu + error), 2);
      }

      if (params.verbose) {
        std::cout << i << "/" << j << " : mu=" << mu << " sum_weights=" << std::accumulate(weights.begin(), weights.end(), 0.0)
                  << " sum_errors=" << sum_errors << std::endl;
      }
    }

    mu /= params.div_factor;
    if (mu < params.max_corr_dist) {
      break;
    }
  }

  prof.push("done");

  return RegistrationResult{1.0, T_target_source};
}

}  // namespace gtsam_points
