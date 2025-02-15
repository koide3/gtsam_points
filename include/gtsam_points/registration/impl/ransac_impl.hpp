// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/registration/ransac.hpp>

#include <unordered_set>
#include <gtsam/geometry/Pose3.h>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/vector3i_hash.hpp>
#include <gtsam_points/types/frame_traits.hpp>

namespace gtsam_points {

template <typename PointCloud, typename Features>
RegistrationResult estimate_pose_ransac(
  const PointCloud& target,
  const PointCloud& source,
  const Features& target_features,
  const Features& source_features,
  const NearestNeighborSearch& target_tree,
  const NearestNeighborSearch& target_features_tree,
  double inlier_voxel_resolution,
  int max_iterations,
  std::mt19937& mt) {
  //
  std::cout << "estimate_pose_ransac" << std::endl;
  const double inv_resolution = 1.0 / inlier_voxel_resolution;
  std::unordered_set<Eigen::Vector3i, Vector3iHash> target_voxels;
  for (size_t i = 0; i < frame::size(target); i++) {
    const Eigen::Array4i coord = fast_floor(frame::point(target, i) * inv_resolution);
    target_voxels.emplace(coord.head<3>());
  }

  const auto estimate_pose = [&]() -> std::optional<Eigen::Isometry3d> {
    std::uniform_int_distribution<> udist(0, frame::size(source) - 1);
    std::vector<size_t> source_indices(3);
    for (size_t i = 0; i < 3; i++) {
      source_indices[i] = udist(mt);
    }

    std::vector<double> sq_dists(3);
    std::vector<size_t> target_indices(3);
    for (size_t i = 0; i < 3; i++) {
      const auto& source_f = source_features[source_indices[i]];
      if (!target_features_tree.knn_search(source_f.data(), 1, &target_indices[i], &sq_dists[i])) {
        std::cerr << "warning: knn_search failed" << std::endl;
        return std::nullopt;
      }
    }

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> target_source_pairs(3);
    for (size_t i = 0; i < 3; i++) {
      target_source_pairs[i].first = frame::point(target, target_indices[i]).template head<3>();
      target_source_pairs[i].second = frame::point(source, source_indices[i]).template head<3>();
    }

    auto T_target_source = gtsam::Pose3::Align(target_source_pairs);
    if (!T_target_source) {
      std::cerr << "warning: Align failed" << std::endl;
      return std::nullopt;
    }

    return Eigen::Isometry3d(T_target_source->matrix());
  };

  size_t best_inliers = 0;
  Eigen::Isometry3d best_T_target_source = Eigen::Isometry3d::Identity();

  for (size_t k = 0; k < max_iterations; k++) {
    const auto T_target_source = estimate_pose();
    if (!T_target_source) {
      continue;
    }

    size_t inliers = 0;
    for (size_t i = 0; i < frame::size(source); i++) {
      const Eigen::Vector4d transformed = (*T_target_source) * frame::point(source, i);
      const Eigen::Array4i coord = fast_floor(transformed * inv_resolution);
      if (target_voxels.count(coord.head<3>())) {
        inliers++;
      }
    }

    if (best_inliers < inliers) {
      best_inliers = inliers;
      best_T_target_source = *T_target_source;

      std::cout << "k=" << k << " inliers=" << inliers << " / " << frame::size(source) << std::endl;
    }
  }

  return RegistrationResult{best_inliers / static_cast<double>(frame::size(target)), best_T_target_source};
}

}  // namespace gtsam_points
