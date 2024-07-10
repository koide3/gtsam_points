// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#include <gtsam_points/ann/incremental_covariance_container.hpp>

namespace gtsam_points {

IncrementalCovarianceContainer::IncrementalCovarianceContainer() {
  points.reserve(10);
}

void IncrementalCovarianceContainer::add(const Setting& setting, const PointCloud& points, size_t i) {
  if (
    this->points.size() >= setting.max_num_points_in_cell ||  //
    std::any_of(
      this->points.begin(),
      this->points.end(),
      [&](const auto& pt) { return (pt - points.points[i]).squaredNorm() < setting.min_sq_dist_in_cell; })  //
  ) {
    return;
  }

  this->flags.emplace_back(0);
  this->points.emplace_back(points.points[i]);
  this->normals.emplace_back(Eigen::Vector4d::Zero());
  this->covs.emplace_back(Eigen::Matrix4d::Zero());
}

size_t IncrementalCovarianceContainer::remove_old_invalid(int age_thresh, size_t lru) {
  int result = 0;
  for (int i = 0; i < points.size(); i++) {
    if (!valid(i) && age(i, lru) > age_thresh) {
      continue;
    }

    if (result != i) {
      points[result] = points[i];
      flags[result] = flags[i];
      normals[result] = normals[i];
      covs[result] = covs[i];
    }
    result++;
  }

  size_t num_removed = points.size() - result;
  flags.erase(flags.begin() + result, flags.end());
  points.erase(points.begin() + result, points.end());
  normals.erase(normals.begin() + result, normals.end());
  covs.erase(covs.begin() + result, covs.end());

  return num_removed;
}
}  // namespace gtsam_points
