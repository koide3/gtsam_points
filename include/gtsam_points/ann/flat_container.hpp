// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <queue>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam_points/ann/knn_result.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/frame_traits.hpp>

namespace gtsam_points {

/// @brief Point container with a flat vector.
struct FlatContainer {
public:
  /// @brief FlatContainer setting.
  struct Setting {
    void set_min_dist_in_cell(double dist) { this->min_sq_dist_in_cell = dist * dist; }
    void set_max_num_points_in_cell(size_t num_points) { this->max_num_points_in_cell = num_points; }

    double min_sq_dist_in_cell = 0.1 * 0.1;  ///< Minimum squared distance between points in a cell.
    size_t max_num_points_in_cell = 20;      ///< Maximum number of points in a cell.
  };

  /// @brief Constructor.
  FlatContainer() { points.reserve(10); }

  /// @brief Number of points.
  size_t size() const { return points.size(); }

  /// @brief Add a point to the container.
  void add(const Setting& setting, const PointCloud& points, size_t i) {
    if (
      this->points.size() >= setting.max_num_points_in_cell ||  //
      std::any_of(
        this->points.begin(),
        this->points.end(),
        [&](const auto& pt) { return (pt - points.points[i]).squaredNorm() < setting.min_sq_dist_in_cell; })  //
    ) {
      return;
    }

    this->points.emplace_back(points.points[i]);
    if (points.normals) {
      this->normals.emplace_back(points.normals[i]);
    }
    if (points.covs) {
      this->covs.emplace_back(points.covs[i]);
    }
    if (points.intensities) {
      this->intensities.emplace_back(points.intensities[i]);
    }
  }

  /// @brief Finalize the container (Nothing to do for FlatContainer).
  void finalize() {}

  /// @brief Find k nearest neighbors.
  /// @param pt           Query point
  /// @param result       Result
  template <typename Result>
  void knn_search(const Eigen::Vector4d& pt, Result& result) const {
    if (points.empty()) {
      return;
    }

    for (size_t i = 0; i < points.size(); i++) {
      const double sq_dist = (points[i] - pt).squaredNorm();
      result.push(i, sq_dist);
    }
  }

public:
  std::vector<Eigen::Vector4d> points;   ///< Points
  std::vector<Eigen::Vector4d> normals;  ///< Normals
  std::vector<Eigen::Matrix4d> covs;     ///< Covariances
  std::vector<double> intensities;       ///< Intensities
};

namespace frame {

template <>
struct traits<FlatContainer> {
  static int size(const FlatContainer& frame) { return frame.size(); }

  static bool has_points(const FlatContainer& frame) { return !frame.points.empty(); }
  static bool has_normals(const FlatContainer& frame) { return !frame.normals.empty(); }
  static bool has_covs(const FlatContainer& frame) { return !frame.covs.empty(); }
  static bool has_intensities(const FlatContainer& frame) { return !frame.intensities.empty(); }

  static const Eigen::Vector4d& point(const FlatContainer& frame, size_t i) { return frame.points[i]; }
  static const Eigen::Vector4d& normal(const FlatContainer& frame, size_t i) { return frame.normals[i]; }
  static const Eigen::Matrix4d& cov(const FlatContainer& frame, size_t i) { return frame.covs[i]; }
  static double intensity(const FlatContainer& frame, size_t i) { return frame.intensities[i]; }
};

}  // namespace frame

}  // namespace gtsam_points
