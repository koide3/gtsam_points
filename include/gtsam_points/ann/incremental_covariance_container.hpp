// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <gtsam_points/types/point_cloud.hpp>

namespace gtsam_points {

/// @brief Linear container for incremental covariance and normal estimation.
struct IncrementalCovarianceContainer {
public:
  /// @brief FlatContainer setting.
  struct Setting {
    double min_sq_dist_in_cell = 0.1 * 0.1;  ///< Minimum squared distance between points in a cell.
    size_t max_num_points_in_cell = 10;      ///< Maximum number of points in a cell.
  };

  IncrementalCovarianceContainer() { points.reserve(10); }

  /// @brief Number of points in the container.
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

    this->flags.emplace_back(0);
    this->points.emplace_back(points.points[i]);
    this->normals.emplace_back(Eigen::Vector4d::Zero());
    this->covs.emplace_back(Eigen::Matrix4d::Zero());
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
      if (!valid(flags[i])) {
        continue;
      }

      result.push(i, (points[i] - pt).squaredNorm());
    }
  }

  /// @brief Find k nearest neighbors.
  /// @param pt           Query point
  /// @param result       Result
  template <typename Result>
  void knn_search_force(const Eigen::Vector4d& pt, Result& result) const {
    if (points.empty()) {
      return;
    }

    for (size_t i = 0; i < points.size(); i++) {
      result.push(i, (points[i] - pt).squaredNorm());
    }
  }

  /// @brief Set the i-th point as valid.
  void set_valid(int i) { flags[i] |= VALID_BIT; }

  /// @brief Check if the i-th point is valid.
  bool valid(int i) const { return flags[i] & VALID_BIT; }

  /// @brief Get the time when the i-th point was inserted.
  size_t birthday(int i) const { return flags[i] & BIRTHDAY_MASK; }

public:
  static constexpr size_t VALID_BIT = 1ull << 63;
  static constexpr size_t BIRTHDAY_MASK = (VALID_BIT >> 1) - 1;

  std::vector<Eigen::Vector4d> points;  ///< Points

  std::vector<size_t> flags;             ///< State flags
  std::vector<Eigen::Vector4d> normals;  ///< Normals
  std::vector<Eigen::Matrix4d> covs;     ///< Covariances
};

namespace frame {

template <>
struct traits<IncrementalCovarianceContainer> {
  static int size(const IncrementalCovarianceContainer& frame) { return frame.size(); }

  static bool has_points(const IncrementalCovarianceContainer& frame) { return !frame.points.empty(); }
  static bool has_normals(const IncrementalCovarianceContainer& frame) { return !frame.normals.empty(); }
  static bool has_covs(const IncrementalCovarianceContainer& frame) { return !frame.covs.empty(); }
  static bool has_intensities(const IncrementalCovarianceContainer& frame) { return false; }

  static const Eigen::Vector4d& point(const IncrementalCovarianceContainer& frame, size_t i) { return frame.points[i]; }
  static const Eigen::Vector4d& normal(const IncrementalCovarianceContainer& frame, size_t i) { return frame.normals[i]; }
  static const Eigen::Matrix4d& cov(const IncrementalCovarianceContainer& frame, size_t i) { return frame.covs[i]; }
  static double intensity(const IncrementalCovarianceContainer& frame, size_t i) { return 0.0; }
};

}  // namespace frame

}  // namespace gtsam_points