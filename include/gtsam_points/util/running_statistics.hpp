// SPDX-License-Identifier: MIT
// Copyright (c) 2024  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <iostream>
#include <Eigen/Core>

namespace gtsam_points {

/// @brief Running statistics calculator.
/// @tparam T Type of data. Must be float, double, or fixed-size Eigen::Array.
/// @note  For vector types, the statistics are calculated element-wise.
template <typename T>
struct RunningStatistics {
public:
  RunningStatistics() : num_data(0), sum(0), sum_sq(0), min_(std::numeric_limits<double>::max()), max_(std::numeric_limits<double>::lowest()) {}

  /// @brief Add a data point.
  void add(const T& x) {
    num_data++;
    sum += x;
    sum_sq += x * x;

    if constexpr (std::is_floating_point_v<T>) {
      min_ = std::min(min_, x);
      max_ = std::max(max_, x);
    } else {
      min_ = min_.cwiseMin(x);
      max_ = max_.cwiseMax(x);
    }
  }

  /// @brief Join two statistics.
  RunningStatistics& operator+=(const RunningStatistics& rhs) {
    num_data += rhs.num_data;
    sum += rhs.sum;
    sum_sq += rhs.sum_sq;

    if constexpr (std::is_floating_point_v<T>) {
      min_ = std::min(min_, rhs.min_);
      max_ = std::max(max_, rhs.max_);
    } else {
      min_ = min_.cwiseMin(rhs.min_);
      max_ = max_.cwiseMax(rhs.max_);
    }

    return *this;
  }

  /// @brief Number of data points.
  size_t size() const { return num_data; }

  /// @brief Mean.
  T mean() const { return sum / num_data; }

  /// @brief Variance.
  T var() const { return (sum_sq - sum * mean()) / num_data; }

  /// @brief Standard deviation.
  T std() const {
    if constexpr (std::is_floating_point_v<T>) {
      return std::sqrt(var());
    } else {
      return var().sqrt();
    }
  }

  /// @brief Minimum value.
  const T& min() const { return min_; }

  /// @brief Maximum value.
  const T& max() const { return max_; }

private:
  size_t num_data;  ///< Number of data points
  T sum;            ///< Sum of data points
  T sum_sq;         ///< Sum of squared data points
  T min_;           ///< Minimum value
  T max_;           ///< Maximum value
};

}  // namespace gtsam_points

namespace std {
template <typename T>
std::ostream& operator<<(std::ostream& os, const gtsam_points::RunningStatistics<T>& stats) {
  if constexpr (std::is_floating_point_v<T>) {
    os << "num_data=" << stats.size() << " mean=" << stats.mean() << " std=" << stats.std() << " min=" << stats.min() << " max=" << stats.max();
  } else {
    os << "num_data=" << stats.size() << " mean=" << stats.mean().transpose() << " std=" << stats.std().transpose()
       << " min=" << stats.min().transpose() << " max=" << stats.max().transpose();
  }

  return os;
}

}  // namespace std