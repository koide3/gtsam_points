#pragma once

#include <iostream>
#include <Eigen/Core>

namespace gtsam_points {

template <typename T>
struct RunningStatistics {
public:
  RunningStatistics() : num_data(0), sum(0), sum_sq(0), min_(std::numeric_limits<double>::max()), max_(std::numeric_limits<double>::lowest()) {}

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

  size_t size() const { return num_data; }

  T mean() const { return sum / num_data; }

  T var() const { return (sum_sq - sum * mean()) / num_data; }

  T std() const {
    if constexpr (std::is_floating_point_v<T>) {
      return std::sqrt(var());
    } else {
      return var().sqrt();
    }
  }

  const T& min() const { return min_; }

  const T& max() const { return max_; }

private:
  size_t num_data;
  T sum;
  T sum_sq;
  T min_;
  T max_;
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