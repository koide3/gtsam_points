// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <queue>
#include <vector>
#include <Eigen/Core>
#include <gtsam_ext/ann/nearest_neighbor_search.hpp>

namespace gtsam_ext {

struct OmniProjectiveSearchParams {
  int rows;
  int cols;
  double v_angle_min;
  double v_angle_max;
};

struct OmniProjectiveSearch : public NearestNeighborSearch {
public:
  OmniProjectiveSearch(const Eigen::Vector4d* points, int num_points) : OmniProjectiveSearch(points, num_points, auto_param(points, num_points)) {}

  OmniProjectiveSearch(const Eigen::Vector4d* points, int num_points, const OmniProjectiveSearchParams& params)
  : points(points),
    num_points(num_points),
    params(params),
    grid(params.rows * params.cols) {
    //
    for (int i = 0; i < num_points; i++) {
      Eigen::Vector2i index = project(points[i]);
      if (index.y() < 0 || index.y() > params.rows - 1) {
        continue;
      }

      auto& cell = lookup(index);
      cell.push_back(i);
    }
  }

  virtual ~OmniProjectiveSearch() override {}

  bool compare(const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) { return lhs.second > rhs.second; }

  virtual size_t knn_search(const double* pt_, size_t k, size_t* k_indices, double* k_sq_dists) const override {
    Eigen::Map<const Eigen::Vector4d> pt(pt_);
    Eigen::Vector2i center = project(pt);

    const int half_window = 1;
    const int y_min = std::max<int>(0, center.y() - half_window);
    const int y_max = std::max<int>(params.rows - 1, center.y() + half_window);

    const auto comp = [](const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) { return lhs.second > rhs.second; };

    std::priority_queue<std::pair<size_t, double>, decltype(comp)> queue(comp);
    for (int x = center.x() - half_window; x <= center.x() + half_window; x++) {
      for (int y = y_min; y <= y_max; y++) {
        const auto& cell = lookup(Eigen::Vector2i(x, y));
        for (const auto index : cell) {
          queue.push(std::make_pair(index, (points[index] - pt).squaredNorm()));
          if (queue.size() > k) {
            queue.pop();
          }
        }
      }
    }

    /*
    int num_found = std::min<int>(candidates.size(), k);

    for (int i = 0; i < num_found; i++) {
      k_indices[i] = candidates[i].first;
      k_sq_dists[i] = candidates[i].second;
    }

    return num_found;
    */
    return 0;
  };

  Eigen::Vector2i project(const Eigen::Vector4d& pt) const {
    double v_angle = std::atan2(pt.z(), pt.head<2>().norm());
    double h_angle = std::atan2(pt.y(), pt.x());

    double v_p = (v_angle - params.v_angle_min) / (params.v_angle_max - params.v_angle_min);
    double h_p = (h_angle + M_PI) / (2.0 * M_PI);

    return Eigen::Array2d(h_p * params.cols, v_p * params.rows).floor().cast<int>();
  }

  std::vector<size_t>& lookup(const Eigen::Vector2i& index) {
    const int x = index.x() >= 0 ? index.x() % params.cols : index.x() + params.cols;
    const int y = std::max<int>(0, std::min<int>(params.rows - 1, index.y()));
    return grid[y * params.cols + x];
  }

  const std::vector<size_t>& lookup(const Eigen::Vector2i& index) const {
    const int x = index.x() >= 0 ? index.x() % params.cols : index.x() + params.cols;
    const int y = std::max<int>(0, std::min<int>(params.rows - 1, index.y()));
    return grid[y * params.cols + x];
  }

private:
  OmniProjectiveSearchParams auto_param(const Eigen::Vector4d* points, int num_points) const {
    std::vector<double> v_angles(num_points);
    for (int i = 0; i < num_points; i++) {
      double v_angle = std::atan2(points[i].z(), points[i].head<2>().norm());
      v_angles[i] = v_angle;
    }
    std::sort(v_angles.begin(), v_angles.end());
    const double v_angle_min = v_angles[v_angles.size() * 0.025];
    const double v_angle_max = v_angles[v_angles.size() * 0.975];
    return OmniProjectiveSearchParams{64, 1024, v_angle_min, v_angle_max};
  }

private:
  const Eigen::Vector4d* points;
  const int num_points;

  const OmniProjectiveSearchParams params;

  std::vector<std::vector<size_t>> grid;
};
}  // namespace gtsam_ext