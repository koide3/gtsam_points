// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>

namespace gtsam_ext {

struct ScanLineInformation {
  std::vector<int> point_counts;
  std::vector<double> tilt_angles;
};

ScanLineInformation estimate_scan_lines(const Eigen::Vector4d* points, int num_points, int num_scan_lines = -1, double angle_eps = 0.05 * M_PI / 180.0);

void extract_edge_plane_points(const ScanLineInformation& scan_lines, const Eigen::Vector4d* points, int num_points);

}  // namespace gtsam_ext
