// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>
#include <gtsam_ext/types/frame_cpu.hpp>

namespace gtsam_ext {

struct ScanLineInformation {
  int size() const { return tilt_angles.size(); }

  int num_points(int i) const { return point_counts[i]; }
  double angle(int i) const { return tilt_angles[i]; }

  std::vector<int> point_counts;
  std::vector<double> tilt_angles;
};

ScanLineInformation estimate_scan_lines(const Eigen::Vector4d* points, int num_points, int num_scan_lines = -1, double angle_eps = 0.05 * M_PI / 180.0);

std::pair<FrameCPU::Ptr, FrameCPU::Ptr> extract_edge_plane_points(const ScanLineInformation& scan_lines, const Eigen::Vector4d* points, int num_points);

}  // namespace gtsam_ext
