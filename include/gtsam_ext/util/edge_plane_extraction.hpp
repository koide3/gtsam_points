// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>
#include <gtsam_ext/types/point_cloud_cpu.hpp>

namespace gtsam_ext {

struct ScanLineInformation {
  int size() const { return tilt_angles.size(); }

  int num_points(int i) const { return point_counts[i]; }
  double angle(int i) const { return tilt_angles[i]; }

  std::vector<int> point_counts;
  std::vector<double> tilt_angles;
};

/**
 * @brief Estimate beam projection angles of velodyne-like point cloud
 *
 * @param points          Points
 * @param num_points      Number of points
 * @param num_scan_lines  Number of scan lines (-1 to automatically estimate)
 * @param angle_eps       Minimum angle between scan lines
 * @return Estimated scan line information
 */
ScanLineInformation
estimate_scan_lines(const Eigen::Vector4d* points, int num_points, int num_scan_lines = -1, double angle_eps = 0.05 * M_PI / 180.0);

/**
 * @brief Extract edge and plane points
 *        Zhang and Singh, "LOAM: Lidar Odometry and Mapping in Real-time", RSS2014
 *
 * @param scan_lines    Scan line information
 * @param points        Points
 * @param num_points    Number of points
 * @return Extracted edge and plane points
 */
std::pair<PointCloudCPU::Ptr, PointCloudCPU::Ptr>
extract_edge_plane_points(const ScanLineInformation& scan_lines, const Eigen::Vector4d* points, int num_points);

}  // namespace gtsam_ext
