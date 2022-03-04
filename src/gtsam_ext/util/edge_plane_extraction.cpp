// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/util/edge_plane_extraction.hpp>

#include <chrono>
#include <iostream>

namespace gtsam_ext {

struct ScanLine {
  ScanLine() : num_points(0), sum_angles(0.0) {}
  ScanLine(double angle) : num_points(1), sum_angles(angle) {}

  double angle() const { return sum_angles / num_points; }

  void add(double angle) {
    num_points++;
    sum_angles += angle;
  }
  void add(const ScanLine& other) {
    num_points += other.num_points;
    sum_angles += other.sum_angles;
  }

  int num_points;
  double sum_angles;
};

ScanLineInformation estimate_scan_lines(const Eigen::Vector4d* points, int num_points, int num_scan_lines, double angle_eps) {
  std::vector<ScanLine> scan_lines = {ScanLine(std::atan2(points[0].z(), points[0].head<2>().norm()))};

  for (int i = 1; i < num_points; i++) {
    const auto& pt = points[i];
    const double tilt = std::atan2(pt.z(), pt.head<2>().norm());

    // TODO: do binary search
    const auto closest =
      std::min_element(scan_lines.begin(), scan_lines.end(), [=](const ScanLine& lhs, const ScanLine& rhs) { return std::abs(lhs.angle() - tilt) < std::abs(rhs.angle() - tilt); });

    if (std::abs(closest->angle() - tilt) < angle_eps) {
      closest->add(tilt);
      continue;
    }

    scan_lines.push_back(ScanLine(tilt));
  }

  // Merge scans with similar angles
  std::sort(scan_lines.begin(), scan_lines.end(), [](const auto& lhs, const auto& rhs) { return lhs.angle() < rhs.angle(); });
  for (int i = 1; i < scan_lines.size(); i++) {
    if (scan_lines[i].angle() - scan_lines[i - 1].angle() < angle_eps) {
      scan_lines[i - 1].add(scan_lines[i]);
      scan_lines.erase(scan_lines.begin() + i);
    }
  }

  ScanLineInformation scan_line_info;
  if (num_scan_lines > 0 && scan_lines.size() >= num_scan_lines) {
    // Find first N lines with large number of points
    std::nth_element(scan_lines.begin(), scan_lines.begin() + num_scan_lines, scan_lines.end(), [](const ScanLine& lhs, const ScanLine& rhs) {
      return lhs.num_points > rhs.num_points;
    });

    // Sort them in increasing order by tilt angles
    std::sort(scan_lines.begin(), scan_lines.begin() + num_scan_lines, [](const ScanLine& lhs, const ScanLine& rhs) { return lhs.angle() < rhs.angle(); });

    for (int i = 0; i < num_scan_lines; i++) {
      scan_line_info.point_counts.push_back(scan_lines[i].num_points);
      scan_line_info.tilt_angles.push_back(scan_lines[i].angle());
    }
  } else {
    // Find the line with the largest number of points
    const auto max_line = std::max_element(scan_lines.begin(), scan_lines.end(), [](const ScanLine& lhs, const ScanLine& rhs) { return lhs.num_points < rhs.num_points; });
    const int thresh = max_line->num_points * 0.3;

    // Remove lines with few points
    const auto remove_loc = std::remove_if(scan_lines.begin(), scan_lines.end(), [=](const ScanLine& line) { return line.num_points < thresh; });
    scan_lines.erase(remove_loc, scan_lines.end());

    // Sort them in increasing order by tilt angles
    std::sort(scan_lines.begin(), scan_lines.end(), [](const ScanLine& lhs, const ScanLine& rhs) { return lhs.angle() < rhs.angle(); });

    for (int i = 0; i < scan_lines.size(); i++) {
      scan_line_info.point_counts.push_back(scan_lines[i].num_points);
      scan_line_info.tilt_angles.push_back(scan_lines[i].angle());
    }
  }

  return scan_line_info;
}

void extract_edge_plane_points(const ScanLineInformation& scan_lines, const Eigen::Vector4d* points, int num_points) {
  const auto t1 = std::chrono::high_resolution_clock::now();

  std::vector<std::pair<double, int>> tilt_angles(num_points);
  for (int i = 0; i < num_points; i++) {
    tilt_angles[i].first = std::atan2(points[i].z(), points[i].head<2>().norm());
    tilt_angles[i].second = i;
  }

  const auto t2 = std::chrono::high_resolution_clock::now();

  std::sort(tilt_angles.begin(), tilt_angles.end(), [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) { return lhs.first < rhs.first; });

  std::vector<std::vector<int>> lines(scan_lines.point_counts.size());

  const auto t3 = std::chrono::high_resolution_clock::now();

  std::cout << "d1:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[msec]" << std::endl;
  std::cout << "d2:" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "[msec]" << std::endl;
}

}  // namespace gtsam_ext