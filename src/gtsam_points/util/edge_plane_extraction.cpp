// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/util/edge_plane_extraction.hpp>

#include <chrono>
#include <iostream>
#include <Eigen/Geometry>

namespace gtsam_points {

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
    const auto closest = std::min_element(scan_lines.begin(), scan_lines.end(), [=](const ScanLine& lhs, const ScanLine& rhs) {
      return std::abs(lhs.angle() - tilt) < std::abs(rhs.angle() - tilt);
    });

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
    std::sort(scan_lines.begin(), scan_lines.begin() + num_scan_lines, [](const ScanLine& lhs, const ScanLine& rhs) {
      return lhs.angle() < rhs.angle();
    });

    for (int i = 0; i < num_scan_lines; i++) {
      scan_line_info.point_counts.push_back(scan_lines[i].num_points);
      scan_line_info.tilt_angles.push_back(scan_lines[i].angle());
    }
  } else {
    // Find the line with the largest number of points
    const auto max_line = std::max_element(scan_lines.begin(), scan_lines.end(), [](const ScanLine& lhs, const ScanLine& rhs) {
      return lhs.num_points < rhs.num_points;
    });
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

void extract_edge_plane_points_line(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& plane_points,
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& edge_points) {
  // TODO: remove hardcoded parameters!!
  const int half_curvature_window = 5;
  const double edge_thresh = 0.35;
  const double plane_thresh = 0.05;
  const int max_edge_num = 2;
  const int max_plane_num = 4;
  const int occlusion_thresh = half_curvature_window * 0.75;

  // Precompute distances to points
  std::vector<double> distances(points.size());
  for (int i = 0; i < points.size(); i++) {
    distances[i] = points[i].norm();
  }

  // Calculate local curvatures (local smoothness)
  std::vector<int> occlusion_counts(points.size());
  std::vector<std::pair<double, int>> curvatures(points.size());
  for (int i = 0; i < points.size(); i++) {
    double sum_dists = 0.0;
    for (int offset = -half_curvature_window; offset <= half_curvature_window; offset++) {
      if (offset == 0) {
        continue;
      }

      int j = i + offset;
      j = j < 0 ? j + points.size() : j;
      j = j >= points.size() ? j - points.size() : j;

      sum_dists += (points[i] - points[j]).norm();

      if (distances[j] < distances[i] * 0.9) {
        occlusion_counts[i]++;
      }
    }

    const double c = 1.0 / (2 * half_curvature_window * distances[i]) * sum_dists;
    curvatures[i].first = c;
    curvatures[i].second = i;
  }

  // Sort points in increasing order by curvatures
  std::sort(curvatures.begin(), curvatures.end(), [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) {
    return lhs.first < rhs.first;
  });

  // Extract plane points
  const auto partition_plane =
    std::lower_bound(curvatures.begin(), curvatures.end(), plane_thresh, [](const std::pair<double, int>& c, const double thresh) {
      return c.first < thresh;
    });

  std::vector<int> num_selected_planes(points.size(), 0);
  for (auto c_point = curvatures.begin(); c_point != partition_plane; c_point++) {
    const int i = c_point->second;

    // Too many points in the subregion
    if (num_selected_planes[i] >= max_plane_num) {
      continue;
    }

    plane_points.push_back(points[i]);
    for (int offset = -half_curvature_window; offset <= half_curvature_window; offset++) {
      if (offset == 0) {
        continue;
      }

      int j = i + offset;
      j = j < 0 ? j + points.size() : j;
      j = j >= points.size() ? j - points.size() : j;
      num_selected_planes[j]++;
    }
  }

  // Extract edge points
  const auto partition_edge =
    std::lower_bound(curvatures.rbegin(), curvatures.rend(), edge_thresh, [](const std::pair<double, int>& c, const double thresh) {
      return c.first > thresh;
    });

  std::vector<int> num_selected_edges(points.size(), 0);
  for (auto c_point = curvatures.rbegin(); c_point != partition_edge; c_point++) {
    const int i = c_point->second;

    // The point can be occluded by closer objects, or there are too many edge points in the subregion
    if (occlusion_counts[i] >= occlusion_thresh || num_selected_edges[i] >= max_edge_num) {
      continue;
    }

    edge_points.push_back(points[i]);
    for (int offset = -half_curvature_window; offset <= half_curvature_window; offset++) {
      if (offset == 0) {
        continue;
      }

      int j = i + offset;
      j = j < 0 ? j + points.size() : j;
      j = j >= points.size() ? j - points.size() : j;
      num_selected_edges[j]++;
    }
  }
}

std::pair<PointCloudCPU::Ptr, PointCloudCPU::Ptr>
extract_edge_plane_points(const ScanLineInformation& scan_lines, const Eigen::Vector4d* points, int num_points) {
  // Estimate tilt and heading angles of each point
  std::vector<std::tuple<double, double, int>> tilt_heading_points(num_points);
  for (int i = 0; i < num_points; i++) {
    std::get<0>(tilt_heading_points[i]) = std::atan2(points[i].z(), points[i].head<2>().norm());
    std::get<1>(tilt_heading_points[i]) = std::atan2(points[i].y(), points[i].x());
    std::get<2>(tilt_heading_points[i]) = i;
  }

  // Sort by tilt angles and partition points in each scan line
  std::sort(
    tilt_heading_points.begin(),
    tilt_heading_points.end(),
    [](const std::tuple<double, double, int>& lhs, const std::tuple<double, double, int>& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

  std::vector<std::vector<std::tuple<double, double, int>>> lines(scan_lines.size());
  for (auto& line : lines) {
    line.reserve(2 * num_points / scan_lines.size());
  }

  int tilt_cursor = 0;
  for (const auto& tilt_heading_point : tilt_heading_points) {
    const double tilt = std::get<0>(tilt_heading_point);
    while (tilt_cursor < scan_lines.size() - 1 &&
           std::abs(scan_lines.angle(tilt_cursor + 1) - tilt) < std::abs(scan_lines.angle(tilt_cursor) - tilt)) {
      tilt_cursor++;
    }
    lines[tilt_cursor].push_back(tilt_heading_point);
  }

  // Extract edge and plane points
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> plane_points, edge_points;

  for (int i = 0; i < scan_lines.size(); i++) {
    auto& line = lines[i];
    std::sort(line.begin(), line.end(), [](const std::tuple<double, double, int>& lhs, const std::tuple<double, double, int>& rhs) {
      return std::get<1>(lhs) < std::get<1>(rhs);
    });

    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> line_points(line.size());
    std::transform(line.begin(), line.end(), line_points.begin(), [&](const std::tuple<double, double, int>& x) { return points[std::get<2>(x)]; });

    extract_edge_plane_points_line(line_points, plane_points, edge_points);
  }

  PointCloudCPU::Ptr edges(new PointCloudCPU(edge_points));
  PointCloudCPU::Ptr planes(new PointCloudCPU(plane_points));

  return std::make_pair(edges, planes);
}

}  // namespace gtsam_points