// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

namespace gtsam_points {

static std::vector<float> read_times(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    std::cerr << "error: failed to open " << path << std::endl;
    return std::vector<float>();
  }

  std::streamsize points_bytes = ifs.tellg();
  size_t num_data = points_bytes / sizeof(float);

  ifs.seekg(0, std::ios::beg);
  std::vector<float> times(num_data);
  ifs.read(reinterpret_cast<char*>(times.data()), sizeof(float) * num_data);

  return times;
}

static std::vector<Eigen::Vector3f> read_points(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    std::cerr << "error: failed to open " << path << std::endl;
    return std::vector<Eigen::Vector3f>();
  }

  std::streamsize points_bytes = ifs.tellg();
  size_t num_points = points_bytes / (sizeof(Eigen::Vector3f));

  ifs.seekg(0, std::ios::beg);
  std::vector<Eigen::Vector3f> points(num_points);
  ifs.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector3f) * num_points);

  return points;
}

static std::vector<Eigen::Vector4f> read_points4(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    std::cerr << "error: failed to open " << path << std::endl;
    return std::vector<Eigen::Vector4f>();
  }

  std::streamsize points_bytes = ifs.tellg();
  size_t num_points = points_bytes / (sizeof(Eigen::Vector4f));

  ifs.seekg(0, std::ios::beg);
  std::vector<Eigen::Vector4f> points(num_points);
  ifs.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector4f) * num_points);

  return points;
}

}  // namespace gtsam_points