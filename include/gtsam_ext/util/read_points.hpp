#pragma once

#include <vector>
#include <fstream>
#include <Eigen/Core>

namespace gtsam_ext {

std::vector<float> read_times(const std::string& path) {
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

std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> read_points(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if(!ifs) {
    std::cerr << "error: failed to open " << path << std::endl;
    return std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>();
  }

  std::streamsize points_bytes = ifs.tellg();
  size_t num_points = points_bytes / (sizeof(Eigen::Vector3f));

  ifs.seekg(0, std::ios::beg);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(num_points);
  ifs.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector3f) * num_points);

  return points;
}
}