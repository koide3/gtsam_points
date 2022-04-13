// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/types/voxelized_frame.hpp>

#include <fstream>
#include <iostream>
#include <boost/iterator/counting_iterator.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_ext {

bool Frame::has_times() const {
  if (!times) {
    std::cerr << "warning: frame doesn't have times" << std::endl;
  }
  return times;
}

bool Frame::has_points() const {
  if (!points) {
    std::cerr << "warning: frame doesn't have points" << std::endl;
  }
  return points;
}

bool Frame::has_normals() const {
  if (!normals) {
    std::cerr << "warning: frame doesn't have normals" << std::endl;
  }
  return normals;
}

bool Frame::has_covs() const {
  if (!covs) {
    std::cerr << "warning: frame doesn't have covs" << std::endl;
  }
  return covs;
}

bool Frame::has_intensities() const {
  if (!intensities) {
    std::cerr << "warning: frame doesn't have intensities" << std::endl;
  }
  return intensities;
}

namespace {
void write_binary(const std::string& filename, void* data, size_t size) {
  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<char*>(data), size);
}
}  // namespace

void Frame::save(const std::string& path) const {
  if (times) {
    write_binary(path + "/times.bin", times, sizeof(double) * num_points);
  }

  if (points) {
    write_binary(path + "/points.bin", points, sizeof(Eigen::Vector4d) * num_points);
  }

  if (normals) {
    write_binary(path + "/normals.bin", normals, sizeof(Eigen::Vector4d) * num_points);
  }

  if (covs) {
    write_binary(path + "/covs.bin", covs, sizeof(Eigen::Matrix4d) * num_points);
  }

  if (intensities) {
    write_binary(path + "/intensities.bin", intensities, sizeof(double) * num_points);
  }
}

void Frame::save_compact(const std::string& path) const {
  if (times) {
    std::vector<float> times_f(num_points);
    std::copy(times, times + num_points, times_f.begin());
    write_binary(path + "/times_compact.bin", times_f.data(), sizeof(float) * num_points);
  }

  if (points) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_f(num_points);
    std::transform(points, points + num_points, points_f.begin(), [](const Eigen::Vector4d& p) { return p.head<3>().cast<float>(); });
    write_binary(path + "/points_compact.bin", points_f.data(), sizeof(Eigen::Vector3f) * num_points);
  }

  if (normals) {
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_f(num_points);
    std::transform(normals, normals + num_points, normals_f.begin(), [](const Eigen::Vector4d& p) { return p.head<3>().cast<float>(); });
    write_binary(path + "/normals_compact.bin", normals_f.data(), sizeof(Eigen::Vector3f) * num_points);
  }

  if (covs) {
    std::vector<Eigen::Matrix<float, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 6, 1>>> covs_f(num_points);
    std::transform(covs, covs + num_points, covs_f.begin(), [](const Eigen::Matrix4d& cov) {
      return (Eigen::Matrix<float, 6, 1>() << cov(0, 0), cov(0, 1), cov(0, 2), cov(1, 1), cov(1, 2), cov(2, 2)).finished();
    });
    write_binary(path + "/covs_compact.bin", covs_f.data(), sizeof(Eigen::Matrix<float, 6, 1>) * num_points);
  }

  if (intensities) {
    std::vector<float> intensities_f(num_points);
    std::copy(intensities, intensities + num_points, intensities_f.begin());
    write_binary(path + "/intensities_compact.bin", intensities_f.data(), sizeof(float) * num_points);
  }
}

}  // namespace gtsam_ext
