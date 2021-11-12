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

double Frame::overlap(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const {
  if (target->voxels == nullptr) {
    std::cerr << "error: target CPU voxelmap has not been created!!" << std::endl;
    abort();
  }

  int num_overlap = 0;
  for (int i = 0; i < num_points; i++) {
    Eigen::Vector4d pt = delta * points[i];
    Eigen::Vector3i coord = target->voxels->voxel_coord(pt);
    if (target->voxels->lookup_voxel(coord)) {
      num_overlap++;
    }
  }

  return static_cast<double>(num_overlap) / num_points;
}

double Frame::overlap(const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets, const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas)
  const {
  //
  if (std::find_if(targets.begin(), targets.end(), [](const auto& target) { return target == nullptr; }) != targets.end()) {
    std::cerr << "error: target CPU voxelmap has not been created!!" << std::endl;
    abort();
  }

  int num_overlap = 0;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < targets.size(); j++) {
      const auto& target = targets[j];
      const auto& delta = deltas[j];

      Eigen::Vector4d pt = delta * points[i];
      Eigen::Vector3i coord = target->voxels->voxel_coord(pt);
      if (target->voxels->lookup_voxel(coord)) {
        num_overlap++;
        break;
      }
    }
  }

  return static_cast<double>(num_overlap) / num_points;
}

double Frame::overlap_auto(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const {
#ifdef BUILD_GTSAM_EXT_GPU
  if (points_gpu && target->voxels_gpu) {
    return overlap_gpu(target, delta);
  }
#endif
  return overlap(target, delta);
}

double Frame::overlap_auto(
  const std::vector<std::shared_ptr<const VoxelizedFrame>>& targets,
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& deltas) const {
#ifdef BUILD_GTSAM_EXT_GPU
  if (points_gpu && !targets.empty() && targets[0]->voxels_gpu) {
    return overlap_gpu(targets, deltas);
  }
#endif
  return overlap(targets, deltas);
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
