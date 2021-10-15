#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/types/voxelized_frame.hpp>

#include <iostream>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_ext {

double Frame::overlap(const std::shared_ptr<const VoxelizedFrame>& target, const Eigen::Isometry3d& delta) const {
  if (target->voxels == nullptr) {
    std::cerr << "error: CPU voxelmap has not been created!!" << std::endl;
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
    std::cerr << "error: CPU voxelmap has not been created!!" << std::endl;
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

}  // namespace gtsam_ext
