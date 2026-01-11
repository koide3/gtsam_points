// SPDX-License-Identifier: MIT
// Copyright (c) 2026  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/util/voxel_raycaster.hpp>
#include <gtsam_points/util/fast_floor.hpp>

namespace gtsam_points {

VoxelRaycaster::Iterator::Iterator(const VoxelRaycaster* traversal, const Eigen::Vector3i& coord)
: traversal(traversal),
  coord(coord),
  t_max(traversal ? traversal->t_max_init : Eigen::Vector3d::Zero()) {}

const Eigen::Vector3i& VoxelRaycaster::Iterator::operator*() const {
  return coord;
}

bool VoxelRaycaster::Iterator::operator!=(const Iterator& other) const {
  return (coord.array() != other.coord.array()).any();
}

VoxelRaycaster::Iterator& VoxelRaycaster::Iterator::operator++() {
  if (!traversal) {
    return *this;
  }

  constexpr double eps = 1e-9;

  const int min_axis = t_max.x() < t_max.y() && t_max.x() < t_max.z() ? 0 : (t_max.y() < t_max.z() ? 1 : 2);
  if (t_max[min_axis] > 1.0 + eps) {
    coord = traversal->end_coord;
    return *this;
  }

  coord[min_axis] += traversal->step[min_axis];
  t_max[min_axis] += traversal->t_delta[min_axis];

  // Check if we've reached or passed the end coordinate
  if (
    (traversal->step.array() > 0 && coord.array() > traversal->end_coord.array()).any() ||
    (traversal->step.array() < 0 && coord.array() < traversal->end_coord.array()).any()) {
    coord = traversal->end_coord;
  }

  return *this;
}

VoxelRaycaster::VoxelRaycaster(const Eigen::Vector4d& start, const Eigen::Vector4d& end, double voxel_size) {
  constexpr double eps = 1e-9;

  const Eigen::Vector3d ray_diff = (end - start).head<3>();
  step = ray_diff.cwiseSign().cast<int>();

  const double inv_voxel_size = 1.0 / voxel_size;
  start_coord = fast_floor(start.array() * inv_voxel_size).head<3>();
  end_coord = fast_floor(end.array() * inv_voxel_size).head<3>();

  const Eigen::Matrix<bool, 3, 1> valid_mask = (ray_diff.cwiseAbs().array() > eps);
  const Eigen::Vector3d inf = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());

  t_delta = valid_mask.select(voxel_size / ray_diff.cwiseAbs().array(), inf);

  const Eigen::Vector3d next_boundary = (start_coord.cast<double>().array() + (step.cast<double>().array() > 0.0).cast<double>()) * voxel_size;
  t_max_init = valid_mask.select((next_boundary - start.head<3>()).array() / ray_diff.array(), inf);
}

VoxelRaycaster::Iterator VoxelRaycaster::begin() const {
  return Iterator(this, start_coord);
}

VoxelRaycaster::Iterator VoxelRaycaster::end() const {
  return Iterator(nullptr, end_coord);
}

const Eigen::Vector3i& VoxelRaycaster::start_voxel() const {
  return start_coord;
}

const Eigen::Vector3i& VoxelRaycaster::end_voxel() const {
  return end_coord;
}

}