// SPDX-License-Identifier: MIT
// Copyright (c) 2026  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <Eigen/Core>

namespace gtsam_points {

/// @brief Raycasting-based voxel traversal algorithm.
///        The iterator traverses voxel coordinates intersected by the ray but excludes the end voxel.
/// @ref   J. Amanatides and A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing", Eurographics, 1987.
///
/// gtsam_points::VoxelRaycaster ray(Eigen::Vector4d(10.0, 20.0, 30.0, 1.0), Eigen::Vector4d(5.0, 10.0, 15.0, 1.0), 1.0);
/// std::vector<Eigen::Vector3i> voxels;
/// for (auto itr = ray.begin(); itr != ray.end(); ++itr) {
///   voxels.emplace_back(*itr);
/// }
/// voxels.emplace_back(ray.end_coord);   // Include the end voxel if needed
///
class VoxelRaycaster {
public:
  /// @brief Iterator for traversing voxel coordinates along the ray.
  class Iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Eigen::Vector3i;
    using difference_type = std::ptrdiff_t;
    using pointer = const Eigen::Vector3i*;
    using reference = Eigen::Vector3i&;

    Iterator(const VoxelRaycaster* traversal, const Eigen::Vector3i& coord);

    const Eigen::Vector3i& operator*() const;

    bool operator!=(const Iterator& other) const;

    Iterator& operator++();

  private:
    const VoxelRaycaster* traversal;
    Eigen::Vector3i coord;  ///< Current voxel coordinate
    Eigen::Vector3d t_max;  ///< Maximum t value for each axis
  };

  /// @brief Constructor
  /// @param start       Start point of the ray (homogeneous coordinates)
  /// @param end         End point of the ray (homogeneous coordinates)
  /// @param voxel_size  Size of each voxel
  VoxelRaycaster(const Eigen::Vector4d& start, const Eigen::Vector4d& end, double voxel_size);

  /// @brief Get the begin iterator. (begin() == start_coord)
  Iterator begin() const;

  /// @brief Get the end iterator. (end() == end_coord)
  Iterator end() const;

  /// @brief Get the start voxel coordinate.
  const Eigen::Vector3i& start_voxel() const;

  /// @brief  Get the end voxel coordinate.
  const Eigen::Vector3i& end_voxel() const;

private:
  Eigen::Vector3i step;         ///< Step direction along each axis
  Eigen::Vector3i start_coord;  ///< Starting voxel coordinate
  Eigen::Vector3i end_coord;    ///< Ending voxel coordinate
  Eigen::Vector3d t_delta;      ///< Delta for each axis
  Eigen::Vector3d t_max_init;   ///< Initial max t for each axis
};

}  // namespace gtsam_points