// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <gtsam_points/ann/fast_occupancy_grid.hpp>

#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/vector3i_hash.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace gtsam_points {

template <typename PointCloud>
void FastOccupancyGrid::insert(const PointCloud& points, const Eigen::Isometry3d& pose) {
  for (int i = 0; i < frame::size(points); i++) {
    const auto& pt = frame::point(points, i);
    const Eigen::Array4i global_coord = fast_floor((pose * pt) * inv_resolution) + coord_offset;
    const Eigen::Array4i block_coord = global_coord / FastOccupancyBlock::stride;
    const Eigen::Array4i cell_coord = global_coord - block_coord * FastOccupancyBlock::stride;

    const std::uint64_t block_index = calc_index(block_coord);
    const std::uint64_t block_loc = find_or_insert_block(block_index);
    blocks[block_loc].second.set_occupied(cell_coord.head<3>());
  }
}

template <typename PointCloud>
int FastOccupancyGrid::calc_overlap(const PointCloud& points, const Eigen::Isometry3d& pose) const {
  int num_overlap = 0;
  for (int i = 0; i < frame::size(points); i++) {
    const auto& pt = frame::point(points, i);
    const Eigen::Array4i global_coord = fast_floor((pose * pt) * inv_resolution) + coord_offset;
    const Eigen::Array4i block_coord = global_coord / FastOccupancyBlock::stride;

    const std::uint64_t block_index = calc_index(block_coord);
    const std::uint64_t block_loc = find_block(block_index);
    if (block_loc == INVALID_INDEX) {
      continue;
    }

    const Eigen::Array4i cell_coord = global_coord - block_coord * FastOccupancyBlock::stride;
    num_overlap += blocks[block_loc].second.occupied(cell_coord.head<3>());
  }

  return num_overlap;
}

template <typename PointCloud>
double FastOccupancyGrid::calc_overlap_rate(const PointCloud& points, const Eigen::Isometry3d& pose) const {
  return calc_overlap(points, pose) / static_cast<double>(frame::size(points));
}

template <typename PointCloud>
std::vector<unsigned char> FastOccupancyGrid::get_overlaps(const PointCloud& points, const Eigen::Isometry3d& pose) const {
  std::vector<unsigned char> overlaps(frame::size(points), 0);

  for (int i = 0; i < frame::size(points); i++) {
    const auto& pt = frame::point(points, i);
    const Eigen::Array4i global_coord = fast_floor((pose * pt) * inv_resolution) + coord_offset;
    const Eigen::Array4i block_coord = global_coord / FastOccupancyBlock::stride;

    const std::uint64_t block_index = calc_index(block_coord);
    const std::uint64_t block_loc = find_block(block_index);
    if (block_loc == INVALID_INDEX) {
      continue;
    }

    const Eigen::Array4i cell_coord = global_coord - block_coord * FastOccupancyBlock::stride;
    overlaps[i] = blocks[block_loc].second.occupied(cell_coord.head<3>());
  }

  return overlaps;
}

}  // namespace gtsam_points
