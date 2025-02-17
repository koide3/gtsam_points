// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include <gtsam_points/ann/fast_occupancy_grid.hpp>
#include <gtsam_points/ann/impl/fast_occupancy_grid_impl.hpp>

#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/vector3i_hash.hpp>

namespace gtsam_points {

FastOccupancyGrid::FastOccupancyGrid(double resolution)
: inv_resolution(1.0 / resolution),
  max_seek_count(10),
  blocks(512, std::make_pair(INVALID_INDEX, FastOccupancyBlock())) {}

FastOccupancyGrid::~FastOccupancyGrid() {}

int FastOccupancyGrid::num_occupied_cells() const {
  return std::accumulate(blocks.begin(), blocks.end(), 0, [](int sum, const auto& block) { return sum + block.second.count(); });
}

std::uint64_t FastOccupancyGrid::calc_index(const Eigen::Vector4i& coord) const {
  return (static_cast<std::uint64_t>((coord[0]) & coord_bit_mask) << (coord_bit_size * 0)) |  //
         (static_cast<std::uint64_t>((coord[1]) & coord_bit_mask) << (coord_bit_size * 1)) |  //
         (static_cast<std::uint64_t>((coord[2]) & coord_bit_mask) << (coord_bit_size * 2));
}

Eigen::Vector3i FastOccupancyGrid::calc_coord(std::uint64_t index) const {
  return Eigen::Vector3i(                                                               //
    static_cast<int>((index >> (coord_bit_size * 0)) & coord_bit_mask) - coord_offset,  //
    static_cast<int>((index >> (coord_bit_size * 1)) & coord_bit_mask) - coord_offset,  //
    static_cast<int>((index >> (coord_bit_size * 2)) & coord_bit_mask) - coord_offset);
}

// std::uint64_t calc_hash(std::uint64_t index) const { return Vector3iHash()(calc_coord(index)); }
std::uint64_t FastOccupancyGrid::calc_hash(std::uint64_t index) const {
  return XORVector3iHash()(calc_coord(index));
}

std::uint64_t FastOccupancyGrid::find_block(std::uint64_t block_index) const {
  const std::uint64_t hash = calc_hash(block_index);
  for (int i = 0; i < max_seek_count; i++) {
    const int loc = (hash + i) & (blocks.size() - 1);
    if (blocks[loc].first == block_index) {
      return loc;
    }
  }

  return INVALID_INDEX;
}

std::uint64_t FastOccupancyGrid::find_or_insert_block(std::uint64_t block_index) {
  const std::uint64_t hash = calc_hash(block_index);
  for (int i = 0; i < max_seek_count; i++) {
    const int loc = (hash + i) & (blocks.size() - 1);
    if (blocks[loc].first == INVALID_INDEX || blocks[loc].first == block_index) {
      blocks[loc].first = block_index;
      return loc;
    }
  }

  rehash(blocks.size() * 2);
  return find_or_insert_block(block_index);
}

void FastOccupancyGrid::rehash(size_t hash_size) {
  std::vector<std::pair<std::uint64_t, FastOccupancyBlock>> new_blocks(hash_size, std::make_pair(INVALID_INDEX, FastOccupancyBlock()));

  for (const auto& block : blocks) {
    if (block.first == INVALID_INDEX) {
      continue;
    }

    bool inserted = false;
    const std::uint64_t hash = calc_hash(block.first);
    for (int i = 0; i < max_seek_count; i++) {
      const int loc = (hash + i) % hash_size;
      if (new_blocks[loc].first == INVALID_INDEX) {
        new_blocks[loc] = block;
        inserted = true;
        break;
      }
    }

    if (!inserted) {
      std::cerr << "failed to rehash (hash_size=" << hash_size << ")" << std::endl;
      return rehash(hash_size * 2);
    }
  }

  blocks = std::move(new_blocks);
}

template void FastOccupancyGrid::insert<PointCloud>(const PointCloud& points, const Eigen::Isometry3d& pose = Eigen::Isometry3d::Identity());
template int FastOccupancyGrid::calc_overlap<PointCloud>(const PointCloud& points, const Eigen::Isometry3d& pose = Eigen::Isometry3d::Identity())
  const;
template double FastOccupancyGrid::calc_overlap_rate<PointCloud>(
  const PointCloud& points,
  const Eigen::Isometry3d& pose = Eigen::Isometry3d::Identity()) const;
template std::vector<unsigned char> FastOccupancyGrid::get_overlaps<PointCloud>(
  const PointCloud& points,
  const Eigen::Isometry3d& pose = Eigen::Isometry3d::Identity()) const;

}  // namespace gtsam_points