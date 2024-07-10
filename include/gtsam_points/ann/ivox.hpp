// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <atomic>

#include <Eigen/Core>

#include <gtsam_points/ann/flat_container.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>

namespace gtsam_points {

/**
 * @brief Voxel-based incremental nearest neighbor search
 *        Bai et al., "Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels", IEEE RA-L, 2022
 * @note  Only the linear iVox is implemented
 */
using iVox = IncrementalVoxelMap<FlatContainer>;

}  // namespace gtsam_points