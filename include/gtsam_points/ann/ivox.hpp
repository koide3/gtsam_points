// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <atomic>

#include <Eigen/Core>

#include <gtsam_points/ann/flat_container.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>

namespace gtsam_points {

using iVox = IncrementalVoxelMap<FlatContainer>;

}  // namespace gtsam_points