// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/ann/incremental_covariance_voxelmap.hpp>

namespace gtsam_points {

using iVoxCovarianceEstimation = IncrementalCovarianceVoxelMap;

}  // namespace gtsam_points