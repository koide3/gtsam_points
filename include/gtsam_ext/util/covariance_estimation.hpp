// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <vector>
#include <Eigen/Core>

namespace gtsam_ext {

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
estimate_covariances(const Eigen::Vector4d* points, int num_points, int k_neighbors = 10, int num_threads = 1);

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
estimate_covariances(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points, int k_neighbors = 10, int num_threads = 1);

}  // namespace gtsam_ext