// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
//
// The FPFH extraction code is heavily inspired by the implementation in PCL.
// https://github.com/PointCloudLibrary/pcl/blob/master/features/include/pcl/features/fpfh.h

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <vector>
#include <Eigen/Core>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

namespace gtsam_points {

constexpr int PFH_DIM = 125;
constexpr int FPFH_DIM = 33;

using PFHSignature = Eigen::Matrix<double, PFH_DIM, 1>;
using FPFHSignature = Eigen::Matrix<double, FPFH_DIM, 1>;

/// @brief PFH estimation parameters
struct PFHEstimationParams {
  double search_radius = 5.0;     ///< Neighbor search radius
  int max_num_neighbors = 10000;  ///< Maximum number of neighbors
  int num_threads = 4;            ///< Number of threads
};

using FPFHEstimationParams = PFHEstimationParams;

/// @brief Compute pair features for PFH.
/// @note  Equivalent to pcl::computePairFeatures.
Eigen::Vector4d compute_pair_features(const Eigen::Vector4d& p1, const Eigen::Vector4d& n1, const Eigen::Vector4d& p2, const Eigen::Vector4d& n2);

/// @brief Estimate PFH features.
/// @param points       Input points
/// @param normals      Input normals
/// @param num_points   Number of input points and normals
/// @param indices      Indices of points to estimate PFH
/// @param num_indices  Number of indices
/// @param search       Nearest neighbor search
/// @param params       PFH estimation parameters
/// @return             PFH features (D = 125, N = num_indices)
std::vector<PFHSignature> estimate_pfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const int* indices,
  int num_indices,
  const NearestNeighborSearch& search,
  const PFHEstimationParams& params = PFHEstimationParams());

/// @brief Estimate PFH features.
/// @param points       Input points
/// @param normals      Input normals
/// @param num_points   Number of input points and normals
/// @param search       Nearest neighbor search
/// @param params       PFH estimation parameters
/// @return             PFH features (D = 125, N = num_indices)
std::vector<PFHSignature> estimate_pfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const NearestNeighborSearch& search,
  const PFHEstimationParams& params = PFHEstimationParams());

/// @brief Estimate FPFH features.
/// @param points       Input points
/// @param normals      Input normals
/// @param num_points   Number of input points and normals
/// @param indices      Indices of points to estimate PFH
/// @param num_indices  Number of indices
/// @param search       Nearest neighbor search
/// @param params       FPFH estimation parameters
/// @return             FPFH features (D = 33, N = num_indices)
std::vector<FPFHSignature> estimate_fpfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const int* indices,
  int num_indices,
  const NearestNeighborSearch& search,
  const FPFHEstimationParams& params = FPFHEstimationParams());

/// @brief Estimate FPFH features.
/// @param points       Input points
/// @param normals      Input normals
/// @param num_points   Number of input points and normals
/// @param search       Nearest neighbor search
/// @param params       FPFH estimation parameters
/// @return             FPFH features (D = 33, N = num_indices)
std::vector<FPFHSignature> estimate_fpfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const NearestNeighborSearch& search,
  const FPFHEstimationParams& params = FPFHEstimationParams());

/// @brief Estimate PFH features.
/// @param points       Input points (Needs to have normals)
/// @param search       Nearest neighbor search
/// @param params       PFH estimation parameters
std::vector<PFHSignature>
estimate_pfh(const PointCloud& points, const NearestNeighborSearch& search, const PFHEstimationParams& params = PFHEstimationParams());

/// @brief Estimate FPFH features.
/// @param points       Input points (Needs to have normals)
/// @param search       Nearest neighbor search
/// @param params       FPFH estimation parameters
std::vector<FPFHSignature>
estimate_fpfh(const PointCloud& points, const NearestNeighborSearch& search, const FPFHEstimationParams& params = FPFHEstimationParams());

}  // namespace gtsam_points