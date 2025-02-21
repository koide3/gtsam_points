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

#include <gtsam_points/features/fpfh_estimation.hpp>

#include <numeric>
#include <Eigen/Geometry>
#include <gtsam_points/config.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/parallelism.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

Eigen::Vector4d compute_pair_features(const Eigen::Vector4d& p1, const Eigen::Vector4d& n1, const Eigen::Vector4d& p2, const Eigen::Vector4d& n2) {
  Eigen::Vector4d dp = p2 - p1;
  const double d = dp.norm();
  if (d == 0.0) {
    return Eigen::Vector4d::Zero();
  }

  Eigen::Vector4d ndp = dp / d;

  const double angle1 = n1.dot(ndp);
  const double angle2 = n2.dot(ndp);

  double alpha = angle1;
  Eigen::Vector4d n1_copy = n1;
  Eigen::Vector4d n2_copy = n2;

  // if (std::acos(std::abs(angle1)) > std::acos(std::abs(angle2))) {
  // acos is monotonic, so we can compare the cosines directly
  if (std::abs(angle1) < std::abs(angle2)) {
    n1_copy = n2;
    n2_copy = n1;
    ndp = -ndp;
    alpha = -angle2;
  }

  const Eigen::Vector4d v = ndp.cross3(n1_copy).normalized();
  if (!v.array().isFinite().all()) {
    return Eigen::Vector4d::Zero();
  }

  const Eigen::Vector4d w = n1_copy.cross3(v);

  const double phi = v.dot(n2_copy);
  const double theta = std::atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
  return Eigen::Vector4d(theta, phi, alpha, d);
}

std::vector<PFHSignature> estimate_pfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const int* indices,
  int num_indices,
  const NearestNeighborSearch& search,
  const PFHEstimationParams& params) {
  //
  constexpr int BINS = 5;
  const Eigen::Array4d fmin(-M_PI, -1.0, -1.0, 0.0);
  const Eigen::Array4d fmax(M_PI, 1.0, 1.0, 100.0);
  const Eigen::Array4d inv_range = 1.0 / (fmax - fmin);
  const Eigen::Array4i steps(1, BINS, BINS * BINS, 0);

  const bool use_bin_cache = params.num_threads == 1;
  std::unordered_map<std::uint64_t, size_t> bin_cache;

  const auto compute_bin = [&](size_t p1, size_t p2) -> size_t {
    if (!use_bin_cache) {
      const Eigen::Array4d f = compute_pair_features(points[p1], normals[p1], points[p2], normals[p2]);
      const Eigen::Array4d nf = (f - fmin) * inv_range;
      const Eigen::Array4i bins = fast_floor(nf * BINS).max(0).min(BINS - 1);
      return (bins * steps).sum();
    }

    const std::uint64_t key = (p2 << 32) | p1;
    const auto found = bin_cache.find(key);
    if (found != bin_cache.end()) {
      return found->second;
    }

    const Eigen::Array4d f = compute_pair_features(points[p1], normals[p1], points[p2], normals[p2]);
    const Eigen::Array4d nf = (f - fmin) * inv_range;
    const Eigen::Array4i bins = fast_floor(nf * BINS).max(0).min(BINS - 1);
    const size_t bin = (bins * steps).sum();

    bin_cache.emplace_hint(found, key, bin);

    return bin;
  };

  std::vector<PFHSignature> features(num_indices);

  const auto perpoint_task = [&](size_t k) {
    const auto& pt = points[indices[k]];
    std::vector<size_t> neighbors;
    std::vector<double> sq_dists;
    search.radius_search(pt.data(), params.search_radius, neighbors, sq_dists, params.max_num_neighbors);

    if (neighbors.size() <= 1) {
      features[k].setZero();
      return;
    }

    Eigen::Matrix<int, PFH_DIM, 1> hist = Eigen::Matrix<int, PFH_DIM, 1>::Zero();
    for (size_t i = 0; i < neighbors.size(); i++) {
      for (size_t j = 0; j < i; j++) {
        const size_t p1 = neighbors[i];
        const size_t p2 = neighbors[j];
        hist[compute_bin(p1, p2)]++;
      }
    }

    const double hist_incr = 100.0 / static_cast<double>(neighbors.size() * (neighbors.size() - 1) / 2);
    features[k] = hist.cast<double>() * hist_incr;
  };

  if (is_omp_default() || params.num_threads == 1) {
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 4)
    for (size_t k = 0; k < num_indices; k++) {
      perpoint_task(k);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_indices, 4), [&](const tbb::blocked_range<int>& range) {
      for (int k = range.begin(); k < range.end(); k++) {
        perpoint_task(k);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  return features;
}

std::vector<PFHSignature> estimate_pfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const NearestNeighborSearch& search,
  const PFHEstimationParams& params) {
  //
  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);
  return estimate_pfh(points, normals, num_points, indices.data(), num_points, search, params);
}

std::vector<FPFHSignature> estimate_fpfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const int* indices,
  int num_indices,
  const NearestNeighborSearch& search,
  const FPFHEstimationParams& params) {
  //
  constexpr int BINS = 11;
  const Eigen::Array4d fmin(-M_PI, -1.0, -1.0, 0.0);
  const Eigen::Array4d fmax(M_PI, 1.0, 1.0, 100.0);
  const Eigen::Array4d inv_range = 1.0 / (fmax - fmin);
  const Eigen::Array4i steps(1, BINS, BINS * BINS, 0);

  std::vector<std::vector<size_t>> all_neighbors(num_points);
  std::vector<std::vector<double>> all_sq_dists(num_points);
  std::vector<Eigen::Matrix<double, BINS * 3, 1>> spfh(num_points);

  const auto perpoint_task_spfh = [&](size_t k) {
    const size_t pt_index = k;
    const auto& pt = points[pt_index];

    auto& neighbors = all_neighbors[k];
    auto& sq_dists = all_sq_dists[k];
    search.radius_search(pt.data(), params.search_radius, neighbors, sq_dists, params.max_num_neighbors);

    if (neighbors.size() <= 1) {
      spfh[k] = Eigen::Matrix<double, BINS * 3, 1>::Zero();
      return;
    }

    Eigen::Matrix<int, BINS * 3, 1> hist = Eigen::Matrix<int, BINS * 3, 1>::Zero();
    for (size_t neighbor_index : neighbors) {
      const size_t p1 = pt_index;
      const size_t p2 = neighbor_index;
      if (p1 == p2) {
        continue;
      }

      const Eigen::Array4d f = compute_pair_features(points[p1], normals[p1], points[p2], normals[p2]);
      const Eigen::Array4d nf = (f - fmin) * inv_range;
      const Eigen::Array4i bins = fast_floor(nf * BINS).max(0).min(BINS - 1);

      hist[bins[0]]++;
      hist[bins[1] + BINS]++;
      hist[bins[2] + BINS * 2]++;
    }

    const double hist_incr = 100.0 / static_cast<double>(neighbors.size() - 1);
    spfh[k] = hist.cast<double>() * hist_incr;
  };

  std::vector<FPFHSignature> features(num_indices);
  const auto perpoint_task_fpfh = [&](size_t k) {
    const size_t pt_index = indices[k];
    const auto& pt = points[pt_index];
    const auto& neighbors = all_neighbors[pt_index];
    const auto& sq_dists = all_sq_dists[pt_index];

    if (neighbors.size() <= 1) {
      features[k].setZero();
      return;
    }

    FPFHSignature fpfh = FPFHSignature::Zero();
    for (size_t i = 0; i < neighbors.size(); i++) {
      if (sq_dists[i] == 0.0) {
        continue;
      }

      const double weight = 1.0 / sq_dists[i];
      fpfh += spfh[neighbors[i]] * weight;
    }

    const double norm1 = 100.0 / fpfh.middleRows<BINS>(0).sum();
    // const double norm2 = 100.0 / fpfh.middleRows<BINS>(BINS).sum();
    // const double norm3 = 100.0 / fpfh.middleRows<BINS>(BINS * 2).sum();

    features[k] = fpfh * norm1;
  };

  if (is_omp_default() || params.num_threads == 1) {
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 4)
    for (size_t k = 0; k < num_points; k++) {
      perpoint_task_spfh(k);
    }

#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 16)
    for (size_t k = 0; k < num_indices; k++) {
      perpoint_task_fpfh(k);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 4), [&](const tbb::blocked_range<int>& range) {
      for (int k = range.begin(); k < range.end(); k++) {
        perpoint_task_spfh(k);
      }
    });
    tbb::parallel_for(tbb::blocked_range<int>(0, num_indices, 4), [&](const tbb::blocked_range<int>& range) {
      for (int k = range.begin(); k < range.end(); k++) {
        perpoint_task_fpfh(k);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  return features;
}

std::vector<FPFHSignature> estimate_fpfh(
  const Eigen::Vector4d* points,
  const Eigen::Vector4d* normals,
  int num_points,
  const NearestNeighborSearch& search,
  const FPFHEstimationParams& params) {
  //
  std::vector<int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);
  return estimate_fpfh(points, normals, num_points, indices.data(), num_points, search, params);
}

std::vector<PFHSignature> estimate_pfh(const PointCloud& points, const NearestNeighborSearch& search, const PFHEstimationParams& params) {
  return estimate_pfh(points.points, points.normals, points.size(), search, params);
}

std::vector<FPFHSignature> estimate_fpfh(const PointCloud& points, const NearestNeighborSearch& search, const FPFHEstimationParams& params) {
  return estimate_fpfh(points.points, points.normals, points.size(), search, params);
}
}  // namespace gtsam_points