// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/segmentation/region_growing.hpp>
#include <gtsam_points/segmentation/impl/region_growing_impl.hpp>

namespace gtsam_points {

template RegionGrowingContext region_growing_init_(
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const Eigen::Vector4d& seed_point,
  const RegionGrowingParams& params);

template bool
region_growing_step_(RegionGrowingContext& context, const PointCloud& points, const NearestNeighborSearch& search, const RegionGrowingParams& params);

template void region_growing_dilation_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params);

template bool region_growing_update_(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params);

RegionGrowingContext region_growing_init(
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const Eigen::Vector4d& seed_point,
  const RegionGrowingParams& params) {
  return region_growing_init_<PointCloud>(points, search, seed_point, params);
}

bool region_growing_update(
  RegionGrowingContext& context,
  const PointCloud& points,
  const NearestNeighborSearch& search,
  const RegionGrowingParams& params) {
  return region_growing_update_<PointCloud>(context, points, search, params);
}

}  // namespace gtsam_points