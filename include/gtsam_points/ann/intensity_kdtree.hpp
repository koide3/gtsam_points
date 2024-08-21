// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <Eigen/Core>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

// forward declaration
namespace nanoflann {

template <class T, class DataSource, typename _DistanceType>
class L2_Simple_Adaptor;

template <typename Distance, class DatasetAdaptor, int DIM, typename IndexType>
class KDTreeSingleIndexAdaptor;

}  // namespace nanoflann

namespace gtsam_points {

/**
 * @brief KdTree on intensity augmented coordinates.
 *        This class concatenates point coordinates (x, y, z) and intensity (i) and finds nearest neighbors on the XYZI space.
 */
struct IntensityKdTree : public NearestNeighborSearch {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, IntensityKdTree, double>, IntensityKdTree, 4, size_t>;

  /**
   * @brief Constructor
   * @param points          Input points
   * @param intensities     Input point intensities
   * @param num_points      Number of points
   * @param intensity_scale Scaling parameter to balance Euclidean coordinates and intensities
   */
  IntensityKdTree(const Eigen::Vector4d* points, const double* intensities, int num_points, double intensity_scale = 1.0);
  virtual ~IntensityKdTree() override;

  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim < 3) {
      return points[idx][dim];
    } else {
      return intensities[idx] * intensity_scale;
    }
  }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

  virtual size_t knn_search(
    const double* pt,
    size_t k,
    size_t* k_indices,
    double* k_sq_dists,
    double max_sq_dist = std::numeric_limits<double>::max()) const override;

public:
  const int num_points;
  const double* intensities;
  const Eigen::Vector4d* points;
  const double intensity_scale;

  double search_eps;

  std::unique_ptr<Index> index;
};

}  // namespace gtsam_points