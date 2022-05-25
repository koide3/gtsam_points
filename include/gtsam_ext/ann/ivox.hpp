// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <atomic>
#include <unordered_map>

#include <Eigen/Core>

#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/types/frame_traits.hpp>
#include <gtsam_ext/ann/nearest_neighbor_search.hpp>
#include <gtsam_ext/util/vector3i_hash.hpp>

namespace gtsam_ext {

/**
 * @brief Container to hold points in a voxel
 */
struct LinearContainer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<LinearContainer>;
  using ConstPtr = std::shared_ptr<const LinearContainer>;

  LinearContainer(const int lru_count);
  ~LinearContainer();

  int size() const { return points.size(); }
  void insert(const Eigen::Vector4d& point, const double insertion_dist_sq_thresh);
  void insert(const Frame& frame, const int i, const double insertion_dist_sq_thresh);

public:
  mutable std::atomic_int last_lru_count;
  int serial_id;

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> normals;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs;
  std::vector<double> intensities;
};

/**
 * @brief Voxel-based incremental nearest neighbor search
 *        Bai et al., "Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels", IEEE RA-L, 2022
 * @note  Only the linear iVox is implemented
 */
struct iVox : public NearestNeighborSearch {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<iVox>;
  using ConstPtr = std::shared_ptr<const iVox>;

  /**
   * @brief Construct a new iVox
   *
   * @param voxel_resolution       Voxel resolution
   * @param insertion_dist_thresh  Minimum distance between points
   * @param lru_thresh             LRC caching threshold
   */
  iVox(const double voxel_resolution = 0.5, const double insertion_dist_thresh = 0.05, const int lru_thresh = 10);
  virtual ~iVox() override;

  /**
   * @brief Insert points into the iVox
   * @param points      Input points
   * @param num_points  Number of points
   */
  virtual void insert(const Eigen::Vector4d* points, int num_points);

  /**
   * @brief Insert points and all available attributes into the iVox
   * @param frame   Input frame
   */
  virtual void insert(const Frame& frame);

  /**
   * @brief Find the closest point
   * @param pt          Query point
   * @param k_indices   Index output
   * @param k_sq_dists  Distance output
   * @return Number of found neighbors (0 or 1)
   */
  size_t nearest_neighbor_search(const double* pt, size_t* k_indices, double* k_sq_dists) const;

  /**
   * @brief Find k-nearest neighbors
   * @note  Indices encode voxel and point IDs of neighbor points
   *        First "voxel_id_bits" of an index indicates a sequential voxel ID
   *        Last "point_id_bits" of an index indicates a point ID in the voxel
   * @param pt          Query point
   * @param k           Number of neighbors
   * @param k_indices   Indices output
   * @param k_sq_dists  Distances output
   * @return Number of found neighbors
   */
  virtual size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const override;

  // Parameter setters
  void set_voxel_resolution(const double resolution) { voxel_resolution = resolution; }
  void set_insertion_dist_thresh(const double dist) { insertion_dist_sq_thresh = dist * dist; }
  void set_lru_thresh(const int lru_thresh) { this->lru_thresh = lru_thresh; }
  void set_neighbor_voxel_mode(const int mode) { offsets = neighbor_offsets(mode); }

  // Attribute check
  bool has_points() const { return points_available; }
  bool has_normals() const { return normals_available; }
  bool has_covs() const { return covs_available; }
  bool has_intensities() const { return intensities_available; }

  // Point attribute accessors
  const Eigen::Vector4d& point(const size_t i) const;
  const Eigen::Vector4d& normal(const size_t i) const;
  const Eigen::Matrix4d& cov(const size_t i) const;
  double intensity(const size_t i) const;

  // Number of voxels
  int num_voxels() const { return voxelmap.size(); }

  // Extract all points in iVox
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> voxel_points() const;

protected:
  inline size_t voxel_id(const size_t i) const { return i >> point_id_bits; }
  inline size_t point_id(const size_t i) const { return i & ((1 << point_id_bits) - 1); }

  const Eigen::Vector3i voxel_coord(const Eigen::Vector4d& point) const;
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> neighbor_offsets(const int neighbor_voxel_mode) const;

protected:
  static constexpr int point_id_bits = 12;
  static constexpr int voxel_id_bits = 32 - point_id_bits;

  bool points_available;
  bool normals_available;
  bool covs_available;
  bool intensities_available;

  double voxel_resolution;               // Voxel resolution
  double insertion_dist_sq_thresh;       // Minimum distance between points in a voxel
  int lru_thresh;                        // LRU caching threshold
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> offsets;  // Neighbor voxel offsets

  int lru_count;  // Counter to manage LRU voxel deletion

  using VoxelMap = std::unordered_map<
    Eigen::Vector3i,
    LinearContainer::Ptr,
    XORVector3iHash,
    std::equal_to<Eigen::Vector3i>,
    Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, LinearContainer::Ptr>>>;
  VoxelMap voxelmap;                         // Voxelmap
  std::vector<LinearContainer::Ptr> voxels;  // Flattened voxelmap for linear indexing
};

namespace frame {

template <>
struct traits<iVox> {
  static bool has_points(const iVox& ivox) { return ivox.has_points(); }
  static bool has_normals(const iVox& ivox) { return ivox.has_normals(); }
  static bool has_covs(const iVox& ivox) { return ivox.has_covs(); }
  static bool has_intensities(const iVox& ivox) { return ivox.has_intensities(); }

  static const Eigen::Vector4d& point(const iVox& ivox, size_t i) { return ivox.point(i); }
  static const Eigen::Vector4d& normal(const iVox& ivox, size_t i) { return ivox.normal(i); }
  static const Eigen::Matrix4d& cov(const iVox& ivox, size_t i) { return ivox.cov(i); }
  static double intensity(const iVox& ivox, size_t i) { return ivox.intensity(i); }
};

}  // namespace frame

}  // namespace gtsam_ext