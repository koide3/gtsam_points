// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <gtsam_points/util/vector3i_hash.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

namespace gtsam_points {

/// @brief Voxel meta information.
struct VoxelInfo {
public:
  /// @brief Default constructor.
  VoxelInfo() : lru(0), coord(-1, -1, -1) {}

  /// @brief Constructor.
  /// @param coord Integer voxel coordinates
  /// @param lru   LRU counter for caching
  VoxelInfo(const Eigen::Vector3i& coord, size_t lru) : lru(lru), coord(coord) {}

public:
  size_t lru;             ///< Last used time
  Eigen::Vector3i coord;  ///< Voxel coordinate
};

/// @brief Incremental voxelmap.
///        This class supports incremental point cloud insertion and LRU-based voxel deletion.
/// @note  This class can be used as a point cloud as well as a neighbor search structure.
/// @note  For the compatibility with other nearest neighbor search methods, this implementation returns indices that encode the voxel and point IDs.
///        The first ```point_id_bits``` (e.g., 32) bits of a point index represent the point ID, and the rest ```voxel_id_bits``` (e.g., 32) bits
///        represent the voxel ID that contains the point. The specified point can be looked up by `voxelmap.point(index)`;
template <typename VoxelContents>
struct IncrementalVoxelMap : public NearestNeighborSearch {
public:
  using Ptr = std::shared_ptr<IncrementalVoxelMap>;
  using ConstPtr = std::shared_ptr<const IncrementalVoxelMap>;

  /// @brief Constructor.
  /// @param leaf_size  Voxel size
  explicit IncrementalVoxelMap(double leaf_size);
  virtual ~IncrementalVoxelMap();

  /// @brief Voxel resolution.
  void set_voxel_resolution(const double leaf_size) { inv_leaf_size = 1.0 / leaf_size; }
  /// @brief LRU cache clearing cycle.
  void set_lru_clear_cycle(const int lru_clear_cycle) { this->lru_clear_cycle = lru_clear_cycle; }
  /// @brief LRU cache horizon.
  void set_lru_horizon(const int lru_horizon) { this->lru_horizon = lru_horizon; }
  /// @brief Neighboring voxel search mode (1, 7, 19, or 27).
  void set_neighbor_voxel_mode(const int mode) { offsets = neighbor_offsets(mode); }
  /// @brief Voxel setting.
  typename VoxelContents::Setting& voxel_insertion_setting() { return voxel_setting; }

  /// @brief Voxel size.
  double leaf_size() const { return 1.0 / inv_leaf_size; }

  /// @brief Number of voxels in the voxelmap.
  size_t num_voxels() const { return flat_voxels.size(); }

  /// @brief Clear the voxelmap.
  virtual void clear();

  /// @brief Insert points to the voxelmap.
  /// @param points Point cloud
  /// @param T      Transformation matrix
  virtual void insert(const PointCloud& points);

  /// @brief  Find k nearest neighbors.
  /// @param pt           Query point
  /// @param k            Number of neighbors to search
  /// @param k_indices    Indices of the k nearest neighbors
  /// @param k_sq_dists   Squared distances of the k nearest neighbors
  /// @return             Number of found neighbors
  virtual size_t knn_search(
    const double* pt,
    size_t k,
    size_t* k_indices,
    double* k_sq_dists,
    double max_sq_dist = std::numeric_limits<double>::max()) const override;

  /// @brief Calculate the global point index from the voxel index and the point index.
  inline size_t calc_index(const size_t voxel_id, const size_t point_id) const { return (voxel_id << point_id_bits) | point_id; }
  inline size_t voxel_id(const size_t i) const { return i >> point_id_bits; }                 ///< Extract the point ID from a global index.
  inline size_t point_id(const size_t i) const { return i & ((1ull << point_id_bits) - 1); }  ///< Extract the voxel ID from a global index.

  bool has_points() const;
  bool has_normals() const;
  bool has_covs() const;
  bool has_intensities() const;

  decltype(auto) point(const size_t i) const { return frame::point(flat_voxels[voxel_id(i)]->second, point_id(i)); }
  decltype(auto) normal(const size_t i) const { return frame::normal(flat_voxels[voxel_id(i)]->second, point_id(i)); }
  decltype(auto) cov(const size_t i) const { return frame::cov(flat_voxels[voxel_id(i)]->second, point_id(i)); }
  decltype(auto) intensity(const size_t i) const { return frame::intensity(flat_voxels[voxel_id(i)]->second, point_id(i)); }

  virtual std::vector<Eigen::Vector4d> voxel_points() const;
  virtual std::vector<Eigen::Vector4d> voxel_normals() const;
  virtual std::vector<Eigen::Matrix4d> voxel_covs() const;
  virtual std::vector<double> voxel_intensities() const;

  virtual PointCloudCPU::Ptr voxel_data() const;

protected:
  std::vector<Eigen::Vector3i> neighbor_offsets(const int neighbor_voxel_mode) const;

  template <typename Func>
  void visit_points(const Func& f) const {
    for (const auto& voxel : flat_voxels) {
      for (int i = 0; i < frame::size(voxel->second); i++) {
        f(voxel->second, i);
      }
    }
  }

protected:
  static_assert(sizeof(size_t) == 8, "size_t must be 64-bit");
  static constexpr int point_id_bits = 32;                  ///< Use the first 32 bits for point id
  static constexpr int voxel_id_bits = 64 - point_id_bits;  ///< Use the remaining bits for voxel id
  double inv_leaf_size;                                     ///< Inverse of the voxel size
  std::vector<Eigen::Vector3i> offsets;                     ///< Neighbor voxel offsets

  size_t lru_horizon;      ///< LRU horizon size. Voxels that have not been accessed for lru_horizon steps are deleted.
  size_t lru_clear_cycle;  ///< LRU clear cycle. Voxel deletion is performed every lru_clear_cycle steps.
  size_t lru_counter;      ///< LRU counter. Incremented every step.

  typename VoxelContents::Setting voxel_setting;                                  ///< Voxel setting.
  std::vector<std::shared_ptr<std::pair<VoxelInfo, VoxelContents>>> flat_voxels;  ///< Voxel contents.
  std::unordered_map<Eigen::Vector3i, size_t, XORVector3iHash> voxels;            ///< Voxel index map.
};

namespace frame {

template <typename VoxelContents>
struct traits<IncrementalVoxelMap<VoxelContents>> {
  static bool has_points(const IncrementalVoxelMap<VoxelContents>& ivox) { return ivox.has_points(); }
  static bool has_normals(const IncrementalVoxelMap<VoxelContents>& ivox) { return ivox.has_normals(); }
  static bool has_covs(const IncrementalVoxelMap<VoxelContents>& ivox) { return ivox.has_covs(); }
  static bool has_intensities(const IncrementalVoxelMap<VoxelContents>& ivox) { return ivox.has_intensities(); }

  static const Eigen::Vector4d& point(const IncrementalVoxelMap<VoxelContents>& ivox, size_t i) { return ivox.point(i); }
  static const Eigen::Vector4d& normal(const IncrementalVoxelMap<VoxelContents>& ivox, size_t i) { return ivox.normal(i); }
  static const Eigen::Matrix4d& cov(const IncrementalVoxelMap<VoxelContents>& ivox, size_t i) { return ivox.cov(i); }
  static double intensity(const IncrementalVoxelMap<VoxelContents>& ivox, size_t i) { return ivox.intensity(i); }
};

}  // namespace frame

}  // namespace gtsam_points
