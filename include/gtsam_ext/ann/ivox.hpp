// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <unordered_map>

#include <Eigen/Core>

#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/types/frame_traits.hpp>
#include <gtsam_ext/ann/nearest_neighbor_search.hpp>

namespace gtsam_ext {

class XORVector3iHash {
public:
  size_t operator()(const Eigen::Vector3i& x) const;
};

struct LinearContainer {
public:
  using Ptr = std::shared_ptr<LinearContainer>;

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

struct iVox : public NearestNeighborSearch {
public:
  iVox(const double voxel_resolution = 0.5);
  virtual ~iVox() override;

  void insert(const Eigen::Vector4d* points, int num_points);
  void insert(const Frame& frame);

  virtual size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists) const override;

  const Eigen::Vector4d& point(const size_t i) const;
  const Eigen::Vector4d& normal(const size_t i) const;
  const Eigen::Matrix4d& cov(const size_t i) const;
  double intensity(const size_t i) const;

private:
  const Eigen::Vector3i voxel_coord(const Eigen::Vector4d& point) const;
  std::vector<Eigen::Vector3i> neighbor_offsets() const;

public:
  static const int voxel_id_bits = 20;
  static const int point_id_bits = 12;

  bool has_points;
  bool has_normals;
  bool has_covs;
  bool has_intensities;

  double voxel_resolution;
  double insertion_dist_sq_thresh;
  int lru_thresh;

  int neighbor_voxel_mode;  // 1, 7, 19, or 27

  int lru_count;

  using VoxelMap = std::unordered_map<Eigen::Vector3i, LinearContainer::Ptr, XORVector3iHash>;
  VoxelMap voxelmap;

  std::vector<LinearContainer::Ptr> voxels;
};

namespace frame {

template <>
struct traits<iVox> {
  static bool has_points(const iVox& ivox) { return ivox.has_points; }
  static bool has_normals(const iVox& ivox) { return ivox.has_normals; }
  static bool has_covs(const iVox& ivox) { return ivox.has_covs; }
  static bool has_intensities(const iVox& ivox) { return ivox.has_intensities; }

  static const Eigen::Vector4d& point(const iVox& ivox, size_t i) { return ivox.point(i); }
  static const Eigen::Vector4d& normal(const iVox& ivox, size_t i) { return ivox.normal(i); }
  static const Eigen::Matrix4d& cov(const iVox& ivox, size_t i) { return ivox.cov(i); }
  static double intensity(const iVox& ivox, size_t i) { return ivox.intensity(i); }
};

}  // namespace frame

}  // namespace gtsam_ext