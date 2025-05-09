#pragma once

#include <memory>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

/// @brief Estimation frame that contains the pose and point cloud data.
struct Frame {
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  int id;                  ///< Frame ID
  double stamp;            ///< Frame timestamp
  Eigen::Isometry3d pose;  ///< Frame pose in world coordinates (groundtruth)

  gtsam_points::PointCloudGPU::Ptr points;                            ///< Point cloud
  std::vector<gtsam_points::GaussianVoxelMapGPU::Ptr> voxelmaps;      ///< Voxel maps
  std::vector<gtsam_points::GaussianVoxelMapCPU::Ptr> voxelmaps_cpu;  ///< Voxel maps
};

/// @brief Benchmark dataset
struct Dataset {
public:
  using Ptr = std::shared_ptr<Dataset>;
  using ConstPtr = std::shared_ptr<const Dataset>;

  /// @brief Constructor
  /// @details Read the dataset from a directory
  Dataset(const std::filesystem::path& data_path, bool force_recreate_voxelmaps = false);

public:
  std::filesystem::path data_path;    ///< Dataset path
  std::vector<Frame::Ptr> frames;     ///< Frames
  std::vector<Frame::Ptr> keyframes;  ///< Keyframes
};
