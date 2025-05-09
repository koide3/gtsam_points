#pragma once

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/cuda/cuda_stream.hpp>
#include <benchmark/dataset.hpp>
#include <benchmark/noise_feeder.hpp>
#include <boost/program_options.hpp>

/// @brief Benchmark class
class BenchmarkOverlap {
public:
  BenchmarkOverlap(const Dataset::ConstPtr& dataset, NoiseFeeder& noise, std::ostream& log_os, const boost::program_options::variables_map& vm);

  bool is_verification_failed() const { return verification_failed; }

private:
  void create_pairs(NoiseFeeder& noise);
  void bench(std::ostream& log_os, const boost::program_options::variables_map& vm);

private:
  Dataset::ConstPtr dataset;        ///< Dataset
  gtsam_points::CUDAStream stream;  ///< CUDA stream

  std::vector<gtsam_points::GaussianVoxelMap::ConstPtr> pairwise_voxelmaps;
  std::vector<gtsam_points::PointCloud::ConstPtr> pairwise_points;
  std::vector<Eigen::Isometry3d> pairwise_deltas;
  std::vector<double> pairwise_overlaps;

  std::vector<std::vector<gtsam_points::GaussianVoxelMap::ConstPtr>> set_voxelmaps;
  std::vector<gtsam_points::PointCloud::ConstPtr> set_points;
  std::vector<std::vector<Eigen::Isometry3d>> set_deltas;
  std::vector<double> set_overlaps;

  bool verification_failed = true;  ///< Verification success flag
};
