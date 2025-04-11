#pragma once

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <benchmark/dataset.hpp>
#include <benchmark/noise_feeder.hpp>
#include <boost/program_options.hpp>

/// @brief Benchmark class
class Benchmark {
public:
  Benchmark(const Dataset::ConstPtr& dataset, NoiseFeeder& noise, std::ostream& log_os, const boost::program_options::variables_map& vm);

private:
  /// @brief Create a graph to align all frames and keyframes
  void create_graph(double noise_t, double noise_r, NoiseFeeder& noise);

  /// @brief Verify the linearization result
  void verify(std::ostream& log_os, const boost::program_options::variables_map& vm);

  /// @brief Optimize the graph using Levenberg-Marquardt algorithm
  void optimize(std::ostream& log_os, const boost::program_options::variables_map& vm);

private:
  Dataset::ConstPtr dataset;                                             ///< Dataset
  std::unique_ptr<gtsam_points::StreamTempBufferRoundRobin> roundrobin;  ///< Stream temp buffer round robin

  gtsam::Values values;               ///< Initial sensor poses
  gtsam::NonlinearFactorGraph graph;  ///< Nonlinear factor graph
};
