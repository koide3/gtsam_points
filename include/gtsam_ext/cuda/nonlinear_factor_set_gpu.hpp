// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#ifdef BUILD_GTSAM_EXT_GPU

#include <memory>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_ext/factors/nonlinear_factor_gpu.hpp>

struct CUstream_st;

namespace gtsam_ext {

/**
 * @brief This class holds a set of GPU-based NonlinearFactors and manages their linearization and cost evaluation tasks
 */
class NonlinearFactorSetGPU {
public:
  NonlinearFactorSetGPU();
  ~NonlinearFactorSetGPU();

  /**
   * @brief Number of GPU factors in this set
   * @return Number of GPU factors in this set
   */
  int size() const { return factors.size(); }

  /**
   * @brief Remove all factors
   */
  void clear() { factors.clear(); }

  /**
   * @brief Reset linearization and cost evaluation counts
   */
  void clear_counts();

  /**
   * @brief Number of issued linearization tasks
   * @return int
   */
  int linearization_count() const { return num_linearizations; }

  /**
   * @brief Number of issued cost evaluation tasks
   * @return int
   */
  int evaluation_count() const { return num_evaluations; }

  /**
   * @brief Add a factor to the GPU factor set if it is a GPU-based one
   * @param factor    Nonlinear factor
   * @return          True if the factor is GPU-based one and added to the set
   */
  bool add(boost::shared_ptr<gtsam::NonlinearFactor> factor);

  /**
   * @brief Add all GPU-based factors in a factor graph to the GPU factor set
   * @param factors   Factor graph
   */
  void add(const gtsam::NonlinearFactorGraph& factors);

  /**
   * @brief Add a GPU-based factor to the GPU factor set
   * @param factor  GPU-based factor
   */
  void add(boost::shared_ptr<NonlinearFactorGPU> factor);

  /**
   * @brief Compute all GPU-based linearization tasks
   * @param linearization_point   Current estimate
   */
  void linearize(const gtsam::Values& linearization_point);

  /**
   * @brief Compute all GPU-based cost evaluation tasks
   * @param values    Current estimate
   */
  void error(const gtsam::Values& values);

  /**
   * @brief Calculate linearized factors
   * @param linearization_point   Current estimated
   * @return Linearized factors
   */
  std::vector<gtsam::GaussianFactor::shared_ptr> calc_linear_factors(const gtsam::Values& linearization_point);

private:
  CUstream_st* stream;

  int num_linearizations;
  int num_evaluations;

  std::vector<boost::shared_ptr<NonlinearFactorGPU>> factors;

  thrust::host_vector<unsigned char, Eigen::aligned_allocator<unsigned char>> linearization_input_buffer_cpu;
  thrust::host_vector<unsigned char, Eigen::aligned_allocator<unsigned char>> linearization_output_buffer_cpu;
  thrust::device_vector<unsigned char> linearization_input_buffer_gpu;
  thrust::device_vector<unsigned char> linearization_output_buffer_gpu;

  thrust::host_vector<unsigned char, Eigen::aligned_allocator<unsigned char>> evaluation_input_buffer_cpu;
  thrust::host_vector<unsigned char, Eigen::aligned_allocator<unsigned char>> evaluation_output_buffer_cpu;
  thrust::device_vector<unsigned char> evaluation_input_buffer_gpu;
  thrust::device_vector<unsigned char> evaluation_output_buffer_gpu;
};

}  // namespace gtsam_ext

#else

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace gtsam_ext {

class NonlinearFactorGPU;

/**
 * @brief Dummy class for GPU-disabled build
 */
class NonlinearFactorSetGPU {
public:
  NonlinearFactorSetGPU() {}
  ~NonlinearFactorSetGPU() {}

  int size() const { return 0; }
  void clear() {}

  void clear_counts() {}
  int linearization_count() const { return 0; }
  int evaluation_count() const { return 0; }

  bool add(boost::shared_ptr<gtsam::NonlinearFactor> factor) { return false; }
  void add(const gtsam::NonlinearFactorGraph& factors) {}
  void add(boost::shared_ptr<NonlinearFactorGPU> factor) {}

  void linearize(const gtsam::Values& linearization_point) {}
  void error(const gtsam::Values& values) {}

  std::vector<gtsam::GaussianFactor::shared_ptr> calc_linear_factors(const gtsam::Values& linearization_point) {
    return std::vector<gtsam::GaussianFactor::shared_ptr>();
  }
};

}  // namespace gtsam_ext

#endif
