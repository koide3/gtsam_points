// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <vector>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/factors/nonlinear_factor_gpu.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>

struct CUstream_st;

namespace gtsam_points {

/**
 * @brief This class holds a set of GPU-based NonlinearFactors and manages their linearization and cost evaluation tasks
 */
class NonlinearFactorSetGPU : public NonlinearFactorSet {
public:
  NonlinearFactorSetGPU();
  ~NonlinearFactorSetGPU();

  /**
   * @brief Number of GPU factors in this set
   * @return Number of GPU factors in this set
   */
  int size() const override { return factors.size(); }

  /**
   * @brief Remove all factors
   */
  void clear() override { factors.clear(); }

  /**
   * @brief Reset linearization and cost evaluation counts
   */
  void clear_counts() override;

  /**
   * @brief Number of issued linearization tasks
   * @return int
   */
  int linearization_count() const override { return num_linearizations; }

  /**
   * @brief Number of issued cost evaluation tasks
   * @return int
   */
  int evaluation_count() const override { return num_evaluations; }

  /**
   * @brief Add a factor to the GPU factor set if it is a GPU-based one
   * @param factor    Nonlinear factor
   * @return          True if the factor is GPU-based one and added to the set
   */
  bool add(boost::shared_ptr<gtsam::NonlinearFactor> factor) override;

  /**
   * @brief Add all GPU-based factors in a factor graph to the GPU factor set
   * @param factors   Factor graph
   */
  void add(const gtsam::NonlinearFactorGraph& factors) override;

  /**
   * @brief Compute all GPU-based linearization tasks
   * @param linearization_point   Current estimate
   */
  void linearize(const gtsam::Values& linearization_point) override;

  /**
   * @brief Compute all GPU-based cost evaluation tasks
   * @param values    Current estimate
   */
  void error(const gtsam::Values& values) override;

  /**
   * @brief Calculate linearized factors
   * @param linearization_point   Current estimated
   * @return Linearized factors
   */
  std::vector<gtsam::GaussianFactor::shared_ptr> calc_linear_factors(const gtsam::Values& linearization_point) override;

private:
  /// @brief Simple buffer class for device memory
  struct DeviceBuffer {
    DeviceBuffer();
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void resize(size_t size, CUstream_st* stream);
    unsigned char* data() { return buffer; }
    const unsigned char* data() const { return buffer; }

    size_t size;
    unsigned char* buffer;
  };

private:
  CUstream_st* stream;

  int num_linearizations;
  int num_evaluations;

  std::vector<boost::shared_ptr<NonlinearFactorGPU>> factors;

  std::vector<unsigned char, Eigen::aligned_allocator<unsigned char>> linearization_input_buffer_cpu;
  std::vector<unsigned char, Eigen::aligned_allocator<unsigned char>> linearization_output_buffer_cpu;
  std::unique_ptr<DeviceBuffer> linearization_input_buffer_gpu;
  std::unique_ptr<DeviceBuffer> linearization_output_buffer_gpu;

  std::vector<unsigned char, Eigen::aligned_allocator<unsigned char>> evaluation_input_buffer_cpu;
  std::vector<unsigned char, Eigen::aligned_allocator<unsigned char>> evaluation_output_buffer_cpu;
  std::unique_ptr<DeviceBuffer> evaluation_input_buffer_gpu;
  std::unique_ptr<DeviceBuffer> evaluation_output_buffer_gpu;
};

}  // namespace gtsam_points
