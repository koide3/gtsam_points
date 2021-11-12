#pragma once

#ifdef BUILD_GTSAM_EXT_GPU

#include <memory>
#include <vector>

#include <thrust/device_vector.h>

#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_ext/factors/nonlinear_factor_gpu.hpp>

namespace gtsam_ext {

/**
 * @brief This class holds a set of GPU-based NonlinearFactors and manages their linearization and cost evaluation tasks
 */
class NonlinearFactorSetGPU {
public:
  NonlinearFactorSetGPU();
  ~NonlinearFactorSetGPU();

  /**
   * @brief Number of GPU factors in the set
   * @return Number of GPU factors in the set
   */
  int size() const { return factors.size(); }

  /**
   * @brief Remove all the factors from the set
   */
  void clear() { factors.clear(); }

  /**
   * @brief Reset the linearization and cost evaluation counts
   */
  void clear_counts();

  int linearization_count() const { return num_linearizations; }
  int evaluation_count() const { return num_evaluations; }

  bool add(boost::shared_ptr<gtsam::NonlinearFactor> factor);
  void add(const gtsam::NonlinearFactorGraph& factors);
  void add(boost::shared_ptr<NonlinearFactorGPU> factor);

  void linearize(const gtsam::Values& linearization_point);
  void error(const gtsam::Values& values);

  std::vector<gtsam::GaussianFactor::shared_ptr> calc_linear_factors(const gtsam::Values& linearization_point);

private:
  int num_linearizations;
  int num_evaluations;

  std::vector<boost::shared_ptr<NonlinearFactorGPU>> factors;

  thrust::host_vector<unsigned char> linearization_input_buffer_cpu;
  thrust::host_vector<unsigned char> linearization_output_buffer_cpu;
  thrust::device_vector<unsigned char> linearization_input_buffer_gpu;
  thrust::device_vector<unsigned char> linearization_output_buffer_gpu;

  thrust::host_vector<unsigned char> evaluation_input_buffer_cpu;
  thrust::host_vector<unsigned char> evaluation_output_buffer_cpu;
  thrust::device_vector<unsigned char> evaluation_input_buffer_gpu;
  thrust::device_vector<unsigned char> evaluation_output_buffer_gpu;
};

}  // namespace gtsam_ext

#else

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace gtsam_ext {

class NonlinearFactorGPU;

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

  std::vector<gtsam::GaussianFactor::shared_ptr> calc_linear_factors(const gtsam::Values& linearization_point) { return std::vector<gtsam::GaussianFactor::shared_ptr>(); }
};

}  // namespace gtsam_ext

#endif
