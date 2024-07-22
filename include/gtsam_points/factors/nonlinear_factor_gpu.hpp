// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam_points {

/**
 * @brief Base class for GPU-based nonlinear factors
 * @note  To efficiently perform linearization (and cost evaluation) on a GPU,
 *        we issue all linearization tasks of GPU-based factors and copy all
 *        required data for linearization (e.g., current estimate) at once.
 *
 *        To allow gtsam_points::NonlinearFactorSetGPU to manage linearization, you need to implement the following methods:
 *        - linearization_(input|output)_size() : Define the size of input and result data for linearization
 *        - set_linearization_point()           : Write the data to be uploaded to the GPU before linearization
 *        - issue_linearization()               : Issue the linearization task
 *        - sync()                              : Perform CPU-GPU synchronization to wait for linearization completion
 *        - store_linearized()                  : Read back the data from the download buffer after linearization
 *
 *        For example, implementation for the linearization of the standard ICP factor would be as follows:
 *        - linearization_input_size()          : sizeof(Eigen::Isometry3f)  Current estimate of T_target_source
 *        - linearization_output_size()         : sizeof(LinearizedSystem6)  Linearized Hessian factor
 *        - set_linearization_point()           : Write the current T_target_source to the lin_input_cpu
 *        - issue_linearization()               : Issue the ICP computation task
 *        - sync()                              : Wait for the ICP computation task
 *        - store_linearized()                  : Read the linearized factor from lin_output_cpu
 *
 *        Optimizers in gtsam_points calls NonlinearFactorSetGPU's linearization routine before calling linearize() of each factor.
 *        You should thus store the linearized factor to a temporary member and just return it when linearize() is called.
 */
class NonlinearFactorGPU : public gtsam::NonlinearFactor {
public:
  template <typename CONTAINER>
  NonlinearFactorGPU(const CONTAINER& keys) : gtsam::NonlinearFactor(keys) {}
  virtual ~NonlinearFactorGPU() {}

  /**
   * @brief Size of data to be uploaded to the GPU before linearization
   * @return size_t
   */
  virtual size_t linearization_input_size() const = 0;

  /**
   * @brief Size of data to be downloaded from the GPU after linearization
   * @return size_t
   */
  virtual size_t linearization_output_size() const = 0;

  /**
   * @brief Size of data to be uploaded to the GPU before cost evaluation
   * @return size_t
   */
  virtual size_t evaluation_input_size() const = 0;

  /**
   * @brief Size of data to be downloaded from the GPU after cost evaluation
   * @return size_t
   */
  virtual size_t evaluation_output_size() const = 0;

  /**
   * @brief Write linearization input data to the upload buffer
   * @param values          Current estimate
   * @param lin_input_cpu   Data upload buffer (size == linearization_input_size)
   */
  virtual void set_linearization_point(const gtsam::Values& values, void* lin_input_cpu) = 0;

  /**
   * @brief Issue linearization task
   * @param lin_input_cpu    Linearization input data on the CPU memory (size == linearization_input_size)
   * @param lin_input_gpu    Linearization input data on the GPU memory (size == linearization_input_size)
   * @param lin_output_gpu   Output data destination on the GPU memory (size == linearization_output_size)
   */
  virtual void
  issue_linearize(const void* lin_input_cpu, const void* lin_input_gpu, void* lin_output_gpu) = 0;

  /**
   * @brief Read linearization output data from the download buffer
   * @param lin_output_cpu  Data download buffer (size == linearization_output_size)
   */
  virtual void store_linearized(const void* lin_output_cpu) = 0;

  /**
   * @brief Write cost evaluation input data to the upload buffer
   * @param values          Current estimate
   * @param eval_input_cpu  Data upload buffer (size == evaluation_input_size)
   */
  virtual void set_evaluation_point(const gtsam::Values& values, void* eval_input_cpu) = 0;

  /**
   * @brief Issue cost evaluation task
   * @param lin_input_cpu     Linearization input data on the CPU memory (size == linearization_input_size)
   * @param eval_input_cpu    Cost evaluation input data on the CPU memory (size == evaluation_input_size)
   * @param lin_input_gpu     Linearization input data on the GPU memory (size == linearization_input_size)
   * @param eval_input_gpu    Cost evaluation input data on the GPU memory (size == evaluation_input_size)
   * @param eval_output_gpu   Output data destination on the GPU memory (size == evaluation_output_size)
   */
  virtual void issue_compute_error(
    const void* lin_input_cpu,
    const void* eval_input_cpu,
    const void* lin_input_gpu,
    const void* eval_input_gpu,
    void* eval_output_gpu) = 0;

  /**
   * @brief Read cost evaluation output data from the download buffer
   * @param eval_output_cpu   Data down load buffer (size == evaluation_output_size)
   */
  virtual void store_computed_error(const void* eval_output_cpu) = 0;

  /**
   * @brief Perform CPU-GPU synchronization and wait for the task
   */
  virtual void sync() = 0;
};

}  // namespace gtsam_points