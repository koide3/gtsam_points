#pragma once

#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace thrust {

template <typename T>
class device_ptr;
}

namespace gtsam_ext {

class NonlinearFactorGPU : public gtsam::NonlinearFactor {
public:
  template <typename CONTAINER>
  NonlinearFactorGPU(const CONTAINER& keys) : gtsam::NonlinearFactor(keys) {}
  virtual ~NonlinearFactorGPU() {}

  virtual size_t linearization_input_size() const = 0;
  virtual size_t linearization_output_size() const = 0;
  virtual size_t evaluation_input_size() const = 0;
  virtual size_t evaluation_output_size() const = 0;

  virtual void set_linearization_point(const gtsam::Values& values, void* lin_input_cpu) = 0;
  virtual void issue_linearize(const void* lin_input_cpu, const thrust::device_ptr<const void>& lin_input_gpu, const thrust::device_ptr<void>& lin_output_gpu) = 0;
  virtual void store_linearized(const void* lin_output_cpu) = 0;

  virtual void set_evaluation_point(const gtsam::Values& values, void* eval_input_cpu) = 0;
  virtual void issue_compute_error(
    const void* lin_input_cpu,
    const void* eval_input_cpu,
    const thrust::device_ptr<const void>& lin_input_gpu,
    const thrust::device_ptr<const void>& eval_input_gpu,
    const thrust::device_ptr<void>& eval_output_gpu) = 0;
  virtual void store_computed_error(const void* eval_output_cpu) = 0;

  virtual void sync() = 0;
};

}