#include <gtsam_ext/cuda/nonlinear_factor_set_gpu_create.hpp>

#include <gtsam_ext/cuda/nonlinear_factor_set_gpu.hpp>

namespace gtsam_ext {

std::shared_ptr<NonlinearFactorSet> create_nonlinear_factor_set_gpu() {
  return std::make_shared<NonlinearFactorSetGPU>();
}

}