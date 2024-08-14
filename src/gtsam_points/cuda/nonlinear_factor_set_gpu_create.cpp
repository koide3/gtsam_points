// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>

#include <cuda_runtime.h>
#include <gtsam_points/cuda/check_error.cuh>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu.hpp>

namespace gtsam_points {

std::shared_ptr<NonlinearFactorSet> create_nonlinear_factor_set_gpu() {
  return std::make_shared<NonlinearFactorSetGPU>();
}

}