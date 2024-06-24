// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

void CUDACheckError::operator<<(cudaError_t error) const {
  if (error == cudaSuccess) {
    return;
  }

  const std::string error_name = cudaGetErrorName(error);
  const std::string error_string = cudaGetErrorString(error);

  std::cerr << "warning: " << error_name << std::endl;
  std::cerr << "       : " << error_string << std::endl;
}

CUDACheckError check_error;

}  // namespace gtsam_points