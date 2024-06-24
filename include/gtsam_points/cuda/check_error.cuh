// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <string>
#include <iostream>

#include <cuda_runtime.h>

namespace gtsam_points {

class CUDACheckError {
public:
  void operator<<(cudaError_t error) const;
};

extern CUDACheckError check_error;

}  // namespace gtsam_points