// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/cuda/cuda_device_sync.hpp>

namespace gtsam_ext {

void cuda_device_synchronize() {
  cudaDeviceSynchronize();
}

}