// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/cuda_device_prop.hpp>

namespace gtsam_points {

std::vector<std::string> cuda_device_names() {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);

  std::vector<std::string> names(num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    names[i] = prop.name;
  }

  return names;
}

}  // namespace gtsam_points
