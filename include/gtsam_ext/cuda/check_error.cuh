#pragma once

#include <string>
#include <iostream>

namespace gtsam_ext {

class CUDACheckError {
public:
  void operator<<(cudaError_t error) const {
    if (error == cudaSuccess) {
      return;
    }

    const std::string error_name = cudaGetErrorName(error);
    const std::string error_string = cudaGetErrorString(error);

    std::cerr << "warning: " << error_name << std::endl;
    std::cerr << "       : " << error_string << std::endl;
  }
};

extern CUDACheckError check_error;

}  // namespace gtsam_ext