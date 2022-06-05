#pragma once

#include <string>
#include <iostream>

namespace gtsam_ext {

class CUDACheckError {
public:
  void operator<<(cudaError_t error) const;
};

extern CUDACheckError check_error;

}  // namespace gtsam_ext