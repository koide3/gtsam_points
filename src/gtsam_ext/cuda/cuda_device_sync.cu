#include <gtsam_ext/cuda/cuda_device_sync.hpp>

namespace gtsam_ext {

void cuda_device_synchronize() {
  cudaDeviceSynchronize();
}

}