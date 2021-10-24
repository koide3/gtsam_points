#pragma once

#include <Eigen/Core>

namespace gtsam_ext {

template <typename DST_T, int DST_ROWS, int DST_COLS, typename SRC_T, int SRC_ROWS, int SRC_COLS>
void copy_data(Eigen::Matrix<DST_T, DST_ROWS, DST_COLS>& dst, const Eigen::Matrix<SRC_T, SRC_ROWS, SRC_COLS>& src) {
  constexpr int ROWS = std::min(DST_ROWS, DST_COLS);
  constexpr int COLS = std::min(DST_COLS, DST_COLS);
  dst.template block<ROWS, COLS>(0, 0) = src.template block<ROWS, COLS>(0, 0).template cast<DST_T>();
}

template <typename DST_T, typename SRC_T>
std::enable_if_t<std::is_floating_point_v<DST_T>, void> copy_data(DST_T& dst, const SRC_T src) {
  dst = static_cast<DST_T>(src);
}

template <typename T_GPU, typename T_CPU, typename DeviceVector>
void copy_to_gpu(DeviceVector& output_storate, T_GPU** output_gpu, const T_CPU* input_cpu, const T_GPU* input_gpu, int num_points) {
  if (!input_cpu && !input_gpu) {
    return;
  }

  output_storate.resize(num_points);
  *output_gpu = thrust::raw_pointer_cast(output_storate.data());

  if (input_gpu) {
    cudaMemcpy(*output_gpu, input_gpu, sizeof(T_GPU) * num_points, cudaMemcpyDeviceToDevice);
  } else {
    std::vector<T_GPU, Eigen::aligned_allocator<T_GPU>> floats(num_points);
    for (int i = 0; i < num_points; i++) {
      copy_data(floats[i], input_cpu[i]);
    }
    cudaMemcpy(*output_gpu, floats.data(), sizeof(T_GPU) * num_points, cudaMemcpyHostToDevice);
  }
}

template <typename T_CPU, typename T_GPU, typename Vector>
void copy_to_cpu(Vector& output_storage, T_CPU** output_cpu, const T_CPU* input_cpu, const T_GPU* input_gpu, int num_points, const T_CPU& init_val) {
  if (!input_cpu && !input_gpu) {
    return;
  }

  output_storage.resize(num_points, init_val);
  *output_cpu = output_storage.data();

  if (input_cpu) {
    memcpy(*output_cpu, input_cpu, sizeof(T_CPU) * num_points);
  } else {
    std::vector<T_GPU, Eigen::aligned_allocator<T_GPU>> floats(num_points);
    cudaMemcpy(floats.data(), input_gpu, sizeof(T_GPU) * num_points, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_points; i++) {
      copy_data((*output_cpu)[i], floats[i]);
    }
  }
}

}  // namespace gtsam_ext