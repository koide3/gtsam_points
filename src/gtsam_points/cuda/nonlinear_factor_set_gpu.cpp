// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/nonlinear_factor_set_gpu.hpp>

#include <cuda_runtime.h>
#include <gtsam_points/cuda/check_error.cuh>

namespace gtsam_points {

NonlinearFactorSetGPU::DeviceBuffer::DeviceBuffer() : size(0), buffer(nullptr) {}

NonlinearFactorSetGPU::DeviceBuffer::~DeviceBuffer() {
  if (buffer) {
    check_error << cudaFreeAsync(buffer, 0);
  }
}

void NonlinearFactorSetGPU::DeviceBuffer::resize(size_t size, CUstream_st* stream) {
  if (this->size < size) {
    if (buffer) {
      check_error << cudaFreeAsync(buffer, stream);
    }
    check_error << cudaMallocAsync(&buffer, size, stream);
    this->size = size;
  }
}

NonlinearFactorSetGPU::NonlinearFactorSetGPU() {
  check_error << cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  linearization_input_buffer_gpu.reset(new DeviceBuffer);
  linearization_output_buffer_gpu.reset(new DeviceBuffer);
  evaluation_input_buffer_gpu.reset(new DeviceBuffer);
  evaluation_output_buffer_gpu.reset(new DeviceBuffer);
}

NonlinearFactorSetGPU::~NonlinearFactorSetGPU() {
  check_error << cudaStreamDestroy(stream);
}

void NonlinearFactorSetGPU::clear_counts() {
  num_linearizations = 0;
  num_evaluations = 0;
}

bool NonlinearFactorSetGPU::add(boost::shared_ptr<gtsam::NonlinearFactor> factor) {
  auto gpu_factor = boost::dynamic_pointer_cast<NonlinearFactorGPU>(factor);
  if (gpu_factor) {
    factors.push_back(gpu_factor);
    return true;
  }

  return false;
}

void NonlinearFactorSetGPU::add(const gtsam::NonlinearFactorGraph& factors) {
  for (auto& factor : factors) {
    add(factor);
  }
}

void NonlinearFactorSetGPU::linearize(const gtsam::Values& linearization_point) {
  if (factors.empty()) {
    return;
  }
  num_linearizations += size();

  size_t input_buffer_size = 0;
  size_t output_buffer_size = 0;
  for (const auto& factor : factors) {
    input_buffer_size += factor->linearization_input_size();
    output_buffer_size += factor->linearization_output_size();
  }

  linearization_input_buffer_cpu.resize(input_buffer_size);
  linearization_input_buffer_gpu->resize(input_buffer_size, stream);
  linearization_output_buffer_cpu.resize(output_buffer_size);
  linearization_output_buffer_gpu->resize(output_buffer_size, stream);

  // set linearization point
  size_t input_cursor = 0;
  size_t output_cursor = 0;
  for (auto& factor : factors) {
    auto input_cpu = linearization_input_buffer_cpu.data() + input_cursor;
    factor->set_linearization_point(linearization_point, input_cpu);
    input_cursor += factor->linearization_input_size();
  }

  // copy input buffer from cpu to gpu
  check_error << cudaMemcpyAsync(
    linearization_input_buffer_gpu->data(),
    linearization_input_buffer_cpu.data(),
    input_buffer_size,
    cudaMemcpyHostToDevice,
    stream);
  check_error << cudaStreamSynchronize(stream);

  // issue linearization tasks
  input_cursor = 0;
  output_cursor = 0;
  for (auto& factor : factors) {
    auto input_cpu = linearization_input_buffer_cpu.data() + input_cursor;
    auto input_gpu = linearization_input_buffer_gpu->data() + input_cursor;
    auto output_gpu = linearization_output_buffer_gpu->data() + output_cursor;
    factor->issue_linearize(input_cpu, input_gpu, output_gpu);
    input_cursor += factor->linearization_input_size();
    output_cursor += factor->linearization_output_size();
  }

  // synchronize
  for (auto& factor : factors) {
    factor->sync();
  }

  // copy output buffer from gpu to cpu
  check_error << cudaMemcpyAsync(
    linearization_output_buffer_cpu.data(),
    linearization_output_buffer_gpu->data(),
    output_buffer_size,
    cudaMemcpyDeviceToHost,
    stream);
  check_error << cudaStreamSynchronize(stream);

  // store calculated results
  input_cursor = 0;
  output_cursor = 0;
  for (auto& factor : factors) {
    auto output_cpu = linearization_output_buffer_cpu.data() + output_cursor;
    factor->store_linearized(output_cpu);
    output_cursor += factor->linearization_output_size();
  }

  // synchronize
  for (auto& factor : factors) {
    factor->sync();
  }
}

void NonlinearFactorSetGPU::error(const gtsam::Values& values) {
  if (factors.empty()) {
    return;
  }
  num_evaluations += size();

  size_t input_buffer_size = 0;
  size_t output_buffer_size = 0;

  for (const auto& factor : factors) {
    input_buffer_size += factor->evaluation_input_size();
    output_buffer_size += factor->evaluation_output_size();
  }
  evaluation_input_buffer_cpu.resize(input_buffer_size);
  evaluation_input_buffer_gpu->resize(input_buffer_size, stream);
  evaluation_output_buffer_cpu.resize(output_buffer_size);
  evaluation_output_buffer_gpu->resize(output_buffer_size, stream);

  // set evaluation point
  size_t lin_input_cursor = 0;
  size_t eval_input_cursor = 0;
  size_t eval_output_cursor = 0;

  for (auto& factor : factors) {
    auto input_cpu = evaluation_input_buffer_cpu.data() + eval_input_cursor;
    factor->set_evaluation_point(values, input_cpu);
    eval_input_cursor += factor->evaluation_input_size();
  }

  // copy input buffer from cpu to gpu
  check_error
    << cudaMemcpyAsync(evaluation_input_buffer_gpu->data(), evaluation_input_buffer_cpu.data(), input_buffer_size, cudaMemcpyHostToDevice, stream);
  check_error << cudaStreamSynchronize(stream);

  // issue error computation
  lin_input_cursor = 0;
  eval_input_cursor = 0;
  eval_output_cursor = 0;
  for (auto& factor : factors) {
    auto lin_input_cpu = linearization_input_buffer_cpu.data() + lin_input_cursor;
    auto lin_input_gpu = linearization_input_buffer_gpu->data() + lin_input_cursor;
    auto eval_input_cpu = evaluation_input_buffer_cpu.data() + eval_input_cursor;
    auto eval_input_gpu = evaluation_input_buffer_gpu->data() + eval_input_cursor;
    auto eval_output_gpu = evaluation_output_buffer_gpu->data() + eval_output_cursor;

    factor->issue_compute_error(lin_input_cpu, eval_input_cpu, lin_input_gpu, eval_input_gpu, eval_output_gpu);

    lin_input_cursor += factor->linearization_input_size();
    eval_input_cursor += factor->evaluation_input_size();
    eval_output_cursor += factor->evaluation_output_size();
  }

  // synchronize
  for (auto& factor : factors) {
    factor->sync();
  }

  // copy output buffer from gpu to cpu
  check_error
    << cudaMemcpyAsync(evaluation_output_buffer_cpu.data(), evaluation_output_buffer_gpu->data(), output_buffer_size, cudaMemcpyDeviceToHost, stream);
  check_error << cudaStreamSynchronize(stream);

  // store computed results
  lin_input_cursor = 0;
  eval_input_cursor = 0;
  eval_output_cursor = 0;

  for (auto& factor : factors) {
    auto output_cpu = evaluation_output_buffer_cpu.data() + eval_output_cursor;
    factor->store_computed_error(output_cpu);
    eval_output_cursor += factor->evaluation_output_size();
  }

  // synchronize
  for (auto& factor : factors) {
    factor->sync();
  }
}

std::vector<gtsam::GaussianFactor::shared_ptr> NonlinearFactorSetGPU::calc_linear_factors(const gtsam::Values& linearization_point) {
  linearize(linearization_point);
  std::vector<gtsam::GaussianFactor::shared_ptr> linear_factors(factors.size());
  for (int i = 0; i < factors.size(); i++) {
    linear_factors[i] = factors[i]->linearize(linearization_point);
  }

  return linear_factors;
}

}  // namespace gtsam_points