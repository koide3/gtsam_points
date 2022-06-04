#include <chrono>
#include <iostream>
#include <filesystem>

#include <Eigen/Core>
#include <cuda_runtime_api.h>

#include <cub/cub.cuh>

#include <easy_profiler.hpp>

#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/types/frame_gpu.hpp>
#include <gtsam_ext/cuda/async_stream.hpp>

namespace gtsam_ext {

struct vector4d_to_3f_kernel {
  __host__ __device__ Eigen::Vector3f operator()(const Eigen::Vector4d& p) { return Eigen::Vector3f(p[0], p[1], p[2]); }
};

struct NewFrameGPU : public FrameCPU {
public:
  using Ptr = std::shared_ptr<NewFrameGPU>;
  using ConstPtr = std::shared_ptr<const NewFrameGPU>;

  NewFrameGPU() : FrameCPU() {}
  ~NewFrameGPU() {
    if (points_gpu) {
      cudaFreeAsync(points_gpu, 0);
    }

    if (covs_gpu) {
      cudaFreeAsync(covs_gpu, 0);
    }
  }

  template <typename T, int D>
  void add_points(const Eigen::Matrix<T, D, 1>* points, int num_points) {
    FrameCPU::add_points<T, D>(points, num_points);

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_f(num_points);
    std::transform(points, points + num_points, points_f.begin(), [](const Eigen::Vector4d& p) { return p.cast<float>().head<3>(); });

    cudaMallocAsync(&points_gpu, sizeof(Eigen::Vector3f) * num_points, 0);
    cudaMemcpyAsync(points_gpu, points_f.data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice, 0);
    cudaStreamSynchronize(0);
  }

  template <typename T, int D>
  void add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points) {
    FrameCPU::add_covs<T, D>(covs, num_points);

    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs_f(num_points);
    std::transform(covs, covs + num_points, covs_f.begin(), [](const Eigen::Matrix4d& c) { return c.cast<float>().block<3, 3>(0, 0); });

    cudaMallocAsync(&covs_gpu, sizeof(Eigen::Matrix3f) * num_points, 0);
    cudaMemcpyAsync(covs_gpu, covs_f.data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice, 0);
    cudaStreamSynchronize(0);
  }

  template <typename T, int D>
  void add_points_async(AsyncStream& stream, const Eigen::Matrix<T, D, 1>* points, int num_points) {
    FrameCPU::add_points<T, D>(points, num_points);

    auto points_f = std::make_shared<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>>(num_points);
    std::transform(points, points + num_points, points_f->data(), [](const Eigen::Vector4d& p) { return p.cast<float>().head<3>(); });
    stream.add_resource(points_f);

    cudaMallocAsync(&points_gpu, sizeof(Eigen::Vector3f) * num_points, stream);
    cudaMemcpyAsync(points_gpu, points_f->data(), sizeof(Eigen::Vector3f) * num_points, cudaMemcpyHostToDevice, stream);
  }

  template <typename T, int D>
  void add_covs_async(AsyncStream& stream, const Eigen::Matrix<T, D, D>* covs, int num_points) {
    FrameCPU::add_covs<T, D>(covs, num_points);

    auto covs_f = std::make_shared<std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>>(num_points);
    std::transform(covs, covs + num_points, covs_f->data(), [](const Eigen::Matrix4d& p) { return p.cast<float>().block<3, 3>(0, 0); });
    stream.add_resource(covs_f);

    cudaMallocAsync(&covs_gpu, sizeof(Eigen::Matrix3f) * num_points, stream);
    cudaMemcpyAsync(covs_gpu, covs_f->data(), sizeof(Eigen::Matrix3f) * num_points, cudaMemcpyHostToDevice, stream);
  }
};

}  // namespace gtsam_ext

void test() {
  glim::EasyProfiler prof("test");
  cudaDeviceSynchronize();

  std::vector<Eigen::Vector4d> points(8192 * 5);
  for (int i = 0; i < points.size(); i++) {
    points[i] << static_cast<double>(i) / points.size(), 0.0, 0.0, 1.0;
  }
  std::vector<Eigen::Matrix4d> covs(points.size());
  for (int i = 0; i < points.size(); i++) {
    covs[i].setIdentity();
  }

  const auto validate = [&](const gtsam_ext::Frame::ConstPtr& frame) {
    std::vector<Eigen::Vector3f> points_f(points.size());
    std::vector<Eigen::Matrix3f> covs_f(covs.size());

    cudaMemcpy(points_f.data(), frame->points_gpu, sizeof(Eigen::Vector3f) * frame->size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(covs_f.data(), frame->covs_gpu, sizeof(Eigen::Matrix3f) * frame->size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < points.size(); i++) {
      if ((points[i].head<3>() - points_f[i].cast<double>()).norm() > 1e-6) {
        std::cerr << "points mismatch!!" << std::endl;
      }

      if ((covs[i].block<3, 3>(0, 0) - covs_f[i].cast<double>()).norm() > 1e-6) {
        std::cerr << "covs mismatch!!" << std::endl;
      }
    }
  };

  const int count = 128;

  prof.push("FrameGPU");
  std::vector<gtsam_ext::Frame::Ptr> frames(count);

  for (int i = 0; i < frames.size(); i++) {
    auto frame = std::make_shared<gtsam_ext::FrameGPU>();
    frame->add_points(points);
    frame->add_covs(covs);
    frames[i] = frame;
  }

  prof.push("validate");
  for (const auto& frame : frames) {
    validate(frame);
  }

  prof.push("~FrameGPU");

  frames.clear();
  frames.resize(count);

  prof.push("NewFrameGPU");
  for (int i = 0; i < frames.size(); i++) {
    auto frame = std::make_shared<gtsam_ext::NewFrameGPU>();
    frame->add_points(points.data(), points.size());
    frame->add_covs(covs.data(), covs.size());
    frames[i] = frame;
  }

  prof.push("validate");
  for (const auto& frame : frames) {
    validate(frame);
  }

  prof.push("~NewFrameGPU");

  frames.clear();
  frames.resize(count);

  prof.push("create_streams");
  std::vector<gtsam_ext::AsyncStream> streams(8);

  prof.push("NewFrameGPU::add_points_async");
  for (int i = 0; i < frames.size(); i++) {
    auto& stream = streams[i % streams.size()];

    auto frame = std::make_shared<gtsam_ext::NewFrameGPU>();
    frame->add_points_async(stream, points.data(), points.size());
    frame->add_covs_async(stream, covs.data(), covs.size());
    frames[i] = frame;
  }

  prof.push("NewFrameGPU::sync");
  for (auto& stream : streams) {
    stream.sync();
  }

  prof.push("validate");
  for (const auto& frame : frames) {
    validate(frame);
  }

  prof.push("~NewFrameGPU");

  frames.clear();
  frames.resize(count);

  prof.push("done");
}

int main(int argc, char** argv) {
  test();
  test();
  test();

  /*
  cudaHostRegister(points.data(), sizeof(Eigen::Vector4f) * points.size(), cudaHostRegisterReadOnly);

  prof.push("sync");

  cudaDeviceSynchronize();

  prof.push("create streams");

  std::vector<std::unique_ptr<AsyncStream>> streams(10);
  for (auto& stream : streams) {
    stream.reset(new AsyncStream());
  }

  std::vector<float*> ptrs(streams.size());

  prof.push("memcpy");

  for (int i = 0; i < streams.size(); i++) {
    cudaMallocAsync(&ptrs[i], sizeof(Eigen::Vector4f) * points.size(), streams[i]->stream);
    cudaMemcpyAsync(ptrs[i], points.data(), sizeof(Eigen::Vector4f) * points.size(), cudaMemcpyHostToDevice, streams[i]->stream);
  }

  prof.push("sync");

  for (auto& stream : streams) {
    stream->sync();
  }

  prof.push("memcpy");

  for (int i = 0; i < streams.size(); i++) {
    cudaMemcpyAsync(ptrs[i], points.data(), sizeof(Eigen::Vector4f) * points.size(), cudaMemcpyHostToDevice, streams[i]->stream);
  }

  prof.push("sync");

  for (auto& stream : streams) {
    stream->sync();
  }

  prof.push("done");

  cudaHostUnregister(points.data());
  */

  return 0;
}