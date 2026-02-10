// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_gicp_factor_gpu.hpp>

#include <cuda_runtime.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>

#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/factors/integrated_gicp_derivatives.cuh>

namespace gtsam_points {

IntegratedGICPFactorGPU::IntegratedGICPFactorGPU(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const PointCloud::ConstPtr& target,
  const PointCloud::ConstPtr& source,
  const KdTreeGPU::ConstPtr& target_tree,
  CUstream_st* stream,
  std::shared_ptr<TempBufferManager> temp_buffer)
: gtsam_points::NonlinearFactorGPU(gtsam::KeyVector{target_key, source_key}),
  is_binary(true),
  fixed_target_pose(Eigen::Isometry3f::Identity()),
  target(target),
  source(source),
  target_tree(target_tree),
  derivatives(new IntegratedGICPDerivatives(target, source, target_tree, stream, temp_buffer)),
  linearized(false),
  linearization_point(Eigen::Isometry3f::Identity()) {
  //
  if (!source->points_gpu) {
    std::cerr << "error: GPU source points have not been allocated!!" << std::endl;
    abort();
  }

  if (!source->covs_gpu) {
    std::cerr << "error: GPU source covs have not been allocated!!" << std::endl;
    abort();
  }

  if (!target->points_gpu) {
    std::cerr << "error: GPU target points have not been allocated!!" << std::endl;
    abort();
  }

  if (!target->covs_gpu) {
    std::cerr << "error: GPU target covs have not been allocated!!" << std::endl;
    abort();
  }

  if (!target_tree) {
    std::cerr << "error: GPU target kdtree has not been created!!" << std::endl;
    abort();
  }
}

IntegratedGICPFactorGPU::IntegratedGICPFactorGPU(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const PointCloud::ConstPtr& target,
  const PointCloud::ConstPtr& source,
  const KdTreeGPU::ConstPtr& target_tree,
  CUstream_st* stream,
  std::shared_ptr<TempBufferManager> temp_buffer)
: gtsam_points::NonlinearFactorGPU(gtsam::KeyVector{source_key}),
  is_binary(false),
  fixed_target_pose(fixed_target_pose.matrix().cast<float>()),
  target(target),
  source(source),
  target_tree(target_tree),
  derivatives(new IntegratedGICPDerivatives(target, source, target_tree, stream, temp_buffer)),
  linearized(false),
  linearization_point(Eigen::Isometry3f::Identity()) {
  //
  if (!source->points_gpu) {
    std::cerr << "error: GPU source points have not been allocated!!" << std::endl;
    abort();
  }

  if (!source->covs_gpu) {
    std::cerr << "error: GPU source covs have not been allocated!!" << std::endl;
    abort();
  }

  if (!target->points_gpu) {
    std::cerr << "error: GPU target points have not been allocated!!" << std::endl;
    abort();
  }

  if (!target->covs_gpu) {
    std::cerr << "error: GPU target covs have not been allocated!!" << std::endl;
    abort();
  }

  if (!target_tree) {
    std::cerr << "error: GPU target kdtree has not been created!!" << std::endl;
    abort();
  }
}

IntegratedGICPFactorGPU::~IntegratedGICPFactorGPU() {}

void IntegratedGICPFactorGPU::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedGICPFactorGPU";
  if (is_binary) {
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;
  } else {
    std::cout << "(fixed, " << keyFormatter(this->keys()[0]) << ")" << std::endl;
  }

  std::cout << "|target|=" << frame::size(*target) << "pts, |source|=" << frame::size(*source) << "pts" << std::endl;
}

size_t IntegratedGICPFactorGPU::memory_usage() const {
  return sizeof(*this) + sizeof(IntegratedGICPDerivatives);
}

size_t IntegratedGICPFactorGPU::memory_usage_gpu() const {
  return sizeof(Eigen::Isometry3f) + sizeof(int) + sizeof(std::pair<int, int>) * derivatives->get_num_inliers();
}

void IntegratedGICPFactorGPU::set_enable_offloading(bool enable) {
  derivatives->set_enable_offloading(enable);
}

void IntegratedGICPFactorGPU::set_enable_surface_validation(bool enable) {
  derivatives->set_enable_surface_validation(enable);
}

void IntegratedGICPFactorGPU::set_inlier_update_thresh(double trans, double angle) {
  derivatives->set_inlier_update_thresh(trans, angle);
}

int IntegratedGICPFactorGPU::num_inliers() const {
  return derivatives->get_num_inliers();
}

double IntegratedGICPFactorGPU::inlier_fraction() const {
  return derivatives->get_num_inliers() / static_cast<double>(frame::size(*source));
}

gtsam::NonlinearFactor::shared_ptr IntegratedGICPFactorGPU::clone() const {
  if (is_binary) {
    return gtsam::make_shared<IntegratedGICPFactorGPU>(keys()[0], keys()[1], target, source, target_tree, nullptr, nullptr);
  }

  return gtsam::make_shared<IntegratedGICPFactorGPU>(
    gtsam::Pose3(fixed_target_pose.cast<double>().matrix()),
    keys()[0],
    target,
    source,
    target_tree,
    nullptr,
    nullptr);
}

size_t IntegratedGICPFactorGPU::linearization_input_size() const {
  return sizeof(Eigen::Isometry3f);
}

size_t IntegratedGICPFactorGPU::linearization_output_size() const {
  return sizeof(LinearizedSystem6);
}

size_t IntegratedGICPFactorGPU::evaluation_input_size() const {
  return sizeof(Eigen::Isometry3f);
}

size_t IntegratedGICPFactorGPU::evaluation_output_size() const {
  return sizeof(float);
}

Eigen::Isometry3f IntegratedGICPFactorGPU::calc_delta(const gtsam::Values& values) const {
  if (!is_binary) {
    gtsam::Pose3 source_pose = values.at<gtsam::Pose3>(keys_[0]);
    gtsam::Pose3 delta = gtsam::Pose3(fixed_target_pose.inverse().cast<double>().matrix()) * source_pose;
    return Eigen::Isometry3f(delta.matrix().cast<float>());
  }

  gtsam::Pose3 target_pose = values.at<gtsam::Pose3>(keys_[0]);
  gtsam::Pose3 source_pose = values.at<gtsam::Pose3>(keys_[1]);
  gtsam::Pose3 delta = target_pose.inverse() * source_pose;

  return Eigen::Isometry3f(delta.matrix().cast<float>());
}

double IntegratedGICPFactorGPU::error(const gtsam::Values& values) const {
  double err;
  if (evaluation_result) {
    err = evaluation_result.value();
    evaluation_result = {};
  } else {
    std::cerr << "warning: computing error in sync mode seriously affects the processing speed!!" << std::endl;

    if (!linearized) {
      linearize(values);
    }

    Eigen::Isometry3f evaluation_point = calc_delta(values);
    err = derivatives->compute_error(linearization_point, evaluation_point);
  }

  return err;
}

gtsam::GaussianFactor::shared_ptr IntegratedGICPFactorGPU::linearize(const gtsam::Values& values) const {
  linearized = true;
  linearization_point = calc_delta(values);

  LinearizedSystem6 l;

  if (linearization_result) {
    l = *linearization_result;
    linearization_result.reset();
  } else {
    std::cerr << "warning: performing linearization in sync mode seriously affects the processing speed!!" << std::endl;
    l = derivatives->linearize(linearization_point);
  }

  gtsam::HessianFactor::shared_ptr factor;

  if (is_binary) {
    factor.reset(new gtsam::HessianFactor(
      keys_[0],
      keys_[1],
      l.H_target.cast<double>(),
      l.H_target_source.cast<double>(),
      -l.b_target.cast<double>(),
      l.H_source.cast<double>(),
      -l.b_source.cast<double>(),
      l.error));
  } else {
    factor.reset(new gtsam::HessianFactor(keys_[0], l.H_source.cast<double>(), -l.b_source.cast<double>(), l.error));
  }

  return factor;
}

void IntegratedGICPFactorGPU::set_linearization_point(const gtsam::Values& values, void* lin_input_cpu) {
  // If -march=native is used and some GPU factor requests a storage size that causes memory misalignment,
  // the following lines that directly operates on the input buffer may cause segfaults
  Eigen::Isometry3f* linearization_point = reinterpret_cast<Eigen::Isometry3f*>(lin_input_cpu);
  *linearization_point = calc_delta(values);
}

void IntegratedGICPFactorGPU::set_evaluation_point(const gtsam::Values& values, void* eval_input_cpu) {
  Eigen::Isometry3f* evaluation_point = reinterpret_cast<Eigen::Isometry3f*>(eval_input_cpu);
  *evaluation_point = calc_delta(values);
}

void IntegratedGICPFactorGPU::issue_linearize(const void* lin_input_cpu, const void* lin_input_gpu, void* lin_output_gpu) {
  auto linearization_point = reinterpret_cast<const Eigen::Isometry3f*>(lin_input_cpu);
  auto linearization_point_gpu = reinterpret_cast<const Eigen::Isometry3f*>(lin_input_gpu);
  auto linearized_gpu = reinterpret_cast<LinearizedSystem6*>(lin_output_gpu);

  derivatives->reset_inliers(*linearization_point, linearization_point_gpu);
  derivatives->issue_linearize(linearization_point_gpu, linearized_gpu);
}

void IntegratedGICPFactorGPU::store_linearized(const void* lin_output_cpu) {
  auto linearized = reinterpret_cast<const LinearizedSystem6*>(lin_output_cpu);
  linearization_result.reset(new LinearizedSystem6(*linearized));
  evaluation_result = linearized->error;

  derivatives->update_inliers(linearized->num_inliers);
}

void IntegratedGICPFactorGPU::issue_compute_error(
  const void* lin_input_cpu,
  const void* eval_input_cpu,
  const void* lin_input_gpu,
  const void* eval_input_gpu,
  void* eval_output_gpu) {
  //
  auto linearization_point = reinterpret_cast<const Eigen::Isometry3f*>(lin_input_cpu);
  auto evaluation_point = reinterpret_cast<const Eigen::Isometry3f*>(eval_input_cpu);

  auto linearization_point_gpu = reinterpret_cast<const Eigen::Isometry3f*>(lin_input_gpu);
  auto evaluation_point_gpu = reinterpret_cast<const Eigen::Isometry3f*>(eval_input_gpu);

  auto error_gpu = reinterpret_cast<float*>(eval_output_gpu);

  derivatives->issue_compute_error(linearization_point_gpu, evaluation_point_gpu, error_gpu);
}

void IntegratedGICPFactorGPU::store_computed_error(const void* eval_output_cpu) {
  auto evaluated = reinterpret_cast<const float*>(eval_output_cpu);
  evaluation_result = *evaluated;
}

void IntegratedGICPFactorGPU::sync() {
  derivatives->sync_stream();
}
}  // namespace gtsam_points
