// SPDX-License-Identifier: MIT

#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/GaussianFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam_points/cuda/nonlinear_factor_set_gpu.hpp>
#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/util/gtsam_migration.hpp>

namespace gtsam_points {
namespace {

struct LinearizationStats {
  __host__ __device__ LinearizationStats operator+(const LinearizationStats& rhs) const {
    LinearizationStats sum;
    sum.system = system + rhs.system;
    sum.transformed_count = transformed_count + rhs.transformed_count;
    sum.source_index_sum = source_index_sum + rhs.source_index_sum;
    return sum;
  }

  __host__ __device__ static LinearizationStats zero() {
    LinearizationStats result;
    result.system = LinearizedSystem6::zero();
    result.transformed_count = 0;
    result.source_index_sum = 0;
    return result;
  }

  LinearizedSystem6 system;
  int transformed_count;
  int source_index_sum;
};

struct ErrorStats {
  __host__ __device__ ErrorStats operator+(const ErrorStats& rhs) const {
    return {error + rhs.error, transformed_count + rhs.transformed_count, source_index_sum + rhs.source_index_sum};
  }

  float error;
  int transformed_count;
  int source_index_sum;
};

struct collect_linearization_stats {
  __device__ LinearizationStats operator()(
    const thrust::pair<int, int>& correspondence,
    const LinearizedSystem6& linearized) const {
    //
    LinearizationStats result;
    result.system = linearized;
    result.transformed_count = 1;
    result.source_index_sum = correspondence.first;
    return result;
  }
};

struct collect_error_stats {
  __device__ ErrorStats operator()(const thrust::pair<int, int>& correspondence, float error) const {
    return {error, 1, correspondence.first};
  }
};

struct scale_linearization {
  explicit scale_linearization(float scale) : scale(scale) {}

  __device__ LinearizedSystem6 operator()(
    const thrust::pair<int, int>&,
    const LinearizedSystem6& linearized) const {
    //
    LinearizedSystem6 scaled = linearized;
    scaled.error *= scale;
    scaled.H_target *= scale;
    scaled.H_source *= scale;
    scaled.H_target_source *= scale;
    scaled.b_target *= scale;
    scaled.b_source *= scale;
    return scaled;
  }

  float scale;
};

struct scale_error {
  explicit scale_error(float scale) : scale(scale) {}

  __device__ float operator()(const thrust::pair<int, int>&, float error) const { return scale * error; }

  float scale;
};

class InspectableVGICPDerivatives : public IntegratedVGICPDerivatives {
public:
  using IntegratedVGICPDerivatives::IntegratedVGICPDerivatives;

  LinearizationStats linearize_with_stats(const Eigen::Isometry3f& x) {
    thrust::device_vector<Eigen::Isometry3f> x_device(1);
    thrust::device_vector<LinearizationStats> output_device(1);
    x_device[0] = x;

    const auto x_ptr = thrust::raw_pointer_cast(x_device.data());
    reset_inliers(x, x_ptr);
    issue_linearize_transform_reduce(
      x_ptr,
      thrust::raw_pointer_cast(output_device.data()),
      collect_linearization_stats(),
      LinearizationStats::zero());
    sync_stream();
    return output_device[0];
  }

  ErrorStats compute_error_with_stats(const Eigen::Isometry3f& xl, const Eigen::Isometry3f& xe) {
    thrust::device_vector<Eigen::Isometry3f> x_device(2);
    thrust::device_vector<ErrorStats> output_device(1);
    x_device[0] = xl;
    x_device[1] = xe;

    issue_compute_error_transform_reduce(
      thrust::raw_pointer_cast(x_device.data()),
      thrust::raw_pointer_cast(x_device.data() + 1),
      thrust::raw_pointer_cast(output_device.data()),
      collect_error_stats(),
      ErrorStats{0.0f, 0, 0});
    sync_stream();
    return output_device[0];
  }
};

class ScalingVGICPDerivatives : public IntegratedVGICPDerivatives {
public:
  ScalingVGICPDerivatives(
    const GaussianVoxelMapGPU::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    float scale)
  : IntegratedVGICPDerivatives(target, source, nullptr, nullptr), scale(scale) {}

  void issue_linearize(const Eigen::Isometry3f* d_x, LinearizedSystem6* d_output) override {
    issue_linearize_transform_reduce(d_x, d_output, scale_linearization(scale), LinearizedSystem6::zero());
  }

  void issue_compute_error(const Eigen::Isometry3f* d_xl, const Eigen::Isometry3f* d_xe, float* d_output) override {
    issue_compute_error_transform_reduce(d_xl, d_xe, d_output, scale_error(scale), 0.0f);
  }

private:
  float scale;
};

class ScalingVGICPFactor : public IntegratedVGICPFactorGPU {
public:
  ScalingVGICPFactor(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const GaussianVoxelMapGPU::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    float scale)
  : IntegratedVGICPFactorGPU(fixed_target_pose, source_key, target, source), source(source), scale(scale) {
    replace_derivatives(std::make_unique<ScalingVGICPDerivatives>(target, source, scale));
  }

  gtsam_points::shared_ptr<gtsam::NonlinearFactor> clone() const override {
    return gtsam::make_shared<ScalingVGICPFactor>(
      gtsam::Pose3(get_fixed_target_pose().cast<double>().matrix()),
      keys()[0],
      get_target(),
      source,
      scale);
  }

private:
  PointCloud::ConstPtr source;
  float scale;
};

class NonCloningScalingVGICPFactor : public IntegratedVGICPFactorGPU {
public:
  NonCloningScalingVGICPFactor(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const GaussianVoxelMapGPU::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    float scale)
  : IntegratedVGICPFactorGPU(fixed_target_pose, source_key, target, source) {
    replace_derivatives(std::make_unique<ScalingVGICPDerivatives>(target, source, scale));
  }
};

class VGICPDerivativesTransformReduceTest : public testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }

    const std::vector<Eigen::Vector3f> target_points{
      Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      Eigen::Vector3f(2.0f, 0.0f, 0.0f),
      Eigen::Vector3f(4.0f, 0.0f, 0.0f),
      Eigen::Vector3f(6.0f, 0.0f, 0.0f),
    };
    std::vector<Eigen::Matrix3f> target_covs(target_points.size(), 0.05f * Eigen::Matrix3f::Identity());

    auto target_frame = std::make_shared<PointCloudGPU>(target_points);
    target_frame->add_covs(target_covs);

    auto target_gpu = std::make_shared<GaussianVoxelMapGPU>(1.0f);
    target_gpu->insert(*target_frame);
    target = target_gpu;

    std::vector<Eigen::Vector3f> source_points = target_points;
    source_points.emplace_back(100.0f, 100.0f, 100.0f);
    std::vector<Eigen::Matrix3f> source_covs(source_points.size(), 0.05f * Eigen::Matrix3f::Identity());

    auto source_gpu = std::make_shared<PointCloudGPU>(source_points);
    source_gpu->add_covs(source_covs);
    source = source_gpu;
  }

  GaussianVoxelMapGPU::ConstPtr target;
  PointCloud::ConstPtr source;
};

TEST_F(VGICPDerivativesTransformReduceTest, CustomResultsRetainContextAndDefaultValues) {
  InspectableVGICPDerivatives derivatives(target, source, nullptr, nullptr);
  const Eigen::Isometry3f identity = Eigen::Isometry3f::Identity();

  const LinearizedSystem6 standard = derivatives.linearize(identity);
  const LinearizationStats transformed = derivatives.linearize_with_stats(identity);

  EXPECT_EQ(transformed.system.num_inliers, standard.num_inliers);
  EXPECT_FLOAT_EQ(transformed.system.error, standard.error);
  EXPECT_TRUE(transformed.system.H_target.isApprox(standard.H_target));
  EXPECT_TRUE(transformed.system.H_source.isApprox(standard.H_source));
  EXPECT_TRUE(transformed.system.H_target_source.isApprox(standard.H_target_source));
  EXPECT_TRUE(transformed.system.b_target.isApprox(standard.b_target));
  EXPECT_TRUE(transformed.system.b_source.isApprox(standard.b_source));

  EXPECT_EQ(transformed.transformed_count, 4);
  EXPECT_EQ(transformed.transformed_count, transformed.system.num_inliers);
  EXPECT_EQ(transformed.source_index_sum, 0 + 1 + 2 + 3);

  Eigen::Isometry3f evaluation = Eigen::Isometry3f::Identity();
  evaluation.translation().x() = 0.1f;
  const float standard_error = derivatives.compute_error(identity, evaluation);
  const ErrorStats transformed_error = derivatives.compute_error_with_stats(identity, evaluation);

  EXPECT_NEAR(transformed_error.error, standard_error, 1e-5f);
  EXPECT_EQ(transformed_error.transformed_count, transformed.transformed_count);
  EXPECT_EQ(transformed_error.source_index_sum, transformed.source_index_sum);
}

TEST_F(VGICPDerivativesTransformReduceTest, ReplacedDerivativesApplyToSynchronousFactorPaths) {
  constexpr float scale = 2.0f;
  IntegratedVGICPFactorGPU standard(gtsam::Pose3(), 0, target, source);
  ScalingVGICPFactor transformed(gtsam::Pose3(), 0, target, source, scale);

  Eigen::Isometry3d source_pose = Eigen::Isometry3d::Identity();
  source_pose.translation().x() = 0.1;
  gtsam::Values values;
  values.insert(0, gtsam::Pose3(source_pose.matrix()));

  const auto standard_linear = standard.linearize(values);
  const auto transformed_linear = transformed.linearize(values);
  EXPECT_TRUE(transformed_linear->information().isApprox(scale * standard_linear->information(), 1e-5));

  const double standard_error = standard.error(values);
  const double transformed_error = transformed.error(values);
  EXPECT_GT(standard_error, 0.0);
  EXPECT_NEAR(transformed_error, scale * standard_error, 1e-5);
}

TEST_F(VGICPDerivativesTransformReduceTest, ClonePreservesReplacedDerivatives) {
  constexpr float scale = 2.0f;
  auto original = gtsam::make_shared<ScalingVGICPFactor>(gtsam::Pose3(), 0, target, source, scale);
  const auto cloned = gtsam_points::dynamic_pointer_cast<ScalingVGICPFactor>(original->clone());
  ASSERT_TRUE(cloned);

  Eigen::Isometry3d source_pose = Eigen::Isometry3d::Identity();
  source_pose.translation().x() = 0.1;
  gtsam::Values values;
  values.insert(0, gtsam::Pose3(source_pose.matrix()));

  const auto original_linear = original->linearize(values);
  const auto cloned_linear = cloned->linearize(values);
  EXPECT_TRUE(cloned_linear->information().isApprox(original_linear->information(), 1e-5));
  EXPECT_NEAR(cloned->error(values), original->error(values), 1e-5);
}

TEST_F(VGICPDerivativesTransformReduceTest, InheritedCloneRejectsReplacedDerivatives) {
  NonCloningScalingVGICPFactor factor(gtsam::Pose3(), 0, target, source, 2.0f);
  EXPECT_THROW(factor.clone(), std::logic_error);
}

TEST_F(VGICPDerivativesTransformReduceTest, ReplacedDerivativesApplyToAsynchronousFactorPaths) {
  constexpr float scale = 2.0f;
  auto standard = gtsam::make_shared<IntegratedVGICPFactorGPU>(gtsam::Pose3(), 0, target, source);
  auto transformed = gtsam::make_shared<ScalingVGICPFactor>(gtsam::Pose3(), 1, target, source, scale);

  Eigen::Isometry3d source_pose = Eigen::Isometry3d::Identity();
  source_pose.translation().x() = 0.1;
  gtsam::Values values;
  values.insert(0, gtsam::Pose3(source_pose.matrix()));
  values.insert(1, gtsam::Pose3(source_pose.matrix()));

  NonlinearFactorSetGPU factor_set;
  ASSERT_TRUE(factor_set.add(standard));
  ASSERT_TRUE(factor_set.add(transformed));

  factor_set.linearize(values);
  const auto standard_linear = standard->linearize(values);
  const auto transformed_linear = transformed->linearize(values);
  EXPECT_TRUE(transformed_linear->information().isApprox(scale * standard_linear->information(), 1e-5));

  factor_set.error(values);
  const double standard_error = standard->error(values);
  const double transformed_error = transformed->error(values);
  EXPECT_GT(standard_error, 0.0);
  EXPECT_NEAR(transformed_error, scale * standard_error, 1e-5);
}

}  // namespace
}  // namespace gtsam_points
