#pragma once

#include <gtsam_ext/types/voxelized_frame.hpp>
#include <gtsam_ext/factors/nonlinear_factor_gpu.hpp>

struct CUstream_st;

namespace gtsam_ext {

class LinearizedSystem6;
class IntegratedVGICPDerivatives;
class TempBufferManager;

class IntegratedVGICPFactorGPU : public gtsam_ext::NonlinearFactorGPU {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedVGICPFactorGPU>;

  IntegratedVGICPFactorGPU(gtsam::Key target_key, gtsam::Key source_key, const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source);

  IntegratedVGICPFactorGPU(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const VoxelizedFrame::ConstPtr& target,
    const Frame::ConstPtr& source,
    CUstream_st* stream,
    std::shared_ptr<TempBufferManager> temp_buffer);

  virtual ~IntegratedVGICPFactorGPU() override;

  void set_inlier_update_thresh(double trans, double angle);

  // forbid copy
  IntegratedVGICPFactorGPU(const IntegratedVGICPFactorGPU&) = delete;
  IntegratedVGICPFactorGPU& operator=(const IntegratedVGICPFactorGPU&) = delete;

  virtual size_t dim() const override { return 6; }
  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

  virtual size_t linearization_input_size() const override;
  virtual size_t linearization_output_size() const override;
  virtual size_t evaluation_input_size() const override;
  virtual size_t evaluation_output_size() const override;

  virtual void set_linearization_point(const gtsam::Values& values, void* lin_input_cpu) override;
  virtual void issue_linearize(const void* lin_input_cpu, const thrust::device_ptr<const void>& lin_input_gpu, const thrust::device_ptr<void>& lin_output_gpu) override;
  virtual void store_linearized(const void* lin_output_cpu) override;

  virtual void set_evaluation_point(const gtsam::Values& values, void* eval_input_cpu) override;
  virtual void issue_compute_error(
    const void* lin_input_cpu,
    const void* eval_input_cpu,
    const thrust::device_ptr<const void>& lin_input_gpu,
    const thrust::device_ptr<const void>& eval_input_gpu,
    const thrust::device_ptr<void>& eval_output_gpu) override;
  virtual void store_computed_error(const void* eval_output_cpu) override;

  virtual void sync() override;

private:
  Eigen::Isometry3f calc_delta(const gtsam::Values& values) const;

private:
  bool is_binary;
  Eigen::Isometry3f fixed_target_pose;

  VoxelizedFrame::ConstPtr target;
  Frame::ConstPtr source;

  std::unique_ptr<IntegratedVGICPDerivatives> derivatives;

  mutable bool linearized;
  mutable Eigen::Isometry3f linearization_point;

  mutable boost::optional<float> evaluation_result;
  mutable std::unique_ptr<LinearizedSystem6> linearization_result;
};

}  // namespace gtsam_ext