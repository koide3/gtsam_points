#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/future.h>

#include <gtsam_ext/types/voxelized_frame.hpp>

struct CUstream_st;

namespace gtsam_ext {

class LinearizedSystem6;
class TempBufferManager;

class IntegratedVGICPDerivatives {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IntegratedVGICPDerivatives(const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source, CUstream_st* ext_stream, std::shared_ptr<TempBufferManager> temp_buffer);
  ~IntegratedVGICPDerivatives();

  void set_inlier_update_thresh(double trans, double angle) {
    inlier_update_thresh_trans = trans;
    inlier_update_thresh_angle = angle;
  }

  // synchronized interface
  LinearizedSystem6 linearize(const Eigen::Isometry3f& x);
  double compute_error(const Eigen::Isometry3f& xl, const Eigen::Isometry3f& xe);
  void update_inliers(const Eigen::Isometry3f& x, const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr, bool force_update = false);

  // async interface
  void sync_stream();
  void issue_linearize(const thrust::device_ptr<const Eigen::Isometry3f>& x, const thrust::device_ptr<LinearizedSystem6>& output);
  void issue_compute_error(const thrust::device_ptr<const Eigen::Isometry3f>& xl, const thrust::device_ptr<const Eigen::Isometry3f>& xe, const thrust::device_ptr<float>& output);

private:
  double inlier_update_thresh_trans;
  double inlier_update_thresh_angle;

  bool external_stream;
  cudaStream_t stream;
  std::shared_ptr<TempBufferManager> temp_buffer;

  VoxelizedFrame::ConstPtr target;
  Frame::ConstPtr source;

  Eigen::Isometry3f inlier_evaluation_point;
  thrust::device_vector<int> source_inliears;
};
}