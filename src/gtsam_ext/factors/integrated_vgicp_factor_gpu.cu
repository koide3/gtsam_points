#include <gtsam_ext/factors/integrated_vgicp_factor_gpu.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/HessianFactor.h>

#include <gtsam_ext/cuda/kernels/linearized_system.cuh>
#include <gtsam_ext/cuda/stream_temp_buffer_roundrobin.hpp>

class IntegratedVGICPDerivatives {};

namespace gtsam_ext {
IntegratedVGICPFactorGPU::IntegratedVGICPFactorGPU(gtsam::Key target_key, gtsam::Key source_key, const VoxelizedFrame::ConstPtr& target, const Frame::ConstPtr& source)
: IntegratedVGICPFactorGPU(target_key, source_key, target, source, nullptr, nullptr) {}

IntegratedVGICPFactorGPU::IntegratedVGICPFactorGPU(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const Frame>& target,
  const std::shared_ptr<const Frame>& source,
  CUstream_st* stream,
  std::shared_ptr<TempBufferManager> temp_buffer)
: gtsam_ext::NonlinearFactorGPU(gtsam::cref_list_of<2>(target_key)(source_key)) {}
}  // namespace gtsam_ext