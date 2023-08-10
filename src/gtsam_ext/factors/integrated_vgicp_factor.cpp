#include <gtsam_ext/types/point_cloud.hpp>
#include <gtsam_ext/factors/integrated_vgicp_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_vgicp_factor_impl.hpp>

template class gtsam_ext::IntegratedVGICPFactor_<gtsam_ext::PointCloud>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedVGICPFactor_<gtsam_ext::DummyFrame>;
