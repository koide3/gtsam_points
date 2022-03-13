#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_vgicp_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_vgicp_factor_impl.hpp>

template class gtsam_ext::IntegratedVGICPFactor<gtsam_ext::Frame>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedVGICPFactor<gtsam_ext::DummyFrame>;
