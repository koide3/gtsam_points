#include <gtsam_ext/types/frame_traits.hpp>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_ct_icp_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_ct_icp_factor_impl.hpp>

template class gtsam_ext::IntegratedCT_ICPFactor_<gtsam_ext::Frame, gtsam_ext::Frame>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedCT_ICPFactor_<gtsam_ext::DummyFrame, gtsam_ext::DummyFrame>;
