#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_colored_gicp_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_colored_gicp_factor_impl.hpp>

template class gtsam_ext::IntegratedColoredGICPFactor_<gtsam_ext::Frame, gtsam_ext::Frame>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedColoredGICPFactor_<gtsam_ext::DummyFrame, gtsam_ext::DummyFrame>;
