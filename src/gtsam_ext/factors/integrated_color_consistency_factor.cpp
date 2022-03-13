#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_color_consistency_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_color_consistency_factor_impl.hpp>

template class gtsam_ext::IntegratedColorConsistencyFactor<gtsam_ext::Frame>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedColorConsistencyFactor<gtsam_ext::DummyFrame>;
