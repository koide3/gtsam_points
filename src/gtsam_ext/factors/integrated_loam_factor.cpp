#include <gtsam_ext/ann/ivox.hpp>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/integrated_loam_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_loam_factor_impl.hpp>

template class gtsam_ext::IntegratedLOAMFactor_<gtsam_ext::Frame, gtsam_ext::Frame>;
template class gtsam_ext::IntegratedPointToPlaneFactor_<gtsam_ext::Frame, gtsam_ext::Frame>;
template class gtsam_ext::IntegratedPointToEdgeFactor_<gtsam_ext::Frame, gtsam_ext::Frame>;

template class gtsam_ext::IntegratedLOAMFactor_<gtsam_ext::iVox, gtsam_ext::Frame>;
template class gtsam_ext::IntegratedPointToPlaneFactor_<gtsam_ext::iVox, gtsam_ext::Frame>;
template class gtsam_ext::IntegratedPointToEdgeFactor_<gtsam_ext::iVox, gtsam_ext::Frame>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedLOAMFactor_<gtsam_ext::DummyFrame, gtsam_ext::DummyFrame>;
template class gtsam_ext::IntegratedPointToPlaneFactor_<gtsam_ext::DummyFrame, gtsam_ext::DummyFrame>;
template class gtsam_ext::IntegratedPointToEdgeFactor_<gtsam_ext::DummyFrame, gtsam_ext::DummyFrame>;
