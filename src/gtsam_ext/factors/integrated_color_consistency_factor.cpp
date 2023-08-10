#include <gtsam_ext/ann/ivox.hpp>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/factors/intensity_gradients_ivox.hpp>
#include <gtsam_ext/factors/integrated_color_consistency_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_color_consistency_factor_impl.hpp>

template class gtsam_ext::IntegratedColorConsistencyFactor_<gtsam_ext::iVox, gtsam_ext::PointCloud, gtsam_ext::IntensityGradients>;
template class gtsam_ext::IntegratedColorConsistencyFactor_<gtsam_ext::PointCloud, gtsam_ext::PointCloud, gtsam_ext::IntensityGradients>;
template class gtsam_ext::IntegratedColorConsistencyFactor_<gtsam_ext::IntensityGradientsiVox, gtsam_ext::PointCloud, gtsam_ext::IntensityGradientsiVox>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedColorConsistencyFactor_<gtsam_ext::DummyFrame, gtsam_ext::DummyFrame>;
