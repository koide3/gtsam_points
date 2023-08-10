#include <gtsam_ext/ann/ivox.hpp>
#include <gtsam_ext/types/point_cloud.hpp>
#include <gtsam_ext/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_ext/factors/impl/integrated_ct_gicp_factor_impl.hpp>

template class gtsam_ext::IntegratedCT_GICPFactor_<gtsam_ext::iVox, gtsam_ext::PointCloud>;
template class gtsam_ext::IntegratedCT_GICPFactor_<gtsam_ext::PointCloud, gtsam_ext::PointCloud>;

#include <gtsam_ext/types/dummy_frame.hpp>
template class gtsam_ext::IntegratedCT_GICPFactor_<gtsam_ext::DummyFrame, gtsam_ext::DummyFrame>;
