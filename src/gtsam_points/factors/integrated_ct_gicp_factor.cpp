// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/ann/ivox_covariance_estimation.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_points/factors/impl/integrated_ct_gicp_factor_impl.hpp>

template class gtsam_points::IntegratedCT_GICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>;
template class gtsam_points::IntegratedCT_GICPFactor_<gtsam_points::iVoxCovarianceEstimation, gtsam_points::PointCloud>;
template class gtsam_points::IntegratedCT_GICPFactor_<gtsam_points::PointCloud, gtsam_points::PointCloud>;

#include <gtsam_points/types/dummy_frame.hpp>
template class gtsam_points::IntegratedCT_GICPFactor_<gtsam_points::DummyFrame, gtsam_points::DummyFrame>;
