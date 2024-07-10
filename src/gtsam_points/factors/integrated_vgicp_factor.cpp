// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/factors/impl/integrated_vgicp_factor_impl.hpp>

template class gtsam_points::IntegratedVGICPFactor_<gtsam_points::PointCloud>;

#include <gtsam_points/types/dummy_frame.hpp>
template class gtsam_points::IntegratedVGICPFactor_<gtsam_points::DummyFrame>;
