// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/types/point_cloud.hpp>
// #include <gtsam_points/factors/intensity_gradients_ivox.hpp>
#include <gtsam_points/factors/integrated_color_consistency_factor.hpp>
#include <gtsam_points/factors/impl/integrated_color_consistency_factor_impl.hpp>

template class gtsam_points::IntegratedColorConsistencyFactor_<gtsam_points::iVox, gtsam_points::PointCloud, gtsam_points::IntensityGradients>;
template class gtsam_points::IntegratedColorConsistencyFactor_<gtsam_points::PointCloud, gtsam_points::PointCloud, gtsam_points::IntensityGradients>;
// template class gtsam_points::IntegratedColorConsistencyFactor_<gtsam_points::IntensityGradientsiVox, gtsam_points::PointCloud,
// gtsam_points::IntensityGradientsiVox>;

#include <gtsam_points/types/dummy_frame.hpp>
template class gtsam_points::IntegratedColorConsistencyFactor_<gtsam_points::DummyFrame, gtsam_points::DummyFrame>;
