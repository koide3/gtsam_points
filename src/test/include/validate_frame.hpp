#pragma once

#include <gtest/gtest.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/types/point_cloud.hpp>

void validate_frame(const gtsam_points::PointCloud::ConstPtr& frame) {
  if (frame->points) {
    for (int i = 0; i < frame->size(); i++) {
      EXPECT_DOUBLE_EQ(frame->points[i].w(), 1.0);
      EXPECT_TRUE(frame->points[i].array().isFinite().all());
    }
  }

  if (frame->normals) {
    for (int i = 0; i < frame->size(); i++) {
      EXPECT_DOUBLE_EQ(frame->normals[i].w(), 0.0);
      EXPECT_TRUE(frame->normals[i].array().isFinite().all());
    }
  }

  if (frame->covs) {
    for (int i = 0; i < frame->size(); i++) {
      EXPECT_TRUE(frame->covs[i].col(3).isZero());
      EXPECT_TRUE(frame->covs[i].row(3).isZero());
      EXPECT_TRUE(frame->covs[i].array().isFinite().all());
    }
  }

  if (frame->intensities) {
    for (int i = 0; i < frame->size(); i++) {
      EXPECT_TRUE(std::isfinite(frame->intensities[i]));
    }
  }

  if (frame->times) {
    for (int i = 0; i < frame->size(); i++) {
      EXPECT_TRUE(std::isfinite(frame->times[i]));
    }
  }
}

#ifdef GTSAM_POINTS_USE_CUDA
void validate_frame_gpu(const gtsam_points::PointCloud::ConstPtr& frame) {
  if (frame->points_gpu) {
    const auto points = gtsam_points::download_points_gpu(*frame);
    ASSERT_EQ(points.size(), frame->size());
    EXPECT_TRUE(std::all_of(points.begin(), points.end(), [](const Eigen::Vector3f& p) { return p.array().isFinite().all(); }));
  }

  if (frame->normals_gpu) {
    const auto normals = gtsam_points::download_normals_gpu(*frame);
    ASSERT_EQ(normals.size(), frame->size());
    EXPECT_TRUE(std::all_of(normals.begin(), normals.end(), [](const Eigen::Vector3f& p) { return p.array().isFinite().all(); }));
  }

  if (frame->covs_gpu) {
    const auto covs = gtsam_points::download_covs_gpu(*frame);
    ASSERT_EQ(covs.size(), frame->size());
    EXPECT_TRUE(std::all_of(covs.begin(), covs.end(), [](const Eigen::Matrix3f& c) { return c.array().isFinite().all(); }));
  }

  if (frame->intensities_gpu) {
    const auto intensities = gtsam_points::download_intensities_gpu(*frame);
    ASSERT_EQ(intensities.size(), frame->size());
    EXPECT_TRUE(std::all_of(intensities.begin(), intensities.end(), [](float x) { return std::isfinite(x); }));
  }

  if (frame->times_gpu) {
    const auto times = gtsam_points::download_times_gpu(*frame);
    ASSERT_EQ(times.size(), frame->size());
    EXPECT_TRUE(std::all_of(times.begin(), times.end(), [](float x) { return std::isfinite(x); }));
  }
}
#endif

void validate_all_propaties(const gtsam_points::PointCloud::ConstPtr& frame, bool test_aux = true) {
  ASSERT_TRUE(frame->points);
  ASSERT_TRUE(frame->normals);
  ASSERT_TRUE(frame->covs);
  ASSERT_TRUE(frame->intensities);
  ASSERT_TRUE(frame->times);

  for (int i = 0; i < frame->size(); i++) {
    EXPECT_TRUE(frame->points[i].array().isFinite().all());
    EXPECT_TRUE(frame->normals[i].array().isFinite().all());
    EXPECT_TRUE(frame->covs[i].array().isFinite().all());
    EXPECT_TRUE(std::isfinite(frame->intensities[i]));
    EXPECT_TRUE(std::isfinite(frame->times[i]));
  }

  if (!test_aux) {
    return;
  }

  const Eigen::Vector4d* aux1 = frame->aux_attribute<Eigen::Vector4d>("aux1");
  const Eigen::Matrix4d* aux2 = frame->aux_attribute<Eigen::Matrix4d>("aux2");
  ASSERT_TRUE(aux1);
  ASSERT_TRUE(aux2);
  for (int i = 0; i < frame->size(); i++) {
    EXPECT_TRUE(aux1[i].array().isFinite().all());
    EXPECT_TRUE(aux2[i].array().isFinite().all());
  }
};