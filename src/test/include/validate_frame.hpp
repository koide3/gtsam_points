#pragma once

#include <gtest/gtest.h>
#include <gtsam_ext/types/point_cloud.hpp>

void validate_frame(const gtsam_ext::PointCloud::ConstPtr& frame) {
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

void validate_all_propaties(const gtsam_ext::PointCloud::ConstPtr& frame, bool test_aux = true) {
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