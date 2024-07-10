#pragma once

#include <gtest/gtest.h>
#include <gtsam_points/types/point_cloud.hpp>

void compare_frames(const gtsam_points::PointCloud::ConstPtr& frame1, const gtsam_points::PointCloud::ConstPtr& frame2, const std::string& label = "") {
  ASSERT_NE(frame1, nullptr) << label;
  ASSERT_NE(frame2, nullptr) << label;

  EXPECT_EQ(frame1->size(), frame2->size()) << "frame size mismatch";

  if (frame1->points) {
    EXPECT_NE(frame2->points, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->points[i] - frame2->points[i]).norm(), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->points, frame2->points);
  }

  if (frame1->times) {
    EXPECT_NE(frame2->times, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT(abs(frame1->times[i] - frame2->times[i]), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->times, frame2->times);
  }

  if (frame1->normals) {
    EXPECT_NE(frame2->normals, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->normals[i] - frame2->normals[i]).norm(), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->normals, frame2->normals);
  }

  if (frame1->covs) {
    EXPECT_NE(frame2->covs, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT((frame1->covs[i] - frame2->covs[i]).norm(), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->covs, frame2->covs);
  }

  if (frame1->intensities) {
    EXPECT_NE(frame2->intensities, nullptr);
    for (int i = 0; i < frame1->size(); i++) {
      EXPECT_LT(abs(frame1->intensities[i] - frame2->intensities[i]), 1e-6) << label;
    }
  } else {
    EXPECT_EQ(frame1->intensities, frame2->intensities);
  }

  for (const auto& aux : frame1->aux_attributes) {
    const auto& name = aux.first;
    const size_t aux_size = aux.second.first;
    const char* aux1_ptr = reinterpret_cast<const char*>(aux.second.second);

    EXPECT_TRUE(frame2->aux_attributes.count(name));
    const auto found = frame2->aux_attributes.find(name);
    if (found == frame2->aux_attributes.end()) {
      continue;
    }

    EXPECT_EQ(found->second.first, aux_size);
    if (found->second.first != aux_size) {
      continue;
    }

    const char* aux2_ptr = reinterpret_cast<const char*>(found->second.second);
    EXPECT_TRUE(std::equal(aux1_ptr, aux1_ptr + aux_size * frame1->size(), aux2_ptr)) << label;
  }
}