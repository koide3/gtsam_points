// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2026  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/device_ptr.h>

namespace gtsam_points {

/**
 * @note Implementations taken from LossFunctions.cpp in GTSAM
 */

inline __host__ __device__ float huber_weight(float distance, float k)  {
  const float absError = fabs(distance);
  return (absError <= k) ? (1.0f) : (k / absError);
}

inline __host__ __device__ float huber_loss(float distance, float k) {
  const float absError = fabs(distance);
  if (absError <= k) {  // |x| <= k
    return distance*distance / 2.0f;
  } else { // |x| > k
    return k * (absError - (k/2));
  }
}

inline __host__ __device__ float cauchy_weight(float distance, float k) {
  return k * k / (k * k + distance*distance);
}

inline __host__ __device__ float cauchy_loss(float distance, float k) {
  const float val = log1p(distance * distance / (k * k));
  return (k * k) * val * 0.5f;
}

inline __host__ __device__ float geman_mcclure_weight_sqdist(float sq_distance, float c) {
  const float c2 = c*c;
  const float c4 = c2*c2;
  const float c2error = c2 + sq_distance;
  return c4/(c2error*c2error);
}

inline __host__ __device__ float geman_mcclure_weight(float distance, float c) {
  const float c2 = c*c;
  const float c4 = c2*c2;
  const float c2error = c2 + distance*distance;
  return c4/(c2error*c2error);
}

inline __host__ __device__ float geman_mcclure_scale_sqdist(float sq_distance, float c) {
  const float c2 = c*c;
  return c2 / (c2 + sq_distance);
}

inline __host__ __device__ float geman_mcclure_loss_sqdist(float sq_distance, float c) {
  const float c2 = c*c;
  return 0.5f * (c2 * sq_distance) / (c2 + sq_distance);
}

inline __host__ __device__ float geman_mcclure_loss(float distance, float c) {
  const float c2 = c*c;
  const float error2 = distance*distance;
  return 0.5f * (c2 * error2) / (c2 + error2);
}


}  // namespace gtsam_points
