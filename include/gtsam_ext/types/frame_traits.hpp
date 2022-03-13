// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <Eigen/Core>

namespace gtsam_ext {

namespace frame {

template <typename T>
struct traits {};

// int size(const T& t)
template <typename T>
auto size(const T& t) -> decltype(traits<T>::size(t), int()) {
  return traits<T>::size(t);
}

// bool has_times(const T& t)
template <typename T>
auto has_times(const T& t) -> decltype(traits<T>::has_times(t), bool()) {
  return traits<T>::has_times(t);
}

inline auto has_times(...) {
  return false;
}

// bool has_points(const T& t)
template <typename T>
auto has_points(const T& t) -> decltype(traits<T>::has_points(t), bool()) {
  return traits<T>::has_points(t);
}

inline auto has_points(...) {
  return false;
}

// bool has_normals(const T& t)
template <typename T>
auto has_normals(const T& t) -> decltype(traits<T>::has_normals(t), bool()) {
  return traits<T>::has_normals(t);
}

inline auto has_normals(...) {
  return false;
}

// bool has_covs(const T& t)
template <typename T>
auto has_covs(const T& t) -> decltype(traits<T>::has_covs(t), bool()) {
  return traits<T>::has_covs(t);
}

inline auto has_covs(...) {
  return false;
}

// bool has_intensities(const T& t)
template <typename T>
auto has_intensities(const T& t) -> decltype(traits<T>::has_intensities(t), bool()) {
  return traits<T>::has_intensities(t);
}

inline auto has_intensities(...) {
  return false;
}

// Point accessors
template <typename T>
double time(const T& t, int i) {
  return traits<T>::time(t, i);
}

template <typename T>
auto point(const T& t, int i) {
  return traits<T>::point(t, i);
}

template <typename T>
auto normal(const T& t, int i) {
  return traits<T>::normal(t, i);
}

template <typename T>
auto cov(const T& t, int i) {
  return traits<T>::cov(t, i);
}

template <typename T>
auto intensity(const T& t, int i) {
  return traits<T>::intensity(t, i);
}

template <typename T>
auto time_gpu(const T& t, int i) {
  return traits<T>::time_gpu(t, i);
}

template <typename T>
auto point_gpu(const T& t, int i) {
  return traits<T>::point_gpu(t, i);
}

template <typename T>
auto normal_gpu(const T& t, int i) {
  return traits<T>::normal_gpu(t, i);
}

template <typename T>
auto cov_gpu(const T& t, int i) {
  return traits<T>::cov_gpu(t, i);
}

template <typename T>
auto intensity_gpu(const T& t, int i) {
  return traits<T>::intensity_gpu(t, i);
}

// low-level interface
template <typename T>
auto points_ptr(const T& t) -> decltype(traits<T>::points_ptr(t), static_cast<const Eigen::Vector4d*>(nullptr)) {
  return traits<T>::points_ptr(t);
}

inline auto points_ptr(...) -> const Eigen::Vector4d* {
  return nullptr;
}

}  // namespace frame

}  // namespace gtsam_ext