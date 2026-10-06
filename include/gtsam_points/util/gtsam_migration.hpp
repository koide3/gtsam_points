// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <gtsam/config.h>
#if __has_include(<gtsam/base/make_shared.h>)
// Removed in GTSAM 4.3.0 (borglab/gtsam#2579), where std::make_shared handles over-aligned types (C++17).
#include <gtsam/base/make_shared.h>
#define GTSAM_POINTS_HAS_GTSAM_MAKE_SHARED
#endif
#include <gtsam/base/Matrix.h>

namespace gtsam_points {

#if GTSAM_VERSION_NUMERIC >= 40300

template <typename T>
using shared_ptr = std::shared_ptr<T>;

template <typename T>
using weak_ptr = std::weak_ptr<T>;

template <class T, class U>
auto dynamic_pointer_cast(const std::shared_ptr<U>& sp) -> std::shared_ptr<T> {
  return std::dynamic_pointer_cast<T>(sp);
}

template <typename T>
using optional = std::optional<T>;

using OptionalMatrixType = gtsam::Matrix*;
using OptionalMatrixVecType = std::vector<gtsam::Matrix>*;

constexpr auto NoneValue = nullptr;

#else

template <typename T>
using shared_ptr = boost::shared_ptr<T>;

template <typename T>
using weak_ptr = boost::weak_ptr<T>;

template <class T, class U>
auto dynamic_pointer_cast(const boost::shared_ptr<U>& sp) -> boost::shared_ptr<T> {
  return boost::dynamic_pointer_cast<T>(sp);
}

template <typename T>
using optional = boost::optional<T>;

using OptionalMatrixType = boost::optional<gtsam::Matrix&>;
using OptionalMatrixVecType = boost::optional<std::vector<gtsam::Matrix>&>;
inline const auto NoneValue = boost::none;

#endif

/// Allocate a gtsam_points::shared_ptr<T>, the smart pointer type used by GTSAM.
/// Not named make_shared to avoid ambiguities with std::make_shared and boost::make_shared.
template <typename T, typename... Args>
auto make_shared_ptr(Args&&... args) {
#ifdef GTSAM_POINTS_HAS_GTSAM_MAKE_SHARED
  return gtsam::make_shared<T>(std::forward<Args>(args)...);
#else
  return std::make_shared<T>(std::forward<Args>(args)...);
#endif
}

}  // namespace gtsam_points
