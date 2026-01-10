// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <optional>
#include <gtsam/config.h>
#include <gtsam/base/make_shared.h>
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
constexpr auto NoneValue = boost::none;

#endif

}  // namespace gtsam_points
