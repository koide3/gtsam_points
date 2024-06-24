// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <string>
#include <iostream>

namespace gtsam_points {

class CurandCheckError {
public:
  void operator<<(int error) const;

  template <typename T>
  void operator<<(T error) const {
    (*this) << static_cast<int>(error);
  }
};

extern CurandCheckError check_curand;

}  // namespace gtsam_points