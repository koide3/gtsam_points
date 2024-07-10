// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <sstream>

#include <boost/utility/in_place_factory.hpp>
#include <boost/utility/typed_in_place_factory.hpp>
#include <boost/format.hpp>

#include <gtsam/nonlinear/ISAM2Result.h>

namespace gtsam_points {
struct ISAM2ResultExt : public gtsam::ISAM2Result {
public:
  ISAM2ResultExt(bool enableDetailedResults = false)
  : gtsam::ISAM2Result(enableDetailedResults),
    delta(0.0),
    update_count(0),
    gpu_evaluation_count(0),
    gpu_linearization_count(0),
    num_factors(0),
    num_values(0),
    elapsed_time(0) {
    variablesRelinearized = 0;
    variablesReeliminated = 0;
  }

  std::string to_string() const {
    std::stringstream sst1;
    std::stringstream sst2;

    char check[] = {' ', 'x'};

    if (errorBefore && errorAfter) {
      bool dec = errorBefore.get() > errorAfter.get();
      sst1 << boost::format("%5s %15s %15s ") % "dec" % "e0" % "ei";
      sst2 << boost::format("%5c %15g %15g ") % check[dec] % errorBefore.get() % errorAfter.get();
    }

    sst1 << boost::format("%5s %5s %5s %5s %5s %5s %10s %10s") % "count" % "new" % "lin" % "elim" % "gpu_e" % "gpu_l" % "delta" % "time_msec";
    sst2 << boost::format("%5d %5d %5d %5d %5d %5d %10g %10g") % update_count % newFactorsIndices.size() % variablesRelinearized % variablesReeliminated % gpu_evaluation_count %
              gpu_linearization_count % delta % (elapsed_time * 1000.0);

    sst1 << std::endl << sst2.str();

    return sst1.str();
  }

public:
  double delta;
  int update_count;

  int gpu_evaluation_count;
  int gpu_linearization_count;

  int num_factors;
  int num_values;

  double elapsed_time;
};
}  // namespace gtsam_points
