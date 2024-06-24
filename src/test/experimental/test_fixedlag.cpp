#include <sstream>
#include <fstream>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_ext.hpp>

int main(int argc, char** argv) {
  std::ifstream ifs("/home/koide/isam2_update2.txt");

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::FixedLagSmootherKeyTimestampMap new_stamps;
  gtsam::FixedLagSmootherKeyTimestampMap all_stamps;


  const double smoother_lag = 5.0;

  gtsam::ISAM2Params isam2_params;
  isam2_params.relinearizeSkip = 1;
  // gtsam::IncrementalFixedLagSmoother smoother(smoother_lag, isam2_params);
  gtsam_points::IncrementalFixedLagSmootherExt smoother(smoother_lag, isam2_params);

  std::string line;
  while(!ifs.eof() && std::getline(ifs, line) && !line.empty() ) {
    std::stringstream sst(line);

    std::string token;
    sst >> token;

    if(token == "factor") {
      int num_keys;
      sst >> num_keys;

      std::vector<gtsam::Key> keys(num_keys);
      for(int i=0; i<num_keys; i++) {
        sst >> keys[i];
      }

      auto noise_model = gtsam::noiseModel::Isotropic::Precision(6, 1.0);
      if(num_keys == 1) {
        new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(keys[0], gtsam::Pose3(), noise_model);
      } else if (num_keys == 2) {
        new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(keys[0], keys[1], gtsam::Pose3(), noise_model);
      }
    } else if(token == "theta") {
      gtsam::Key key;
      sst >> key;

      new_values.insert(key, gtsam::Pose3());
    } else if (token == "time") {
      gtsam::Key key;
      double stamp;
      sst >> key >> stamp;

      new_stamps[key] = stamp;
    } else if (token == "update") {
      std::cout << "--- update ---" << std::endl;
      all_stamps.insert(new_stamps.begin(), new_stamps.end());

      if(!new_stamps.empty()) {
        const double now = new_stamps.begin()->second;
        std::cout << boost::format("current:%.6f  horizon:%.6f") % now % (now - smoother_lag) << std::endl;
      }

      std::cout << "* new factors" << std::endl;
      for(const auto& factor: new_factors) {
        for(const auto key: factor->keys()) {
          std::cout << "(" << gtsam::Symbol(key) << boost::format(", %.6f)") % all_stamps[key] << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "* new values" << std::endl;
      for(const auto& value: new_values) {
        std::cout << gtsam::Symbol(value.key) << std::endl;
      }

      std::cout << "* new stamps" << std::endl;
      for(const auto& stamp: new_stamps) {
        std::cout << gtsam::Symbol(stamp.first) << " " << boost::format("%.6f") % stamp.second << std::endl;
      }

      smoother.update(new_factors, new_values, new_stamps);

      new_values.clear();
      new_factors.resize(0);
      new_stamps.clear();
    }
  }

  return 0;
}