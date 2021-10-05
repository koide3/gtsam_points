#include <iostream>
#include <boost/format.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/expressions.hpp>
#include <gtsam_ext/util/normal_estimation.hpp>
#include <gtsam_ext/factors/continuous_time_icp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

void test(int test_id) {
  auto times = gtsam_ext::read_times((boost::format("data/newer_06/times_%02d.bin") % test_id).str());
  auto raw_points = gtsam_ext::read_points((boost::format("data/newer_06/raw_%02d.bin") % test_id).str());
  auto deskewed_points = gtsam_ext::read_points((boost::format("data/newer_06/deskewed_%02d.bin") % test_id).str());

  for (auto& pt : raw_points) {
    Eigen::Quaternionf q(0, 0, 0, 1);
    pt = q * pt;
  }

  gtsam_ext::FrameCPU::Ptr source(new gtsam_ext::FrameCPU(raw_points));
  source->add_times(times);

  gtsam_ext::FrameCPU::Ptr target(new gtsam_ext::FrameCPU(deskewed_points));
  target->add_normals(gtsam_ext::estimate_normals(target->points_storage));

  gtsam::Values values;
  values.insert(0, gtsam::Pose3::identity());
  values.insert(1, gtsam::Pose3::identity());

  auto noise_model = gtsam::noiseModel::Isotropic::Precision(1, 1.0);
  auto robust_model = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(0.1), noise_model);
  auto cticp_factor = gtsam_ext::create_integrated_cticp_factor(0, 1, target, source, robust_model);

  gtsam::NonlinearFactorGraph graph;
  graph.add(cticp_factor);

  gtsam_ext::LevenbergMarquardtExtParams lm_params;
  lm_params.callback = [](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) { std::cout << status.to_string() << std::endl; };
  gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  std::cout << "--- values ---" << std::endl;
  values.print();

  auto deskewed_source = cticp_factor->deskewed_source_points(values);
  std::cout << "deskewed:" << deskewed_source.size() << std::endl;
}

int main(int argc, char** argv) {
  for (int i = 0; i < 3; i++) {
    test(i);
  }
  return 0;
}