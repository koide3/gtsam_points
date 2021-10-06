#include <iostream>
#include <boost/format.hpp>

#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_ext/types/kdtree.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/normal_estimation.hpp>
#include <gtsam_ext/factors/continuous_time_icp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

double pointcloud_distance(const gtsam_ext::Frame::ConstPtr& frame1, const gtsam_ext::Frame::ConstPtr& frame2) {
  gtsam_ext::KdTree tree(frame2->size(), frame2->points);

  double sum_dists = 0.0;
  for (int i = 0; i<frame1->size(); i++) {
    size_t k_index;
    double k_sq_dist;
    tree.knnSearch(frame1->points[i].data(), 1, &k_index, &k_sq_dist);
    sum_dists += k_sq_dist;
  }

  double rmse = std::sqrt(sum_dists / frame1->size());
  return rmse;
}

void test(int test_id) {
  auto times = gtsam_ext::read_times((boost::format("data/newer_06/times_%02d.bin") % test_id).str());
  auto raw_points = gtsam_ext::read_points((boost::format("data/newer_06/raw_%02d.bin") % test_id).str());
  auto deskewed_points = gtsam_ext::read_points((boost::format("data/newer_06/deskewed_%02d.bin") % test_id).str());

  ASSERT_EQ(times.empty(), false)                 << "Failed to load point times";
  ASSERT_EQ(times.size(), raw_points.size())      << "Failed to raw points";
  ASSERT_EQ(times.size(), deskewed_points.size()) << "Failed to deskewed points";

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
  gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  auto corrected_source = cticp_factor->deskewed_source_points(values);
  gtsam_ext::Frame::Ptr correcred(new gtsam_ext::FrameCPU(corrected_source));

  // The corrected point cloud should have a small distance to the target
  // double dist_before = pointcloud_distance(source, target);
  double dist_after = pointcloud_distance(correcred, target);
  EXPECT_LT(dist_after, 0.1) << "Too large point cloud distance " << test_id;
}

TEST(CTICP_Test, AlignmentTest) {
  for (int i = 0; i < 3; i++) {
    test(i);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}