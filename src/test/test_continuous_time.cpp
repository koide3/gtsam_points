#include <iostream>
#include <boost/format.hpp>

#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/factors/integrated_ct_icp_factor.hpp>
#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

struct ContinuousTimeFactorsTestBase : public testing::Test {
public:
  virtual void SetUp() {
    for (int i = 0; i < 3; i++) {
      auto times = gtsam_points::read_times((boost::format("data/newer_06/times_%02d.bin") % i).str());
      auto raw_points = gtsam_points::read_points((boost::format("data/newer_06/raw_%02d.bin") % i).str());
      auto deskewed_points = gtsam_points::read_points((boost::format("data/newer_06/deskewed_%02d.bin") % i).str());

      ASSERT_EQ(times.empty(), false) << "Failed to load point times";
      ASSERT_EQ(times.size(), raw_points.size()) << "Failed to raw points";
      ASSERT_EQ(times.size(), deskewed_points.size()) << "Failed to deskewed points";

      for (auto& pt : raw_points) {
        Eigen::Quaternionf q(0, 0, 0, 1);
        pt = q * pt;
      }

      gtsam_points::PointCloudCPU::Ptr source(new gtsam_points::PointCloudCPU(raw_points));
      source->add_times(times);
      source->add_covs(gtsam_points::estimate_covariances(source->points, source->size()));

      gtsam_points::PointCloudCPU::Ptr target(new gtsam_points::PointCloudCPU(deskewed_points));
      target->add_covs(gtsam_points::estimate_covariances(target->points, target->size()));
      target->add_normals(gtsam_points::estimate_normals(target->points, target->covs, target->size()));

      raw_source_frames.push_back(source);
      deskewed_target_frames.push_back(target);
    }
  }

  std::vector<gtsam_points::PointCloud::Ptr> raw_source_frames;
  std::vector<gtsam_points::PointCloud::Ptr> deskewed_target_frames;
};

TEST_F(ContinuousTimeFactorsTestBase, LoadCheck) {
  ASSERT_EQ(raw_source_frames.size(), 3) << "Failed to load source frames";
  ASSERT_EQ(deskewed_target_frames.size(), 3) << "Failed to load target frames";
}

struct ContinuousTimeFactorTest : public ContinuousTimeFactorsTestBase, public testing::WithParamInterface<std::tuple<std::string, std::string>> {
public:
  double pointcloud_distance(const gtsam_points::PointCloud::ConstPtr& frame1, const gtsam_points::PointCloud::ConstPtr& frame2) {
    gtsam_points::KdTree tree(frame2->points, frame2->size());

    double sum_dists = 0.0;
    for (int i = 0; i < frame1->size(); i++) {
      size_t k_index;
      double k_sq_dist;
      tree.knn_search(frame1->points[i].data(), 1, &k_index, &k_sq_dist);
      sum_dists += k_sq_dist;
    }

    double rmse = std::sqrt(sum_dists / frame1->size());
    return rmse;
  }
};

INSTANTIATE_TEST_SUITE_P(
  gtsam_points,
  ContinuousTimeFactorTest,
  testing::Combine(testing::Values("CTICP", "CTGICP"), testing::Values("NONE", "OMP", "TBB")),
  [](const auto& info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

TEST_P(ContinuousTimeFactorTest, AlignmentTest) {
  const auto param = GetParam();
  const std::string method = std::get<0>(param);
  const std::string parallelism = std::get<1>(param);
  const int num_threads = parallelism == "NONE" ? 1 : 2;

#ifndef GTSAM_POINTS_USE_TBB
  if (parallelism == "TBB") {
    std::cerr << "Skip test for TBB" << std::endl;
    return;
  }
#endif

  if (parallelism == "TBB") {
    gtsam_points::set_tbb_as_default();
  } else {
    gtsam_points::set_omp_as_default();
  }

  for (int i = 0; i < 3; i++) {
    gtsam::Values values;
    values.insert(0, gtsam::Pose3::Identity());
    values.insert(1, gtsam::Pose3::Identity());

    const auto target = deskewed_target_frames[i];
    const auto source = raw_source_frames[i];

    gtsam_points::IntegratedCT_ICPFactor::shared_ptr factor;
    if (method == "CTICP") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedCT_ICPFactor>(0, 1, target, source);
      f->set_num_threads(num_threads);
      factor = f;
    } else if (method == "CTGICP") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedCT_GICPFactor>(0, 1, target, source);
      f->set_num_threads(num_threads);
      factor = f;
    }

    gtsam::NonlinearFactorGraph graph;
    graph.add(factor);

    gtsam_points::LevenbergMarquardtExtParams lm_params;
    gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
    values = optimizer.optimize();

    auto corrected_source = factor->deskewed_source_points(values);
    gtsam_points::PointCloud::Ptr correcred(new gtsam_points::PointCloudCPU(corrected_source));

    // The corrected point cloud should have a small distance to the target
    // double dist_before = pointcloud_distance(source, target);
    double dist_after = pointcloud_distance(correcred, target);
    EXPECT_LT(dist_after, 0.1) << "Too large point cloud distance " << i;
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}