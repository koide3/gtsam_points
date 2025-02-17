#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>
#include <boost/filesystem.hpp>

#include <gtest/gtest.h>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/parallelism.hpp>

class KdTreeTest : public testing::Test, public testing::WithParamInterface<std::string> {
  virtual void SetUp() {
    const int num_points = 1000;
    const int num_queries = 100;

    std::mt19937 mt;
    std::uniform_real_distribution<> udist(-1.0, 1.0);

    points.resize(num_points);
    points_x.resize(num_points);
    for (int i = 0; i < num_points; i++) {
      points[i] << 100.0 * udist(mt), 100.0 * udist(mt), 100.0 * udist(mt), 1.0;
      points_x[i] = points[i];
    }

    queries.resize(num_queries);
    for (int i = 0; i < num_queries / 2; i++) {
      queries[i] << 100.0 * udist(mt), 100.0 * udist(mt), 100.0 * udist(mt), 1.0;
    }
    for (int i = num_queries / 2; i < num_queries; i++) {
      queries[i] << 200.0 * udist(mt), 200.0 * udist(mt), 200.0 * udist(mt), 1.0;
    }

    constexpr int max_k = 20;
    gt_indices.resize(num_queries);
    gt_sq_dists.resize(num_queries);

    const double max_sq_radius = 100.0 * 100.0;
    gt_radius_indices.resize(num_queries);
    gt_radius_sq_dists.resize(num_queries);

    for (int i = 0; i < num_queries; i++) {
      std::vector<std::pair<size_t, double>> dists(num_points);
      for (int j = 0; j < num_points; j++) {
        dists[j] = {j, (points[j] - queries[i]).squaredNorm()};
      }

      std::sort(dists.begin(), dists.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

      gt_indices[i].resize(max_k);
      gt_sq_dists[i].resize(max_k);
      for (int j = 0; j < max_k; j++) {
        gt_indices[i][j] = dists[j].first;
        gt_sq_dists[i][j] = dists[j].second;
      }

      for (int j = 0; j < num_points && dists[j].second < max_sq_radius; j++) {
        gt_radius_indices[i].emplace_back(dists[j].first);
        gt_radius_sq_dists[i].emplace_back(dists[j].second);
      }
    }
  }

public:
  std::vector<Eigen::Vector4d> points;
  std::vector<Eigen::VectorXd> points_x;
  std::vector<Eigen::Vector4d> queries;
  std::vector<std::vector<size_t>> gt_indices;
  std::vector<std::vector<double>> gt_sq_dists;

  std::vector<std::vector<size_t>> gt_radius_indices;
  std::vector<std::vector<double>> gt_radius_sq_dists;
};

TEST_F(KdTreeTest, LoadCheck) {
  ASSERT_NE(points.size(), 0);
  ASSERT_NE(queries.size(), 0);
  ASSERT_EQ(gt_indices.size(), queries.size());
  ASSERT_EQ(gt_sq_dists.size(), queries.size());
}

INSTANTIATE_TEST_SUITE_P(
  gtsam_points,
  KdTreeTest,
  testing::Values("KdTree", "KdTreeMT", "KdTreeTBB", "KdTree2", "KdTree2MT", "KdTree2TBB", "KdTreeX", "KdTreeX4"),
  [](const auto& info) { return info.param; });

TEST_P(KdTreeTest, KnnTest) {
  gtsam_points::NearestNeighborSearch::ConstPtr kdtree;

  if (GetParam().find("TBB") != std::string::npos) {
    gtsam_points::set_tbb_as_default();
  } else {
    gtsam_points::set_omp_as_default();
  }

  if (GetParam() == "KdTree") {
    kdtree = std::make_shared<gtsam_points::KdTree>(points.data(), points.size());
  } else if (GetParam() == "KdTreeMT") {
    kdtree = std::make_shared<gtsam_points::KdTree>(points.data(), points.size(), 2);
  } else if (GetParam() == "KdTreeTBB") {
    kdtree = std::make_shared<gtsam_points::KdTree>(points.data(), points.size(), 2);
  } else if (GetParam() == "KdTree2") {
    auto pts = std::make_shared<gtsam_points::PointCloudCPU>(points);
    kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(pts);
  } else if (GetParam() == "KdTree2MT") {
    auto pts = std::make_shared<gtsam_points::PointCloudCPU>(points);
    kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(pts, 2);
  } else if (GetParam() == "KdTree2TBB") {
    auto pts = std::make_shared<gtsam_points::PointCloudCPU>(points);
    kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(pts, 2);
  } else if (GetParam() == "KdTreeX") {
    kdtree = std::make_shared<gtsam_points::KdTreeX<-1>>(points_x.data(), points_x.size());
  } else if (GetParam() == "KdTreeX4") {
    kdtree = std::make_shared<gtsam_points::KdTreeX<4>>(points.data(), points.size());
  } else {
    FAIL() << "Unknown KdTree type: " << GetParam();
  }

  const std::vector<int> ks = {1, 2, 3, 5, 10, 15, 20};
  const double max_sq_dist = 10.0;

  for (int k : ks) {
    for (int i = 0; i < queries.size(); i++) {
      const auto& query = queries[i];

      std::vector<size_t> k_indices(k);
      std::vector<double> k_sq_dists(k);

      const auto num_found = kdtree->knn_search(query.data(), k, k_indices.data(), k_sq_dists.data());
      EXPECT_EQ(num_found, k);

      for (int j = 0; j < k; j++) {
        const double sq_dist = (points[k_indices[j]] - query).squaredNorm();
        EXPECT_NEAR(k_sq_dists[j], sq_dist, 1.0e-6);
        EXPECT_NEAR(k_sq_dists[j], gt_sq_dists[i][j], 1.0e-6);
      }
    }

    for (int i = 0; i < queries.size(); i++) {
      const auto& query = queries[i];

      std::vector<size_t> k_indices(k);
      std::vector<double> k_sq_dists(k);

      const auto num_found = kdtree->knn_search(query.data(), k, k_indices.data(), k_sq_dists.data(), max_sq_dist);
      EXPECT_LE(num_found, k);

      for (int j = 0; j < num_found; j++) {
        const double sq_dist = (points[k_indices[j]] - query).squaredNorm();
        EXPECT_NEAR(k_sq_dists[j], sq_dist, 1.0e-6);
        EXPECT_NEAR(k_sq_dists[j], gt_sq_dists[i][j], 1.0e-6);
      }

      for (int j = num_found; j < k; j++) {
        EXPECT_GT(gt_sq_dists[i][j], max_sq_dist);
      }
    }
  }
}

TEST_P(KdTreeTest, RadiusTest) {
  gtsam_points::NearestNeighborSearch::ConstPtr kdtree;

  if (GetParam().find("TBB") != std::string::npos) {
    gtsam_points::set_tbb_as_default();
  } else {
    gtsam_points::set_omp_as_default();
  }

  if (GetParam() == "KdTree") {
    kdtree = std::make_shared<gtsam_points::KdTree>(points.data(), points.size());
  } else if (GetParam() == "KdTreeMT") {
    kdtree = std::make_shared<gtsam_points::KdTree>(points.data(), points.size(), 2);
  } else if (GetParam() == "KdTreeTBB") {
    kdtree = std::make_shared<gtsam_points::KdTree>(points.data(), points.size(), 2);
  } else if (GetParam() == "KdTree2") {
    auto pts = std::make_shared<gtsam_points::PointCloudCPU>(points);
    kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(pts);
  } else if (GetParam() == "KdTree2MT") {
    auto pts = std::make_shared<gtsam_points::PointCloudCPU>(points);
    kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(pts, 2);
  } else if (GetParam() == "KdTree2TBB") {
    auto pts = std::make_shared<gtsam_points::PointCloudCPU>(points);
    kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(pts, 2);
  } else if (GetParam() == "KdTreeX") {
    kdtree = std::make_shared<gtsam_points::KdTreeX<-1>>(points_x.data(), points_x.size());
  } else if (GetParam() == "KdTreeX4") {
    kdtree = std::make_shared<gtsam_points::KdTreeX<4>>(points.data(), points.size());
  } else {
    FAIL() << "Unknown KdTree type: " << GetParam();
  }

  const std::vector<double> radii = {0.0, 1.0, 10.0, 50.0, 100.0};
  const std::vector<int> max_nn = {0, 1, 5, 10, 100};

  for (double radius : radii) {
    for (int i = 0; i < queries.size(); i++) {
      const auto& query = queries[i];

      std::vector<size_t> indices;
      std::vector<double> sq_dists;
      const auto num_found = kdtree->radius_search(query.data(), radius, indices, sq_dists);

      ASSERT_EQ(num_found, indices.size());
      ASSERT_EQ(num_found, sq_dists.size());

      std::vector<double> gt_sq_dists;
      std::copy_if(gt_radius_sq_dists[i].begin(), gt_radius_sq_dists[i].end(), std::back_inserter(gt_sq_dists), [&](double d) {
        return d < radius * radius;
      });

      EXPECT_EQ(num_found, gt_sq_dists.size());
      for (int j = 0; j < num_found; j++) {
        EXPECT_NEAR(gt_sq_dists[j], sq_dists[j], 1e-3);
        EXPECT_NEAR(gt_sq_dists[j], (points[indices[j]] - query).squaredNorm(), 1e-3);
      }

      std::set<size_t> full_indices_set(indices.begin(), indices.end());

      for (int nn : max_nn) {
        std::vector<size_t> indices;
        std::vector<double> sq_dists;
        const auto num_found = kdtree->radius_search(query.data(), radius, indices, sq_dists, nn);

        // EXPECT_LE(num_found, nn);
        ASSERT_EQ(num_found, indices.size());
        ASSERT_EQ(num_found, sq_dists.size());

        for (size_t index : indices) {
          EXPECT_TRUE(full_indices_set.count(index));
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}