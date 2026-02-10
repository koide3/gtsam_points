/**
 * Test for voxel raycaster
 */
#include <random>
#include <unordered_set>
#include <gtest/gtest.h>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/vector3i_hash.hpp>
#include <gtsam_points/util/voxel_raycaster.hpp>

std::vector<Eigen::Vector3i> naive_ray_traversal(const Eigen::Vector4d& start, const Eigen::Vector4d& end, double voxel_size) {
  constexpr double eps = 1e-4;
  std::vector<Eigen::Vector3i> voxels;

  const double inv_voxel_size = 1.0 / voxel_size;
  const Eigen::Vector3i start_coord = gtsam_points::fast_floor(start.array() * inv_voxel_size).head<3>();
  const Eigen::Vector3i end_coord = gtsam_points::fast_floor(end.array() * inv_voxel_size).head<3>();
  if (start_coord == end_coord) {
    return voxels;
  }
  voxels.emplace_back(start_coord);

  Eigen::Vector4d pt = start;
  while ((pt - end).head<3>().norm() > eps) {
    const Eigen::Vector3d diff = end.head<3>() - pt.head<3>();
    const double step = std::min(diff.norm(), eps);
    pt.head<3>() += diff.normalized() * step;

    const Eigen::Vector3i voxel_coord = gtsam_points::fast_floor(pt.array() / voxel_size).head<3>();
    if (voxel_coord == end_coord) {
      break;
    }

    if ((voxels.back().array() != voxel_coord.array()).any()) {
      voxels.emplace_back(voxel_coord);
    }
  }

  return voxels;
}

TEST(TestVoxelRayCasting, RandomTest) {
  std::mt19937 mt;
  std::vector<double> resolutions = {0.1, 0.5, 1.0, 2.5, 5.0};

  std::uniform_real_distribution<> udist(-10.0, 10.0);
  for (double resolution : resolutions) {
    for (int i = 0; i < 25; i++) {
      const Eigen::Vector4d start(udist(mt), udist(mt), udist(mt), 1.0);
      const Eigen::Vector4d end(udist(mt), udist(mt), udist(mt), 1.0);

      gtsam_points::VoxelRaycaster ray(start, end, resolution);
      std::vector<Eigen::Vector3i> voxels_ray(ray.begin(), ray.end());
      std::vector<Eigen::Vector3i> voxels_naive = naive_ray_traversal(start, end, resolution);

      if (voxels_ray.size() > 20) {
        std::unordered_set<Eigen::Vector3i, gtsam_points::Vector3iHash> set_ray(voxels_ray.begin(), voxels_ray.end());
        std::unordered_set<Eigen::Vector3i, gtsam_points::Vector3iHash> set_naive(voxels_naive.begin(), voxels_naive.end());
        const int joint_size = std::count_if(set_ray.begin(), set_ray.end(), [&set_naive](const Eigen::Vector3i& v) { return set_naive.count(v); });
        const double error_ratio = 1.0 - static_cast<double>(joint_size) / std::max(set_ray.size(), set_naive.size());
        EXPECT_LT(error_ratio, 0.025) << " (size_ray=" << set_ray.size() << ", size_naive=" << set_naive.size() << ", joint_size=" << joint_size
                                      << ")";
      } else {
        ASSERT_EQ(voxels_ray.size(), voxels_naive.size());
        for (int j = 0; j < voxels_ray.size(); j++) {
          EXPECT_EQ((voxels_ray[j] - voxels_naive[j]).norm(), 0);
        }
      }
    }
  }
}

TEST(TestVoxelRayCasting, CornerTest) {
  std::mt19937 mt(342789);
  std::uniform_real_distribution<> udist(10.0, 10.0);

  for (int i = 0; i < 100; i++) {
    const Eigen::Vector4d pt(udist(mt), udist(mt), udist(mt), 1.0);
    gtsam_points::VoxelRaycaster ray(pt, pt, 0.25);
    std::vector<Eigen::Vector3i> voxels(ray.begin(), ray.end());
    EXPECT_EQ(voxels.size(), 0) << "pt=" << pt.transpose();
  }

  std::uniform_real_distribution<> udist2(0.25, 0.75);
  for (int i = 0; i < 100; i++) {
    const Eigen::Vector4d start(udist2(mt), udist2(mt), udist2(mt), 1.0);
    const Eigen::Vector4d end(udist2(mt), udist2(mt), udist2(mt), 1.0);
    gtsam_points::VoxelRaycaster ray(start, end, 1.0);
    std::vector<Eigen::Vector3i> voxels(ray.begin(), ray.end());
    EXPECT_EQ(voxels.size(), 0) << "start=" << start.transpose() << ", end=" << end.transpose();
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}