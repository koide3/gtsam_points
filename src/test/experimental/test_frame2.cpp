#include <chrono>
#include <iostream>
#include <filesystem>
#include <boost/format.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/voxelized_frame_gpu.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/cuda/cuda_device_sync.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu.hpp>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/normal_distributions.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  gtsam_points::cuda_device_synchronize();
  const std::string data_path = "/home/koide/workspace/gtsam_points/data/kitti_07_dump";

  std::ifstream ifs(data_path + "/graph.txt");
  std::vector<Eigen::Isometry3d> poses(5);
  for (int i = 0; i < 5; i++) {
    std::string token;
    Eigen::Vector3d trans;
    Eigen::Quaterniond quat;
    ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

    poses[i].setIdentity();
    poses[i].linear() = quat.toRotationMatrix();
    poses[i].translation() = trans;
  }

  std::vector<gtsam_points::PointCloud::ConstPtr> frames(5);
  std::vector<std::vector<gtsam_points::GaussianVoxelMapGPU::ConstPtr>> voxelmaps(5);

  double voxel_resolution = 0.1;
  int voxelmap_levels = 8;

  for (int i = 0; i < 5; i++) {
    auto points = gtsam_points::read_points((boost::format("%s/%06d/points.bin") % data_path % i).str());

    auto frame = std::make_shared<gtsam_points::PointCloudGPU>();
    frame->add_points(points);
    frame->add_covs(gtsam_points::estimate_covariances(frame->points, frame->size()));
    frames[i] = frame;

    for (int j = 0; j < voxelmap_levels; j++) {
      double resolution = voxel_resolution * std::pow(2.0, j);
      auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(resolution);
      voxelmap->insert(*frame);
      voxelmaps[i].push_back(voxelmap);
    }
  }

  gtsam_points::StreamTempBufferRoundRobin stream_buffer_roundrobin(16);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  for (int i = 0; i < 5; i++) {
    gtsam::Pose3 noise = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * 0.1);
    values.insert(i, gtsam::Pose3(poses[i].matrix()) * noise);
  }

  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, values.at<gtsam::Pose3>(0), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  for (int i = 0; i < 5; i++) {
    for (int j = 1; j < 5; j++) {
      for (auto& voxelmap : voxelmaps[i]) {
        auto stream_buffer = stream_buffer_roundrobin.get_stream_buffer();
        auto& stream = stream_buffer.first;
        auto& buffer = stream_buffer.second;
        graph.emplace_shared<gtsam_points::IntegratedVGICPFactorGPU>(i, j, voxelmap, frames[j], stream, buffer);
      }
    }
  }

  auto viewer = guik::LightViewer::instance();

  auto visualize = [&](const gtsam::Values& values) {
    for (int i = 0; i < 5; i++) {
      viewer->update_drawable(
        "frame_" + std::to_string(i),
        std::make_shared<glk::PointCloudBuffer>(frames[i]->points, frames[i]->size()),
        guik::Rainbow(values.at<gtsam::Pose3>(i).matrix().cast<float>()));
    }
  };

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  lm_params.setDiagonalDamping(false);
  lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    return;
    std::cout << status.to_string() << std::endl;
    visualize(values);
    viewer->spin_once();
  };

  auto t1 = std::chrono::high_resolution_clock::now();
  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();
  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "d:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;

  // viewer->spin();

  return 0;
}
