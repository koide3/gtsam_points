#include <chrono>
#include <iostream>
#include <filesystem>
#include <boost/format.hpp>

#include <gtsam_ext/cuda/cuda_device_sync.hpp>

#include <gtsam_ext/types/frame_gpu.hpp>
#include <gtsam_ext/types/voxelized_frame_gpu.hpp>
#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/covariance_estimation.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/normal_distributions.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  gtsam_ext::cuda_device_synchronize();
  const std::string data_path = "/home/koide/workspace/gtsam_ext/data/kitti_07_dump";

  std::ifstream ifs(data_path + "/graph.txt");
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses(5);
  for (int i = 0; i < 5; i++) {
    std::string token;
    Eigen::Vector3d trans;
    Eigen::Quaterniond quat;
    ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

    poses[i].setIdentity();
    poses[i].linear() = quat.toRotationMatrix();
    poses[i].translation() = trans;
  }

  std::vector<gtsam_ext::Frame::ConstPtr> frames;
  for (int i = 0; i < 5; i++) {
    auto points = gtsam_ext::read_points((boost::format("%s/%06d/points.bin") % data_path % i).str());

    auto frame = std::make_shared<gtsam_ext::VoxelizedFrameGPU>();
    frame->add_points(points);
    frame->add_covs(gtsam_ext::estimate_covariances(frame->points, frame->size()));

    frame->create_voxelmap(0.5);
    frames.push_back(frame);
  }

  for (int i = 0; i < 5; i++) {
    for (int j = i; j < 5; j++) {
      Eigen::Isometry3d delta = poses[i].inverse() * poses[j];
      std::cout << i << " vs " << j << " " << gtsam_ext::overlap(frames[i], frames[j], delta) << std::endl;
      std::cout << i << " vs " << j << " " << gtsam_ext::overlap_gpu(frames[i], frames[j], delta) << std::endl;
    }
  }

  auto viewer = guik::LightViewer::instance();

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    auto merged = gtsam_ext::merge_frames_gpu(poses, frames, 0.5, 0.1);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "d:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;

  // viewer->spin();

  return 0;
}