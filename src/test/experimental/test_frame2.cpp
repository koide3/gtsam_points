#include <chrono>
#include <iostream>
#include <filesystem>
#include <boost/format.hpp>

#include <gtsam_ext/cuda/cuda_stream.hpp>
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

  gtsam_ext::CUDAStream stream;

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> deltas;
  std::vector<gtsam_ext::Frame::ConstPtr> others;
  for (int i = 1; i < 5; i++) {
    Eigen::Isometry3d delta = poses[i].inverse() * poses[0];
    deltas.push_back(delta);
    others.push_back(frames[i]);
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10000; i++) {
    double overlap = gtsam_ext::overlap_gpu(others, frames[0], deltas, stream);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "d:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;

  std::cout << "overlap:" << gtsam_ext::overlap(others, frames[0], deltas) << std::endl;
  std::cout << "overlap_gpu:" << gtsam_ext::overlap_gpu(others, frames[0], deltas) << std::endl;

  return 0;

  auto viewer = guik::LightViewer::instance();

  for (int i = 0; i < 100; i++) {
    auto merged = gtsam_ext::merge_frames_gpu(poses, frames, 0.5, stream);
  }

  return 0;

  gtsam_ext::Frame::Ptr merged = gtsam_ext::merge_frames_gpu(poses, frames, 0.5);

  viewer->update_drawable("frame", std::make_shared<glk::NormalDistributions>(merged->points, merged->covs, merged->size()), guik::Rainbow());
  // viewer->spin();

  return 0;
}