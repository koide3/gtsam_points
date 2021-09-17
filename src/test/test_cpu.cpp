#include <fstream>
#include <iostream>
#include <boost/format.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_ext/types/voxelized_frame_gpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>
#include <gtsam_ext/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_ext/factors/integrated_icp_factor.hpp>
#include <gtsam_ext/factors/integrated_gicp_factor.hpp>
#include <gtsam_ext/factors/integrated_vgicp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/primitives/primitives.hpp>
#include <glk/thin_lines.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/normal_distributions.hpp>
#include <guik/viewer/light_viewer.hpp>

struct Submap {
public:
  using Ptr = std::shared_ptr<Submap>;

  Submap(const std::string& submap_path) {
    std::ifstream points_ifs(submap_path + "/points.bin", std::ios::binary | std::ios::ate);
    std::ifstream covs_ifs(submap_path + "/covs.bin", std::ios::binary | std::ios::ate);
    if (!points_ifs || !covs_ifs) {
      std::cerr << "error: failed to open " << submap_path + "/(points|covs).bin" << std::endl;
      abort();
    }

    std::streamsize points_bytes = points_ifs.tellg();
    std::streamsize covs_bytes = covs_ifs.tellg();
    size_t num_points = points_bytes / (sizeof(Eigen::Vector3f));
    size_t num_covs = covs_bytes / (sizeof(Eigen::Matrix3f));
    if (num_points != num_covs) {
      std::cerr << "error: mismatch of num of points/covs is found!!" << std::endl;
      abort();
    }

    points_ifs.seekg(0, std::ios::beg);
    covs_ifs.seekg(0, std::ios::beg);

    points.resize(num_points);
    covs.resize(num_points);
    points_ifs.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector3f) * num_points);
    covs_ifs.read(reinterpret_cast<char*>(covs.data()), sizeof(Eigen::Matrix3f) * num_points);
  }

public:
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> covs;
};

struct GlobalMap {
public:
  GlobalMap(const std::string& dump_path) {
    std::ifstream ifs(dump_path + "/graph.txt");
    if (!ifs) {
      std::cerr << "error: failed to open " << dump_path << "/graph.txt" << std::endl;
      abort();
    }

    std::string token;
    int num_frames, num_all_frames, num_factors;

    ifs >> token >> num_frames;
    ifs >> token >> num_all_frames;
    ifs >> token >> num_factors;

    for (int i = 0; i < num_frames; i++) {
      Eigen::Vector3d trans;
      Eigen::Quaterniond quat;
      ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.linear() = quat.toRotationMatrix();
      pose.translation() = trans;

      submap_poses.push_back(pose);
      submaps.push_back(std::make_shared<Submap>((boost::format("%s/%06d") % dump_path % i).str()));
    }

    for (int i = 0; i < num_factors; i++) {
      int first, second;
      ifs >> token >> first >> second;
      factors.push_back(std::make_pair(first, second));
    }
  }

public:
  std::vector<Submap::Ptr> submaps;
  std::vector<std::pair<int, int>> factors;
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> submap_poses;
};

int main(int argc, char** argv) {
  GlobalMap globalmap("/home/koide/datasets/gvlo_kitti_07");

  auto viewer = guik::LightViewer::instance();

  std::vector<gtsam_ext::VoxelizedFrameGPU::Ptr> frames;
  for (int i = 0; i < globalmap.submaps.size(); i++) {
    const auto& submap = globalmap.submaps[i];
    const auto& submap_pose = globalmap.submap_poses[i];

    gtsam_ext::VoxelizedFrameGPU::Ptr frame(new gtsam_ext::VoxelizedFrameGPU(2.0, submap->points, submap->covs));
    frames.push_back(frame);

    viewer->update_drawable("submap_" + std::to_string(i), std::make_shared<glk::PointCloudBuffer>(submap->points), guik::Rainbow(submap_pose.cast<float>()));
  }

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3::identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

  for (int i = 0; i < globalmap.submaps.size(); i++) {
    gtsam::Pose3 noise = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * 0.1);
    values.insert(i, gtsam::Pose3(globalmap.submap_poses[i].matrix()) * noise);

    int target = (i + 1) % globalmap.submaps.size();
    gtsam_ext::IntegratedICPFactor::shared_ptr factor(new gtsam_ext::IntegratedICPFactor(target, i, frames[target], frames[i]));
    factor->set_num_threads(16);
    graph.add(factor);
  }

  gtsam_ext::LevenbergMarquardtExtParams lm_params;
  lm_params.setDiagonalDamping(true);
  lm_params.callback = [&](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    viewer->append_text(status.to_string());
    for (int i = 0; i < globalmap.submaps.size(); i++) {
      auto drawable = viewer->find_drawable("submap_" + std::to_string(i));
      drawable.first->add("model_matrix", values.at<gtsam::Pose3>(i).matrix().cast<float>().eval());
    }
    viewer->spin_once();
  };

  gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  viewer->spin();

  return 0;
}