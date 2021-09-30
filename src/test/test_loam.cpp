#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/factors/integrated_icp_factor.hpp>
#include <gtsam_ext/factors/integrated_loam_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

gtsam_ext::FrameCPU::Ptr load_frame(const std::string& filename) {
  std::ifstream points_ifs(filename, std::ios::binary | std::ios::ate);
  if(!points_ifs) {
    std::cerr << "error: failed to open " << filename << std::endl;
    return nullptr;
  }

  std::streamsize points_bytes = points_ifs.tellg();
  size_t num_points = points_bytes / (sizeof(Eigen::Vector3f));

  points_ifs.seekg(0, std::ios::beg);
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;
  points.resize(num_points);
  points_ifs.read(reinterpret_cast<char*>(points.data()), sizeof(Eigen::Vector3f) * num_points);

  return gtsam_ext::FrameCPU::Ptr(new gtsam_ext::FrameCPU(points));
}

int main(int argc, char** argv) {
  auto plane1 = load_frame("data/newer_01/planes_000000.bin");
  auto plane2 = load_frame("data/newer_01/planes_000010.bin");
  auto edge1 = load_frame("data/newer_01/edges_000000.bin");
  auto edge2 = load_frame("data/newer_01/edges_000010.bin");

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("cloud", std::make_shared<glk::PointCloudBuffer>(plane1->points_storage), guik::FlatColor(1, 0, 0, 1));
  viewer->update_drawable("cloud2", std::make_shared<glk::PointCloudBuffer>(plane2->points_storage), guik::FlatColor(0, 1, 0, 1));

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;

  values.insert(0, gtsam::Pose3::identity());
  values.insert(1, gtsam::Pose3::identity());

  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::identity(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  gtsam_ext::IntegratedLOAMFactor::shared_ptr factor(new gtsam_ext::IntegratedLOAMFactor(0, 1, edge1, plane1, edge2, plane2));
  factor->set_enable_correspondence_validation(true);
  graph.add(factor);

  gtsam_ext::LevenbergMarquardtExtParams lm_params;
  lm_params.callback = [&](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
    viewer->append_text(status.to_string());
    auto drawable = viewer->find_drawable("cloud2");
    drawable.first->add("model_matrix", values.at<gtsam::Pose3>(1).matrix().cast<float>().eval());
    viewer->spin_until_click();
  };

  gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
  values = optimizer.optimize();

  viewer->spin();
  return 0;
}