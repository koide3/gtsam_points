#include <gtsam/geometry/Pose3.h>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/util/covariance_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>

#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  const std::string target_path = "/home/koide/workspace/gtsam_points/data/kitti_00/000000.bin";
  const std::string source_path = "/home/koide/workspace/gtsam_points/data/kitti_00/000001.bin";

  const auto read = [](const std::string& filename) {
    const auto points_raw = gtsam_points::read_points(filename);

    auto points = std::make_shared<gtsam_points::PointCloudCPU>(points_raw);
    points = gtsam_points::voxelgrid_sampling(points, 0.5);
    points->add_covs(gtsam_points::estimate_covariances(*points));

    return gtsam_points::PointCloudGPU::clone(*points);
  };

  const auto target = read(target_path);
  const auto source = read(source_path);

  auto target_voxels = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(0.5);
  target_voxels->insert(*target);

  std::cout << "target=" << target->size() << " source=" << source->size() << std::endl;

  gtsam::Values values;
  values.insert(0, gtsam::Pose3());
  values.insert(1, gtsam::Pose3());

  auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(0, 1, target_voxels, source);

  for (int i = 0; i < 8192; i++) {
    auto linearized = factor->linearize(values);
    // linearized->print("linearized");

    auto linearized2 = factor->linearize(values);
    // linearized2->print("linearized2");

    gtsam::Matrix e = linearized->augmentedInformation() - linearized2->augmentedInformation();

    if (e.array().abs().maxCoeff() > 1e-6) {
      std::cout << "error: " << e.array().abs().maxCoeff() << std::endl;
      std::cout << "diff=" << std::endl << e << std::endl;
    }
  }

  auto viewer = guik::viewer();
  viewer->update_points("target", target->points, target->size(), guik::FlatBlue());
  viewer->update_points("source", source->points, source->size(), guik::FlatRed());
  viewer->spin();

  return 0;
}