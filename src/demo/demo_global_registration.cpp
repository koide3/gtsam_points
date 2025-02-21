#include <iostream>
#include <guik/viewer/light_viewer.hpp>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/ann/kdtreex.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/fpfh_estimation.hpp>
#include <gtsam_points/registration/ransac.hpp>
#include <gtsam_points/registration/graduated_non_convexity.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

class GlobalRegistrationDemo {
public:
  GlobalRegistrationDemo() {
    const std::string data_path = "./data";
    const auto target_raw = gtsam_points::read_points(data_path + "/kitti_00/000000.bin");
    const auto source_raw = gtsam_points::read_points(data_path + "/kitti_00/000001.bin");
    if (target_raw.empty()) {
      std::cerr << "Failed to read target points" << std::endl;
      abort();
    }
    if (source_raw.empty()) {
      std::cerr << "Failed to read source points" << std::endl;
      abort();
    }

    target = std::make_shared<gtsam_points::PointCloudCPU>(target_raw);
    target = gtsam_points::voxelgrid_sampling(target, 0.5);
    target->add_normals(gtsam_points::estimate_normals(target->points, target->size(), 10));
    target_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(target);

    source = std::make_shared<gtsam_points::PointCloudCPU>(source_raw);
    source = gtsam_points::voxelgrid_sampling(source, 0.5);
    source->add_normals(gtsam_points::estimate_normals(source->points, source->size(), 10));
    source_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(source);

    num_threads = 4;
    method = 0;
    fpfh_radius = 5.0;
    enable_4dof = false;

    auto viewer = guik::viewer();

    viewer->update_points("target", target->points, target->size(), guik::FlatBlue());
    viewer->update_points("source", source->points, source->size(), guik::FlatRed());

    viewer->register_ui_callback("ui_callback", [this] {
      auto viewer = guik::viewer();

      ImGui::Combo("method", &method, "GNC\0RANSAC\0");
      ImGui::DragFloat("fpfh_radius", &fpfh_radius, 0.1, 0.1, 10.0);
      ImGui::Checkbox("4DoF", &enable_4dof);
      ImGui::DragInt("num_threads", &num_threads, 1, 1, 16);

      if (ImGui::Button("Add noise to source pose")) {
        viewer->append_text("Adding noise to source pose...");

        Eigen::Isometry3d noise = Eigen::Isometry3d::Identity();
        noise.translation() = Eigen::Vector3d::Random() * 5.0;

        Eigen::Vector4d rnoise = Eigen::Vector4d::Random();
        if (enable_4dof) {
          rnoise(0) = 0.0;
          rnoise(1) = 0.0;
        }
        noise.linear() = Eigen::Quaterniond(rnoise.normalized()).toRotationMatrix();

        update_source(noise);
      }

      if (ImGui::Button("Align!!")) {
        viewer->append_text("*********");

        gtsam_points::FPFHEstimationParams fpfh_params;
        fpfh_params.search_radius = fpfh_radius;
        fpfh_params.num_threads = num_threads;

        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto target_fpfh = gtsam_points::estimate_fpfh(*target, *target_tree, fpfh_params);
        const auto source_fpfh = gtsam_points::estimate_fpfh(*source, *source_tree, fpfh_params);

        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto target_fpfh_tree = std::make_shared<gtsam_points::KdTreeX<33>>(target_fpfh.data(), target_fpfh.size());
        const auto source_fpfh_tree = std::make_shared<gtsam_points::KdTreeX<33>>(source_fpfh.data(), source_fpfh.size());

        const auto t2 = std::chrono::high_resolution_clock::now();
        gtsam_points::RegistrationResult result;
        if (method == 0) {
          gtsam_points::RANSACParams ransac_params;
          ransac_params.dof = enable_4dof ? 4 : 6;
          ransac_params.num_threads = num_threads;
          ransac_params.seed = mt();

          result = gtsam_points::estimate_pose_ransac(
            *target,
            *source,
            target_fpfh.data(),
            source_fpfh.data(),
            *target_tree,
            *target_fpfh_tree,
            ransac_params);
        } else {
          gtsam_points::GNCParams gnc_params;
          gnc_params.dof = enable_4dof ? 4 : 6;
          gnc_params.num_threads = num_threads;
          gnc_params.seed = mt();

          result = gtsam_points::estimate_pose_gnc(
            *target,
            *source,
            target_fpfh.data(),
            source_fpfh.data(),
            *target_tree,
            *target_fpfh_tree,
            *source_fpfh_tree,
            gnc_params);
        }
        const auto t3 = std::chrono::high_resolution_clock::now();

        const double feature_time = std::chrono::duration<double>(t1 - t0).count();
        const double kdtree_time = std::chrono::duration<double>(t2 - t1).count();
        const double registration_time = std::chrono::duration<double>(t3 - t2).count();

        viewer->append_text("- Feature extraction=" + std::to_string(feature_time) + "s");
        viewer->append_text("- KdTree creation=" + std::to_string(kdtree_time) + "s");
        viewer->append_text("- Registration=" + std::to_string(registration_time) + "s");

        update_source(result.T_target_source);
      }
    });
  }

  void update_source(const Eigen::Isometry3d& T_target_source) {
    gtsam_points::transform_inplace(source, T_target_source);
    source_tree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(source);

    guik::viewer()->update_points("source", source->points, source->size(), guik::FlatRed());
  }

  void run() { guik::viewer()->spin(); }

private:
  std::mt19937 mt;

  int num_threads;
  int method;
  float fpfh_radius;
  bool enable_4dof;

  gtsam_points::PointCloudCPU::Ptr target;
  gtsam_points::PointCloudCPU::Ptr source;

  gtsam_points::NearestNeighborSearch::Ptr target_tree;
  gtsam_points::NearestNeighborSearch::Ptr source_tree;
};

int main(int argc, char** argv) {
  GlobalRegistrationDemo demo;
  demo.run();

  return 0;
}