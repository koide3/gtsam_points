#include <glk/io/ply_io.hpp>
#include <glk/indexed_pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/segmentation/min_cut.hpp>
#include <gtsam_points/segmentation/region_growing.hpp>

gtsam_points::PointCloud::Ptr read_points() {
  const std::string dataset_path = "/home/koide/workspace/gtsam_points/data/kitti_07_dump";

  std::ifstream ifs(dataset_path + "/graph.txt");
  if (!ifs) {
    std::cerr << "Failed to open " << dataset_path + "/graph.txt" << std::endl;
    return nullptr;
  }

  std::vector<Eigen::Vector4d> all_points;
  for (int i = 0; i < 5; i++) {
    std::string token;
    Eigen::Vector3d trans;
    Eigen::Quaterniond quat;
    ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = trans;
    pose.linear() = quat.normalized().toRotationMatrix();

    const auto points_raw = gtsam_points::read_points(dataset_path + "/00000" + std::to_string(i) + "/points.bin");
    for (const auto& pt : points_raw) {
      all_points.emplace_back(pose * pt.homogeneous().cast<double>());
    }
  }

  auto points = std::make_shared<gtsam_points::PointCloudCPU>(all_points);
  points->add_normals(gtsam_points::estimate_normals(*points, 10));
  return points;
}

int main(int argc, char** argv) {
  auto points = read_points();
  if (!points) {
    return 1;
  }

  auto kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(points);

  auto viewer = guik::viewer();
  viewer->update_points("points", points->points, points->size(), guik::Rainbow());

  Eigen::Vector4d picked_pt(0.0, 0.0, 0.0, 1.0);

  int segmentation_method = 0;

  gtsam_points::MinCutParams mc_params;
  gtsam_points::RegionGrowingParams rg_params;

  viewer->register_ui_callback("ui_callback", [&] {
    auto picked = viewer->pick_point(1, 5);
    if (picked) {
      picked_pt << picked->x(), picked->y(), picked->z(), 1.0;
    }

    ImGui::Combo("Method", &segmentation_method, "Min-Cut\0Region Growing\0");

    if (segmentation_method == 0) {
      viewer->update_wire_sphere("fg_radius", guik::FlatRed().translate(picked_pt).scale(mc_params.foreground_mask_radius).set_alpha(0.5));
      viewer->update_wire_sphere("bg_radius", guik::FlatBlue().translate(picked_pt).scale(mc_params.background_mask_radius).set_alpha(0.5));
    } else {
      viewer->remove_drawable("fg_radius");
      viewer->remove_drawable("bg_radius");
    }

    if (segmentation_method == 0) {
      float distance_sigma = mc_params.distance_sigma;
      float angle_sigma = mc_params.angle_sigma * 180.0 / M_PI;
      float foreground_radius = mc_params.foreground_mask_radius;
      float background_radius = mc_params.background_mask_radius;

      ImGui::DragFloat("Distance Sigma", &distance_sigma, 0.01f, 0.0f, 1.0f);
      ImGui::DragFloat("Angle Sigma", &angle_sigma, 0.1f, 0.0f, 180.0f);
      ImGui::DragFloat("Foreground Radius", &foreground_radius, 0.01f, 0.0f, 1000.0f);
      ImGui::DragFloat("Background Radius", &background_radius, 0.01f, 0.0f, 1000.0f);
      ImGui::DragInt("k-neighbors", &mc_params.k_neighbors, 1, 1, 100);
      ImGui::DragInt("Num Threads", &mc_params.num_threads, 1, 1, 100);

      mc_params.distance_sigma = distance_sigma;
      mc_params.angle_sigma = angle_sigma * M_PI / 180.0;
      mc_params.foreground_mask_radius = foreground_radius;
      mc_params.background_mask_radius = background_radius;
    } else {
      float distance_threshold = rg_params.distance_threshold;
      float angle_threshold = rg_params.angle_threshold * 180.0 / M_PI;
      float dilation_radius = rg_params.dilation_radius;

      ImGui::DragFloat("Distance Threshold", &distance_threshold, 0.01f, 0.0f, 1.0f);
      ImGui::DragFloat("Angle Threshold", &angle_threshold, 0.1f, 0.0f, 180.0f);
      ImGui::DragFloat("Dilation Radius", &dilation_radius, 0.01f, 0.0f, 1000.0f);
      ImGui::DragInt("Num Threads", &rg_params.num_threads, 1, 1, 100);
    }

    if (ImGui::Button("Segmentation")) {
      std::vector<size_t> indices;
      if (segmentation_method == 0) {
        std::vector<int> filtered_indices;
        for (size_t i = 0; i < points->size(); i++) {
          if ((points->points[i] - picked_pt).norm() < mc_params.background_mask_radius + 1.0) {
            filtered_indices.push_back(i);
          }
        }

        auto filtered = gtsam_points::sample(points, filtered_indices);
        auto filtered_kdtree = std::make_shared<gtsam_points::KdTree2<gtsam_points::PointCloud>>(filtered);

        auto result = gtsam_points::min_cut(*filtered, *filtered_kdtree, picked_pt, mc_params);

        indices.resize(result.cluster_indices.size());
        std::transform(result.cluster_indices.begin(), result.cluster_indices.end(), indices.begin(), [&](size_t i) { return filtered_indices[i]; });
      } else {
        auto rg = gtsam_points::region_growing_init(*points, *kdtree, picked_pt, rg_params);
        gtsam_points::region_growing_update(rg, *points, *kdtree, rg_params);
        indices = rg.cluster_indices;
      }

      auto cluster = gtsam_points::sample(points, std::vector<int>(indices.begin(), indices.end()));
      viewer->update_points("cluster", cluster->points, cluster->size(), guik::FlatOrange().set_point_scale(3.0f));
    }

    viewer->update_sphere("picked", guik::FlatRed().translate(picked_pt).scale(0.1));
  });

  viewer->spin();
}