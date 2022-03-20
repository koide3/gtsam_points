#include <random>
#include <iostream>

#include <gtsam_ext/ann/ivox.hpp>
#include <gtsam_ext/util/read_points.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  const auto points_f = gtsam_ext::read_points("data/kitti_07_dump/000000/points.bin");
  std::vector<Eigen::Vector4d> points(points_f.size());
  std::transform(points_f.begin(), points_f.end(), points.begin(), [](const Eigen::Vector3f& p) {
    return Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0);
  });

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("points", std::make_shared<glk::PointCloudBuffer>(points), guik::Rainbow());

  gtsam_ext::iVox ivox(1.0);
  ivox.insert(points.data(), points.size());

  std::mt19937 mt;
  for (int i = 0; i < 1024; i++) {
    const int point_id = std::uniform_int_distribution<>(0, points.size() - 1)(mt);
    const auto& point = points[point_id];

    Eigen::Affine3d model_matrix = Eigen::Isometry3d::Identity() * Eigen::Translation3d(point.head<3>()) * Eigen::UniformScaling<double>(0.3);
    viewer->update_drawable("point", glk::Primitives::sphere(), guik::FlatRed(model_matrix.cast<float>()));

    std::cout << "model_matrix" << std::endl << model_matrix.matrix() << std::endl;

    const int k = 20;
    std::vector<size_t> k_indices(k);
    std::vector<double> k_sq_dists(k);
    const auto num_neighbors = ivox.knn_search(point.data(), k, k_indices.data(), k_sq_dists.data());

    std::vector<Eigen::Vector4d> neighbor_points;

    std::cout << "num_neighbors:" << num_neighbors << std::endl;
    for (int i = 0; i < num_neighbors; i++) {
      neighbor_points.push_back(ivox.point(k_indices[i]));
    }
    viewer->update_drawable("neighbors", std::make_shared<glk::PointCloudBuffer>(neighbor_points), guik::FlatOrange().add("point_scale", 3.0f));

    viewer->spin_until_click();
  }

  return 0;
}