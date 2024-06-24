#include <iostream>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <glk/io/ply_io.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  const auto points_f = gtsam_points::read_points("data/kitti_00/000000.bin");
  auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points_f);
  auto filtered = gtsam_points::remove_outliers(frame, 10, 1.0, 8);

  glk::save_ply_binary("/home/koide/000000.ply", points_f.data(), points_f.size());

  std::cout << frame->size() << " " << filtered->size() << std::endl;

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("points", std::make_shared<glk::PointCloudBuffer>(frame->points, frame->size()), guik::Rainbow());
  viewer->spin_until_click();

  viewer->update_drawable("points", std::make_shared<glk::PointCloudBuffer>(filtered->points, filtered->size()), guik::Rainbow());
  viewer->spin_until_click();

  return 0;
}