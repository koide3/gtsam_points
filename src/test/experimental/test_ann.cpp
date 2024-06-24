#include <chrono>
#include <iostream>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/projective_search.hpp>

int main(int argc, char** argv) {
  auto points0 = gtsam_points::read_points("data/kitti_07_dump/000000/points.bin");
  auto points1 = gtsam_points::read_points("data/kitti_07_dump/000001/points.bin");

  auto frame0 = std::make_shared<gtsam_points::PointCloudCPU>(points0);
  auto frame1 = std::make_shared<gtsam_points::PointCloudCPU>(points1);

  gtsam_points::KdTree kdtree(frame0->points, frame0->size());
  gtsam_points::OmniProjectiveSearch search(frame0->points, frame0->size());

  for (int i = 0; i < frame1->size(); i++) {
    size_t k_index;
    double k_sq_dist;
    size_t num_found = search.knn_search(frame1->points[i].data(), 1, &k_index, &k_sq_dist);

    if (num_found == 0) {
      std::cerr << "No neighbors. You Alone!!" << std::endl;
      continue;
    }

    std::cout << "---" << std::endl;
    std::cout << i << ":" << k_index << " " << frame1->points[i].transpose() << ":" << frame0->points[k_index].transpose() << " d=" << k_sq_dist << std::endl;

    kdtree.knn_search(frame1->points[i].data(), 1, &k_index, &k_sq_dist);
    std::cout << i << ":" << k_index << " " << frame1->points[i].transpose() << ":" << frame0->points[k_index].transpose() << " d=" << k_sq_dist << std::endl;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < frame1->size(); i++) {
    size_t k_index;
    double k_sq_dist;
    size_t num_found = kdtree.knn_search(frame1->points[i].data(), 1, &k_index, &k_sq_dist);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < frame1->size(); i++) {
    size_t k_index;
    double k_sq_dist;
    size_t num_found = search.knn_search(frame1->points[i].data(), 1, &k_index, &k_sq_dist);
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6 << "[msec]" << std::endl;
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e6 << "[msec]" << std::endl;

  return 0;
}