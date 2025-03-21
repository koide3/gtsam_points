// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <regex>
#include <numeric>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

#include <gtsam_points/config.hpp>
#include <gtsam_points/util/parallelism.hpp>

namespace gtsam_points {

// constructors & deconstructor
template <typename T, int D>
PointCloudCPU::PointCloudCPU(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  add_points(points, num_points);
}

template PointCloudCPU::PointCloudCPU(const Eigen::Matrix<float, 3, 1>* points, int num_points);
template PointCloudCPU::PointCloudCPU(const Eigen::Matrix<float, 4, 1>* points, int num_points);
template PointCloudCPU::PointCloudCPU(const Eigen::Matrix<double, 3, 1>* points, int num_points);
template PointCloudCPU::PointCloudCPU(const Eigen::Matrix<double, 4, 1>* points, int num_points);

PointCloudCPU::PointCloudCPU() {}

PointCloudCPU::~PointCloudCPU() {}

PointCloudCPU::Ptr PointCloudCPU::clone(const PointCloud& points) {
  auto new_points = std::make_shared<gtsam_points::PointCloudCPU>();

  if (points.points) {
    new_points->add_points(points.points, points.size());
  }

  if (points.times) {
    new_points->add_times(points.times, points.size());
  }

  if (points.normals) {
    new_points->add_normals(points.normals, points.size());
  }

  if (points.covs) {
    new_points->add_covs(points.covs, points.size());
  }

  if (points.intensities) {
    new_points->add_intensities(points.intensities, points.size());
  }

  for (const auto& attrib : points.aux_attributes) {
    const auto& name = attrib.first;
    const size_t elem_size = attrib.second.first;
    const unsigned char* data_ptr = static_cast<const unsigned char*>(attrib.second.second);

    auto storage = std::make_shared<std::vector<unsigned char>>(points.size() * elem_size);
    memcpy(storage->data(), data_ptr, elem_size * points.size());

    new_points->aux_attributes_storage[name] = storage;
    new_points->aux_attributes[name] = std::make_pair(elem_size, storage->data());
  }

  return new_points;
}

// add_times
template <typename T>
void PointCloudCPU::add_times(const T* times, int num_points) {
  assert(num_points == size());
  times_storage.resize(num_points);
  if (times) {
    std::copy(times, times + num_points, times_storage.begin());
  }
  this->times = this->times_storage.data();
}

template void PointCloudCPU::add_times(const float* times, int num_points);
template void PointCloudCPU::add_times(const double* times, int num_points);

// add_points
template <typename T, int D>
void PointCloudCPU::add_points(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  points_storage.resize(num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
  if (points) {
    for (int i = 0; i < num_points; i++) {
      points_storage[i].head<D>() = points[i].template head<D>().template cast<double>();
    }
  }
  this->points = points_storage.data();
  this->num_points = num_points;
}

template void PointCloudCPU::add_points(const Eigen::Matrix<float, 3, 1>* points, int num_points);
template void PointCloudCPU::add_points(const Eigen::Matrix<float, 4, 1>* points, int num_points);
template void PointCloudCPU::add_points(const Eigen::Matrix<double, 3, 1>* points, int num_points);
template void PointCloudCPU::add_points(const Eigen::Matrix<double, 4, 1>* points, int num_points);

// add_normals
template <typename T, int D>
void PointCloudCPU::add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
  assert(num_points == size());
  normals_storage.resize(num_points, Eigen::Vector4d::Zero());
  if (normals) {
    for (int i = 0; i < num_points; i++) {
      normals_storage[i].head<D>() = normals[i].template head<D>().template cast<double>();
    }
  }
  this->normals = normals_storage.data();
}

template void PointCloudCPU::add_normals(const Eigen::Matrix<float, 3, 1>* normals, int num_points);
template void PointCloudCPU::add_normals(const Eigen::Matrix<float, 4, 1>* normals, int num_points);
template void PointCloudCPU::add_normals(const Eigen::Matrix<double, 3, 1>* normals, int num_points);
template void PointCloudCPU::add_normals(const Eigen::Matrix<double, 4, 1>* normals, int num_points);

// add_covs
template <typename T, int D>
void PointCloudCPU::add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points) {
  assert(num_points == size());
  covs_storage.resize(num_points, Eigen::Matrix4d::Zero());
  if (covs) {
    for (int i = 0; i < num_points; i++) {
      covs_storage[i].block<D, D>(0, 0) = covs[i].template block<D, D>(0, 0).template cast<double>();
    }
  }
  this->covs = covs_storage.data();
}

template void PointCloudCPU::add_covs(const Eigen::Matrix<float, 3, 3>* covs, int num_points);
template void PointCloudCPU::add_covs(const Eigen::Matrix<float, 4, 4>* covs, int num_points);
template void PointCloudCPU::add_covs(const Eigen::Matrix<double, 3, 3>* covs, int num_points);
template void PointCloudCPU::add_covs(const Eigen::Matrix<double, 4, 4>* covs, int num_points);

// add_intensities
template <typename T>
void PointCloudCPU::add_intensities(const T* intensities, int num_points) {
  assert(num_points == size());
  intensities_storage.resize(num_points);
  if (intensities) {
    std::copy(intensities, intensities + num_points, intensities_storage.begin());
  }
  this->intensities = this->intensities_storage.data();
}

template void PointCloudCPU::add_intensities(const float* intensities, int num_points);
template void PointCloudCPU::add_intensities(const double* intensities, int num_points);

// PointCloudCPU::load
PointCloudCPU::Ptr PointCloudCPU::load(const std::string& path) {
  PointCloudCPU::Ptr frame(new PointCloudCPU);

  if (boost::filesystem::exists(path + "/points.bin")) {
    std::ifstream ifs(path + "/points.bin", std::ios::binary | std::ios::ate);
    std::streamsize points_bytes = ifs.tellg();
    size_t num_points = points_bytes / (sizeof(Eigen::Vector4d));

    frame->num_points = num_points;
    frame->points_storage.resize(num_points);
    frame->points = frame->points_storage.data();

    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(frame->points), sizeof(Eigen::Vector4d) * frame->size());

    if (boost::filesystem::exists(path + "/times.bin")) {
      frame->times_storage.resize(frame->size());
      frame->times = frame->times_storage.data();
      std::ifstream ifs(path + "/times.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(frame->times), sizeof(double) * frame->size());
    }

    if (boost::filesystem::exists(path + "/normals.bin")) {
      frame->normals_storage.resize(frame->size());
      frame->normals = frame->normals_storage.data();
      std::ifstream ifs(path + "/normals.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(frame->normals), sizeof(Eigen::Vector4d) * frame->size());
    }

    if (boost::filesystem::exists(path + "/covs.bin")) {
      frame->covs_storage.resize(frame->size());
      frame->covs = frame->covs_storage.data();
      std::ifstream ifs(path + "/covs.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(frame->covs), sizeof(Eigen::Matrix4d) * frame->size());
    }

    if (boost::filesystem::exists(path + "/intensities.bin")) {
      frame->intensities_storage.resize(frame->size());
      frame->intensities = frame->intensities_storage.data();
      std::ifstream ifs(path + "/intensities.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(frame->intensities), sizeof(double) * frame->size());
    }
  } else if (boost::filesystem::exists(path + "/points_compact.bin")) {
    std::ifstream ifs(path + "/points_compact.bin", std::ios::binary | std::ios::ate);
    std::streamsize points_bytes = ifs.tellg();
    size_t num_points = points_bytes / (sizeof(Eigen::Vector3f));

    frame->num_points = num_points;
    frame->points_storage.resize(num_points);
    frame->points = frame->points_storage.data();
    std::vector<Eigen::Vector3f> points_f(num_points);

    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(points_f.data()), sizeof(Eigen::Vector3f) * frame->size());
    std::transform(points_f.begin(), points_f.end(), frame->points, [](const Eigen::Vector3f& p) { return Eigen::Vector4d(p[0], p[1], p[2], 1.0); });

    if (boost::filesystem::exists(path + "/times_compact.bin")) {
      frame->times_storage.resize(frame->size());
      frame->times = frame->times_storage.data();
      std::vector<float> times_f(frame->size());

      std::ifstream ifs(path + "/times_compact.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(times_f.data()), sizeof(float) * frame->size());
      std::copy(times_f.begin(), times_f.end(), frame->times);
    }

    if (boost::filesystem::exists(path + "/normals_compact.bin")) {
      frame->normals_storage.resize(frame->size());
      frame->normals = frame->normals_storage.data();
      std::vector<Eigen::Vector3f> normals_f(frame->size());

      std::ifstream ifs(path + "/normals_compact.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(normals_f.data()), sizeof(Eigen::Vector3f) * frame->size());
      std::transform(normals_f.begin(), normals_f.end(), frame->normals, [](const Eigen::Vector3f& p) {
        return Eigen::Vector4d(p[0], p[1], p[2], 0.0);
      });
    }

    if (boost::filesystem::exists(path + "/covs_compact.bin")) {
      frame->covs_storage.resize(frame->size());
      frame->covs = frame->covs_storage.data();
      std::vector<Eigen::Matrix<float, 6, 1>> covs_f(frame->size());

      std::ifstream ifs(path + "/covs_compact.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(covs_f.data()), sizeof(Eigen::Matrix<float, 6, 1>) * frame->size());
      std::transform(covs_f.begin(), covs_f.end(), frame->covs, [](const Eigen::Matrix<float, 6, 1>& c) {
        Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
        cov(0, 0) = c[0];
        cov(0, 1) = cov(1, 0) = c[1];
        cov(0, 2) = cov(2, 0) = c[2];
        cov(1, 1) = c[3];
        cov(1, 2) = cov(2, 1) = c[4];
        cov(2, 2) = c[5];
        return cov;
      });
    }

    if (boost::filesystem::exists(path + "/intensities_compact.bin")) {
      frame->intensities_storage.resize(frame->size());
      frame->intensities = frame->intensities_storage.data();
      std::vector<float> intensities_f(frame->size());

      std::ifstream ifs(path + "/intensities_compact.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(intensities_f.data()), sizeof(float) * frame->size());
      std::copy(intensities_f.begin(), intensities_f.end(), frame->intensities);
    }

  } else {
    std::cerr << "error: " << path << " does not constain points(_compact)?.bin" << std::endl;
    return nullptr;
  }

  boost::filesystem::directory_iterator itr(path);
  boost::filesystem::directory_iterator end;
  const std::regex aux_name_regex("/aux_([^_]+).bin");
  for (; itr != end; itr++) {
    std::smatch matched;
    if (!std::regex_search(itr->path().string(), matched, aux_name_regex)) {
      continue;
    }
    const std::string name = matched.str(1);

    std::ifstream ifs(itr->path().string(), std::ios::ate | std::ios::binary);
    const size_t bytes = ifs.tellg();
    ifs.seekg(0);

    const int elem_size = bytes / frame->size();
    if (elem_size * frame->size() != bytes) {
      std::cerr << "warning: elem_size=" << elem_size << " num_points=" << frame->size() << " bytes=" << bytes << std::endl;
      std::cerr << "       : bytes != elem_size * num_points" << std::endl;
    }

    auto storage = std::make_shared<std::vector<char>>(bytes);
    ifs.read(storage->data(), bytes);

    frame->aux_attributes_storage[name] = storage;
    frame->aux_attributes[name] = std::make_pair(elem_size, storage->data());
  }

  return frame;
}

}  // namespace gtsam_points
