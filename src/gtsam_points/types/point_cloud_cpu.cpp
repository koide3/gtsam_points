// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <regex>
#include <numeric>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/util/sort_omp.hpp>
#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/util/vector3i_hash.hpp>
#include <gtsam_points/util/parallelism.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#endif

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
      ifs.read(reinterpret_cast<char*>(intensities_f.data()), sizeof(Eigen::Vector4f) * frame->size());
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

// sample
PointCloudCPU::Ptr sample(const PointCloud::ConstPtr& frame, const std::vector<int>& indices) {
  PointCloudCPU::Ptr sampled(new PointCloudCPU);
  sampled->num_points = indices.size();
  sampled->points_storage.resize(indices.size());
  sampled->points = sampled->points_storage.data();
  std::transform(indices.begin(), indices.end(), sampled->points, [&](const int i) { return frame->points[i]; });

  if (frame->times) {
    sampled->times_storage.resize(indices.size());
    sampled->times = sampled->times_storage.data();
    std::transform(indices.begin(), indices.end(), sampled->times, [&](const int i) { return frame->times[i]; });
  }

  if (frame->normals) {
    sampled->normals_storage.resize(indices.size());
    sampled->normals = sampled->normals_storage.data();
    std::transform(indices.begin(), indices.end(), sampled->normals, [&](const int i) { return frame->normals[i]; });
  }

  if (frame->covs) {
    sampled->covs_storage.resize(indices.size());
    sampled->covs = sampled->covs_storage.data();
    std::transform(indices.begin(), indices.end(), sampled->covs, [&](const int i) { return frame->covs[i]; });
  }

  if (frame->intensities) {
    sampled->intensities_storage.resize(indices.size());
    sampled->intensities = sampled->intensities_storage.data();
    std::transform(indices.begin(), indices.end(), sampled->intensities, [&](const int i) { return frame->intensities[i]; });
  }

  for (const auto& attrib : frame->aux_attributes) {
    const auto& name = attrib.first;
    const size_t elem_size = attrib.second.first;
    const unsigned char* data_ptr = static_cast<const unsigned char*>(attrib.second.second);

    auto storage = std::make_shared<std::vector<unsigned char>>(indices.size() * elem_size);
    for (int i = 0; i < indices.size(); i++) {
      const auto src = data_ptr + elem_size * indices[i];
      auto dst = storage->data() + elem_size * i;
      memcpy(dst, src, elem_size);
    }

    sampled->aux_attributes_storage[name] = storage;
    sampled->aux_attributes[name] = std::make_pair(elem_size, storage->data());
  }

  return sampled;
}

// random_sampling
PointCloudCPU::Ptr random_sampling(const PointCloud::ConstPtr& frame, const double sampling_rate, std::mt19937& mt) {
  if (sampling_rate >= 0.99) {
    // No need to do sampling
    return PointCloudCPU::Ptr(PointCloudCPU::clone(*frame));
  }

  const int num_samples = frame->size() * sampling_rate;

  std::vector<int> sample_indices(num_samples);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);
  std::sample(boost::counting_iterator<int>(0), boost::counting_iterator<int>(frame->size()), sample_indices.begin(), num_samples, mt);
  std::sort(sample_indices.begin(), sample_indices.end());

  return sample(frame, sample_indices);
}

template <typename T>
struct Averager {
public:
  Averager(const T& zero) : num(0), sum(zero) {}

  const Averager& operator+=(const T& value) {
    sum += value;
    num++;
    return *this;
  }

  const void clear(const T& zero) {
    num = 0;
    sum = zero;
  }

  const T average() const { return sum / num; }

public:
  size_t num;
  T sum;
};

// voxelgrid_sampling
PointCloudCPU::Ptr voxelgrid_sampling(const PointCloud::ConstPtr& frame, const double voxel_resolution, int num_threads) {
  if (frame->size() == 0) {
    return PointCloudCPU::Ptr(new PointCloudCPU());
  }

  constexpr std::int64_t invalid_coord = std::numeric_limits<std::int64_t>::max();
  constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3=63bits in 64bit int)
  constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive
  const double inv_resolution = 1.0 / voxel_resolution;

  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(frame->size());
  const auto calc_coord = [&](std::int64_t i) -> std::uint64_t {
    if (!frame->points[i].array().isFinite().all()) {
      return invalid_coord;
    }

    const Eigen::Array4i coord = fast_floor(frame->points[i] * inv_resolution) + coord_offset;
    if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
      std::cerr << "warning: voxel coord is out of range!!" << std::endl;
      return invalid_coord;
    }

    // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
    const std::uint64_t bits =                                                           //
      (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |  //
      (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |  //
      (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
    return bits;
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 32)
    for (std::int64_t i = 0; i < frame->size(); i++) {
      coord_pt[i] = {calc_coord(i), i};
    }
    // Sort by voxel coords
    quick_sort_omp(coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }, num_threads);
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<std::int64_t>(0, frame->size(), 32), [&](const tbb::blocked_range<std::int64_t>& range) {
      for (std::int64_t i = range.begin(); i < range.end(); i++) {
        coord_pt[i] = {calc_coord(i), i};
      }
    });

    // Sort by voxel coords
    tbb::parallel_sort(coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  PointCloudCPU::Ptr downsampled(new PointCloudCPU);
  downsampled->points_storage.resize(frame->size());
  if (frame->times) {
    downsampled->times_storage.resize(frame->size());
  }
  if (frame->normals) {
    downsampled->normals_storage.resize(frame->size());
  }
  if (frame->covs) {
    downsampled->covs_storage.resize(frame->size());
  }
  if (frame->intensities) {
    downsampled->intensities_storage.resize(frame->size());
  }

  const int block_size = 1024;
  std::atomic_uint64_t num_points = 0;

  const auto perblock_task = [&](std::int64_t block_begin) {
    std::vector<std::pair<std::uint64_t, size_t>*> sub_blocks;
    sub_blocks.reserve(block_size);

    const size_t block_end = std::min<size_t>(coord_pt.size(), block_begin + block_size);

    sub_blocks.emplace_back(coord_pt.data() + block_begin);
    for (size_t i = block_begin + 1; i != block_end; i++) {
      if (coord_pt[i - 1].first != coord_pt[i].first) {
        sub_blocks.emplace_back(coord_pt.data() + i);
      }
    }
    sub_blocks.emplace_back(coord_pt.data() + block_end);

    const size_t point_index_begin = num_points.fetch_add(sub_blocks.size() - 1);
    for (int i = 0; i < sub_blocks.size() - 1; i++) {
      Averager<Eigen::Vector4d> average(Eigen::Vector4d::Zero());
      for (auto pt = sub_blocks[i]; pt != sub_blocks[i + 1]; pt++) {
        average += frame->points[pt->second];
      }
      downsampled->points_storage[point_index_begin + i] = average.average();

      if (frame->times) {
        Averager<double> time_average(0.0);
        for (auto pt = sub_blocks[i]; pt != sub_blocks[i + 1]; pt++) {
          time_average += frame->times[pt->second];
        }
        downsampled->times_storage[point_index_begin + i] = time_average.average();
      }

      if (frame->normals) {
        Averager<Eigen::Vector4d> normal_average(Eigen::Vector4d::Zero());
        for (auto pt = sub_blocks[i]; pt != sub_blocks[i + 1]; pt++) {
          normal_average += frame->normals[pt->second];
        }
        downsampled->normals_storage[point_index_begin + i] = normal_average.average();
      }

      if (frame->covs) {
        Averager<Eigen::Matrix4d> cov_average(Eigen::Matrix4d::Zero());
        for (auto pt = sub_blocks[i]; pt != sub_blocks[i + 1]; pt++) {
          cov_average += frame->covs[pt->second];
        }
        downsampled->covs_storage[point_index_begin + i] = cov_average.average();
      }

      if (frame->intensities) {
        Averager<double> intensity_average(0.0);
        for (auto pt = sub_blocks[i]; pt != sub_blocks[i + 1]; pt++) {
          intensity_average += frame->intensities[pt->second];
        }
        downsampled->intensities_storage[point_index_begin + i] = intensity_average.average();
      }
    }
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 4)
    for (std::int64_t block_begin = 0; block_begin < coord_pt.size(); block_begin += block_size) {
      perblock_task(block_begin);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    const size_t num_blocks = (coord_pt.size() + block_size - 1) / block_size;
    tbb::parallel_for(tbb::blocked_range<std::int64_t>(0, num_blocks), [&](const tbb::blocked_range<std::int64_t>& range) {
      for (std::int64_t block_begin = range.begin() * block_size; block_begin < range.end() * block_size; block_begin += block_size) {
        perblock_task(block_begin);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  downsampled->num_points = num_points;
  downsampled->points_storage.resize(num_points);
  downsampled->points = downsampled->points_storage.data();

  if (frame->times) {
    downsampled->times_storage.resize(num_points);
    downsampled->times = downsampled->times_storage.data();
  }

  if (frame->normals) {
    downsampled->normals_storage.resize(num_points);
    downsampled->normals = downsampled->normals_storage.data();
  }

  if (frame->covs) {
    downsampled->covs_storage.resize(num_points);
    downsampled->covs = downsampled->covs_storage.data();
  }

  if (frame->intensities) {
    downsampled->intensities_storage.resize(num_points);
    downsampled->intensities = downsampled->intensities_storage.data();
  }

  if (!frame->aux_attributes.empty()) {
    std::cout << "warning: voxelgrid_sampling does not support aux attributes!!" << std::endl;
  }

  return downsampled;
}

// randomgrid_sampling
PointCloudCPU::Ptr
randomgrid_sampling(const PointCloud::ConstPtr& frame, const double voxel_resolution, const double sampling_rate, std::mt19937& mt, int num_threads) {
  if (sampling_rate >= 0.99) {
    // No need to do sampling
    return PointCloudCPU::clone(*frame);
  }

  constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3=63bits in 64bit int)
  constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive
  const double inv_resolution = 1.0 / voxel_resolution;

  const auto calc_coord = [&](std::int64_t i) -> std::uint64_t {
    if (!frame->points[i].array().isFinite().all()) {
      return invalid_coord;
    }

    const Eigen::Array4i coord = fast_floor(frame->points[i] * inv_resolution) + coord_offset;
    if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
      std::cerr << "warning: voxel coord is out of range!!" << std::endl;
      return invalid_coord;
    }

    // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
    const std::uint64_t bits =                                                           //
      (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |  //
      (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |  //
      (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
    return bits;
  };

  size_t num_voxels = 0;
  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(frame->size());

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 32)
    for (std::int64_t i = 0; i < frame->size(); i++) {
      coord_pt[i] = {calc_coord(i), i};
    }

    // Sort by voxel coords
    quick_sort_omp(coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }, num_threads);

#pragma omp parallel for num_threads(num_threads) schedule(guided, 128) reduction(+ : num_voxels)
    for (size_t i = 1; i < coord_pt.size(); i++) {
      if (coord_pt[i - 1].first != coord_pt[i].first) {
        num_voxels++;
      }
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<std::int64_t>(0, frame->size(), 32), [&](const tbb::blocked_range<std::int64_t>& range) {
      for (std::int64_t i = range.begin(); i < range.end(); i++) {
        coord_pt[i] = {calc_coord(i), i};
      }
    });

    // Sort by voxel coords
    tbb::parallel_sort(coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::atomic_uint64_t num_voxels_ = 0;
    tbb::parallel_for(tbb::blocked_range<std::int64_t>(1, coord_pt.size(), 128), [&](const tbb::blocked_range<std::int64_t>& range) {
      size_t local_num_voxels = 0;
      for (size_t i = range.begin(); i < range.end(); i++) {
        if (coord_pt[i - 1].first != coord_pt[i].first) {
          local_num_voxels++;
        }
      }
      num_voxels_ += local_num_voxels;
    });

    num_voxels = num_voxels_;
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  const size_t points_per_voxel = std::ceil((sampling_rate * frame->size()) / num_voxels);
  const size_t max_num_points = frame->size() * sampling_rate * 1.2;

  const int block_size = 1024;
  std::atomic_uint64_t num_points = 0;

  std::vector<std::mt19937> mts(num_threads);
  std::generate(mts.begin(), mts.end(), [&mt]() { return std::mt19937(mt()); });

  std::vector<int> indices(frame->size());

  const auto perblock_task = [&](std::int64_t block_begin) {
    std::vector<size_t> sub_indices;
    sub_indices.reserve(block_size);

    std::vector<size_t> block_indices;
    block_indices.reserve(block_size);

    const auto flush_block_indices = [&] {
      if (block_indices.size() < points_per_voxel) {
        sub_indices.insert(sub_indices.end(), block_indices.begin(), block_indices.end());
      } else {
        int thread_num = 0;
#ifdef _OPENMP
        thread_num = omp_get_thread_num();
#endif
        std::sample(block_indices.begin(), block_indices.end(), std::back_inserter(sub_indices), points_per_voxel, mts[thread_num]);
      }
      block_indices.clear();
    };

    const size_t block_end = std::min<size_t>(coord_pt.size(), block_begin + block_size);
    for (size_t i = block_begin; i != block_end; i++) {
      if (i != block_begin && coord_pt[i - 1].first != coord_pt[i].first) {
        flush_block_indices();
      }

      if (coord_pt[i].first != invalid_coord) {
        block_indices.emplace_back(coord_pt[i].second);
      }
    }
    flush_block_indices();

    std::copy(sub_indices.begin(), sub_indices.end(), indices.begin() + num_points.fetch_add(sub_indices.size()));
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 4)
    for (std::int64_t block_begin = 0; block_begin < coord_pt.size(); block_begin += block_size) {
      perblock_task(block_begin);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    const size_t num_blocks = (coord_pt.size() + block_size - 1) / block_size;
    tbb::parallel_for(tbb::blocked_range<std::int64_t>(0, num_blocks), [&](const tbb::blocked_range<std::int64_t>& range) {
      for (std::int64_t block_begin = range.begin() * block_size; block_begin < range.end() * block_size; block_begin += block_size) {
        perblock_task(block_begin);
      }
    });

#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }

  indices.resize(num_points);

  if (indices.size() > max_num_points) {
    std::vector<int> sub_indices(max_num_points);
    std::sample(indices.begin(), indices.end(), sub_indices.begin(), max_num_points, mt);
    indices = std::move(sub_indices);
  }

  // Sort indices to keep points ordered (and for better memory accessing)
  quick_sort_omp(indices.begin(), indices.end(), std::less<int>(), num_threads);

  // Sample points and return it
  return sample(frame, indices);
}

// sort_by_time
PointCloudCPU::Ptr sort_by_time(const PointCloud::ConstPtr& frame) {
  if (!frame->has_times()) {
    std::cerr << "warning: frame does not have per-point times" << std::endl;
  }

  return sort(frame, [&](const int lhs, const int rhs) { return frame->times[lhs] < frame->times[rhs]; });
}

// transform
template <>
PointCloudCPU::Ptr transform(const PointCloud::ConstPtr& frame, const Eigen::Transform<double, 3, Eigen::Isometry>& transformation) {
  auto transformed = gtsam_points::PointCloudCPU::clone(*frame);
  for (int i = 0; i < frame->size(); i++) {
    transformed->points[i] = transformation * frame->points[i];
  }

  if (frame->normals) {
    for (int i = 0; i < frame->size(); i++) {
      transformed->normals[i] = transformation.matrix() * frame->normals[i];
    }
  }

  if (frame->covs) {
    for (int i = 0; i < frame->size(); i++) {
      transformed->covs[i] = transformation.matrix() * frame->covs[i] * transformation.matrix().transpose();
    }
  }

  return transformed;
}

template <>
PointCloudCPU::Ptr transform(const PointCloud::ConstPtr& frame, const Eigen::Transform<float, 3, Eigen::Isometry>& transformation) {
  return transform<double, Eigen::Isometry>(frame, transformation.cast<double>());
}

template <>
PointCloudCPU::Ptr transform(const PointCloud::ConstPtr& frame, const Eigen::Transform<double, 3, Eigen::Affine>& transformation) {
  auto transformed = PointCloudCPU::clone(*frame);
  for (int i = 0; i < frame->size(); i++) {
    transformed->points[i] = transformation * frame->points[i];
  }

  if (frame->normals) {
    Eigen::Matrix4d normal_matrix = Eigen::Matrix4d::Zero();
    normal_matrix.block<3, 3>(0, 0) = transformation.linear().inverse().transpose();
    for (int i = 0; i < frame->size(); i++) {
      transformed->normals[i] = normal_matrix * frame->normals[i];
    }
  }

  if (frame->covs) {
    for (int i = 0; i < frame->size(); i++) {
      transformed->covs[i] = transformation.matrix() * frame->covs[i] * transformation.matrix().transpose();
    }
  }

  return transformed;
}

template <>
PointCloudCPU::Ptr transform(const PointCloud::ConstPtr& frame, const Eigen::Transform<float, 3, Eigen::Affine>& transformation) {
  return transform<double, Eigen::Affine>(frame, transformation.cast<double>());
}

// transform_inplace
template <>
void transform_inplace(PointCloud& frame, const Eigen::Transform<double, 3, Eigen::Isometry>& transformation) {
  for (int i = 0; i < frame.size(); i++) {
    frame.points[i] = transformation * frame.points[i];
  }

  if (frame.normals) {
    for (int i = 0; i < frame.size(); i++) {
      frame.normals[i] = transformation.matrix() * frame.normals[i];
    }
  }

  if (frame.covs) {
    for (int i = 0; i < frame.size(); i++) {
      frame.covs[i] = transformation.matrix() * frame.covs[i] * transformation.matrix().transpose();
    }
  }
}

template <>
void transform_inplace(PointCloud& frame, const Eigen::Transform<float, 3, Eigen::Isometry>& transformation) {
  transform_inplace<double, Eigen::Isometry>(frame, transformation.cast<double>());
}

template <>
void transform_inplace(PointCloud& frame, const Eigen::Transform<double, 3, Eigen::Affine>& transformation) {
  for (int i = 0; i < frame.size(); i++) {
    frame.points[i] = transformation * frame.points[i];
  }

  if (frame.normals) {
    Eigen::Matrix4d normal_matrix = Eigen::Matrix4d::Zero();
    normal_matrix.block<3, 3>(0, 0) = transformation.linear().inverse().transpose();
    for (int i = 0; i < frame.size(); i++) {
      frame.normals[i] = normal_matrix * frame.normals[i];
    }
  }

  if (frame.covs) {
    for (int i = 0; i < frame.size(); i++) {
      frame.covs[i] = transformation.matrix() * frame.covs[i] * transformation.matrix().transpose();
    }
  }
}

template <>
void transform_inplace(PointCloud& frame, const Eigen::Transform<float, 3, Eigen::Affine>& transformation) {
  return transform_inplace<double, Eigen::Affine>(frame, transformation.cast<double>());
}

// statistical outlier removal
std::vector<int> find_inlier_points(const PointCloud::ConstPtr& frame, const std::vector<int>& neighbors, const int k, const double std_thresh) {
  std::vector<double> dists(frame->size());

  for (int i = 0; i < frame->size(); i++) {
    const auto& pt = frame->points[i];

    double sum_dist = 0.0;
    for (int j = 0; j < k; j++) {
      const int index = neighbors[i * k + j];
      sum_dist += (frame->points[index] - pt).norm();
    }

    dists[i] = sum_dist / k;
  }

  double sum_dists = 0.0;
  double sum_sq_dists = 0.0;
  for (int i = 0; i < dists.size(); i++) {
    sum_dists += dists[i];
    sum_sq_dists += dists[i] * dists[i];
  }

  const double mean = sum_dists / frame->size();
  const double var = sum_sq_dists / frame->size() - mean * mean;
  const double dist_thresh = mean + std::sqrt(var) * std_thresh;

  std::vector<int> inliers;
  inliers.reserve(frame->size());

  for (int i = 0; i < frame->size(); i++) {
    if (dists[i] < dist_thresh) {
      inliers.emplace_back(i);
    }
  }

  return inliers;
}

PointCloudCPU::Ptr remove_outliers(const PointCloud::ConstPtr& frame, const std::vector<int>& neighbors, const int k, const double std_thresh) {
  const auto inliers = find_inlier_points(frame, neighbors, k, std_thresh);
  return sample(frame, inliers);
}

PointCloudCPU::Ptr remove_outliers(const PointCloud::ConstPtr& frame, const int k, const double std_thresh, const int num_threads) {
  KdTree kdtree(frame->points, frame->size());

  std::vector<int> neighbors(frame->size() * k, -1);

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_neighbors(k);
    std::vector<double> k_sq_dists(k);
    kdtree.knn_search(frame->points[i].data(), k, k_neighbors.data(), k_sq_dists.data());
    std::copy(k_neighbors.begin(), k_neighbors.end(), neighbors.begin() + i * k);
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for schedule(guided, 8) num_threads(num_threads)
    for (int i = 0; i < frame->size(); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame->size(), 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i != range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error : TBB is not available" << std::endl;
    abort();
#endif
  }

  return remove_outliers(frame, neighbors, k, std_thresh);
}

std::vector<double> distances(const PointCloud::ConstPtr& points, size_t max_scan_count) {
  std::vector<double> dists;
  dists.reserve(points->size() < max_scan_count ? points->size() : max_scan_count * 2);

  const size_t step = points->size() < max_scan_count ? 1 : points->size() / max_scan_count;
  for (size_t i = 0; i < points->size(); i += step) {
    const auto& p = points->points[i];
    dists.emplace_back(p.head<3>().norm());
  }

  return dists;
}

std::pair<double, double> minmax_distance(const PointCloud::ConstPtr& points, size_t max_scan_count) {
  const auto dists = distances(points, max_scan_count);
  if (dists.empty()) {
    return {0.0, 0.0};
  }

  const auto minmax = std::minmax_element(dists.begin(), dists.end());
  return {*minmax.first, *minmax.second};
}

double median_distance(const PointCloud::ConstPtr& points, size_t max_scan_count) {
  auto dists = distances(points, max_scan_count);
  if (dists.empty()) {
    return 0.0;
  }

  std::nth_element(dists.begin(), dists.begin() + dists.size() / 2, dists.end());
  return dists[dists.size() / 2];
}

}  // namespace gtsam_points
