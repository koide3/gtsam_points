#include <gtsam_ext/types/frame_cpu.hpp>

#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace gtsam_ext {

template <typename T, int D>
FrameCPU::FrameCPU(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  add_points(points, num_points);
}

template <typename T, int D>
FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) : FrameCPU(points.data(), points.size()) {}

FrameCPU::FrameCPU(const Frame& frame) {
  if (frame.points) {
    add_points(frame.points, frame.size());
  }

  if (frame.times) {
    add_times(frame.times, frame.size());
  }

  if (frame.normals) {
    add_normals(frame.normals, frame.size());
  }

  if (frame.covs) {
    add_covs(frame.covs, frame.size());
  }
}

FrameCPU::FrameCPU() {}

FrameCPU::~FrameCPU() {}

template <typename T>
void FrameCPU::add_times(const T* times, int num_points) {
  assert(num_points == size());
  times_storage.resize(num_points);
  std::copy(times, times + num_points, times_storage.begin());
  this->times = this->times_storage.data();
}

template <typename T>
void FrameCPU::add_times(const std::vector<T>& times) {
  add_times(times.data(), times.size());
}

template <typename T, int D>
void FrameCPU::add_points(const Eigen::Matrix<T, D, 1>* points, int num_points) {
  points_storage.resize(num_points, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
  for (int i = 0; i < num_points; i++) {
    points_storage[i].head<D>() = points[i].template head<D>().template cast<double>();
  }
  this->points = points_storage.data();
  this->num_points = num_points;
}

template <typename T, int D>
void FrameCPU::add_points(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  add_points(points.data(), points.size());
}

template <typename T, int D>
void FrameCPU::add_normals(const Eigen::Matrix<T, D, 1>* normals, int num_points) {
  assert(num_points == size());
  normals_storage.resize(num_points, Eigen::Vector4d::Zero());
  for (int i = 0; i < num_points; i++) {
    normals_storage[i].head<D>() = normals[i].template head<D>().template cast<double>();
  }
  this->normals = normals_storage.data();
}

template <typename T, int D>
void FrameCPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  add_normals(normals.data(), normals.size());
}

template <typename T, int D>
void FrameCPU::add_covs(const Eigen::Matrix<T, D, D>* covs, int num_points) {
  assert(num_points == size());
  covs_storage.resize(num_points, Eigen::Matrix4d::Zero());
  for (int i = 0; i < num_points; i++) {
    covs_storage[i].block<D, D>(0, 0) = covs[i].template block<D, D>(0, 0).template cast<double>();
  }
  this->covs = covs_storage.data();
}

template <typename T, int D>
void FrameCPU::add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  add_covs(covs.data(), covs.size());
}

template <typename T>
void FrameCPU::add_intensities(const T* intensities, int num_points) {
  assert(num_points == size());
  intensities_storage.resize(num_points);
  std::copy(intensities, intensities + num_points, intensities_storage.begin());
  this->intensities = this->intensities_storage.data();
}

template <typename T>
void FrameCPU::add_intensities(const std::vector<T>& intensities) {
  add_intensities(intensities.data(), intensities.size());
}

template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void FrameCPU::add_intensities(const float* intensities, int num_points);
template void FrameCPU::add_intensities(const double* intensities, int num_points);

template void FrameCPU::add_times(const std::vector<float>& times);
template void FrameCPU::add_times(const std::vector<double>& times);
template void FrameCPU::add_intensities(const std::vector<float>& intensities);
template void FrameCPU::add_intensities(const std::vector<double>& intensities);
template void FrameCPU::add_points(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>& points);
template void FrameCPU::add_points(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>& points);
template void FrameCPU::add_points(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>& points);
template void FrameCPU::add_points(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>& points);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>& normals);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>& normals);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>& normals);
template void FrameCPU::add_normals(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>& normals);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<float, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 3>>>& covs);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<float, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 4>>>& covs);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3>>>& covs);
template void FrameCPU::add_covs(const std::vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4>>>& covs);

FrameCPU::Ptr random_sampling(const Frame::ConstPtr& frame, const double sampling_rate, std::mt19937& mt) {
  if (sampling_rate >= 0.99) {
    return FrameCPU::Ptr(new FrameCPU(*frame));
  }

  const int num_samples = frame->size() * sampling_rate;

  std::vector<int> sample_indices(num_samples);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);
  std::sample(boost::counting_iterator<int>(0), boost::counting_iterator<int>(frame->size()), sample_indices.begin(), num_samples, mt);
  std::sort(sample_indices.begin(), sample_indices.end());

  FrameCPU::Ptr sampled(new FrameCPU);
  sampled->num_points = num_samples;

  sampled->points_storage.resize(num_samples);
  sampled->points = sampled->points_storage.data();
  for (int i = 0; i < num_samples; i++) {
    sampled->points[i] = frame->points[sample_indices[i]];
  }

  if (frame->times) {
    sampled->times_storage.resize(num_samples);
    sampled->times = sampled->times_storage.data();
    for (int i = 0; i < num_samples; i++) {
      sampled->times[i] = frame->times[sample_indices[i]];
    }
  }

  if (frame->covs) {
    sampled->covs_storage.resize(num_samples);
    sampled->covs = sampled->covs_storage.data();
    for (int i = 0; i < num_samples; i++) {
      sampled->covs[i] = frame->covs[sample_indices[i]];
    }
  }

  if (frame->normals) {
    sampled->normals_storage.resize(num_samples);
    sampled->normals = sampled->normals_storage.data();
    for (int i = 0; i < num_samples; i++) {
      sampled->normals[i] = frame->normals[sample_indices[i]];
    }
  }

  if (frame->intensities) {
    sampled->intensities_storage.resize(num_samples);
    sampled->intensities = sampled->intensities_storage.data();
    for (int i = 0; i < num_samples; i++) {
      sampled->intensities[i] = frame->intensities[sample_indices[i]];
    }
  }

  return sampled;
}

FrameCPU::Ptr FrameCPU::load(const std::string& path) {
  FrameCPU::Ptr frame(new FrameCPU);

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
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points_f(num_points);

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
      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normals_f(frame->size());

      std::ifstream ifs(path + "/normals_compact.bin", std::ios::binary);
      ifs.read(reinterpret_cast<char*>(normals_f.data()), sizeof(Eigen::Vector3f) * frame->size());
      std::transform(normals_f.begin(), normals_f.end(), frame->normals, [](const Eigen::Vector3f& p) { return Eigen::Vector4d(p[0], p[1], p[2], 0.0); });
    }

    if (boost::filesystem::exists(path + "/covs_compact.bin")) {
      frame->covs_storage.resize(frame->size());
      frame->covs = frame->covs_storage.data();
      std::vector<Eigen::Matrix<float, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 6, 1>>> covs_f(frame->size());

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

  return frame;
}

}  // namespace gtsam_ext
