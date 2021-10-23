#include <gtsam_ext/types/frame_cpu.hpp>

#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace gtsam_ext {

template <typename T, int D>
FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& points) {
  points_storage.resize(points.size(), Eigen::Vector4d(0, 0, 0, 1));
  for (int i = 0; i < points.size(); i++) {
    points_storage[i].head<D>() = points[i].template head<D>().template cast<double>();
  }

  this->num_points = points.size();
  this->points = &points_storage[0];
}

FrameCPU::FrameCPU(const Frame& frame) {
  num_points = frame.size();

  points_storage.assign(frame.points, frame.points + frame.size());
  points = points_storage.data();

  if (frame.times) {
    times_storage.assign(frame.times, frame.times + frame.size());
    times = times_storage.data();
  }

  if (frame.normals) {
    normals_storage.assign(frame.normals, frame.normals + frame.size());
    normals = normals_storage.data();
  }

  if (frame.covs) {
    covs_storage.assign(frame.covs, frame.covs + frame.size());
    covs = covs_storage.data();
  }
}

FrameCPU::FrameCPU() {}

FrameCPU::~FrameCPU() {}

template <typename T>
void FrameCPU::add_times(const std::vector<T>& times) {
  assert(times.size() == size());
  times_storage.resize(times.size());
  std::transform(times.begin(), times.end(), times_storage.begin(), [](const auto& t) { return static_cast<double>(t); });
  this->times = times_storage.data();
}

template <typename T, int D>
void FrameCPU::add_normals(const std::vector<Eigen::Matrix<T, D, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, D, 1>>>& normals) {
  assert(normals.size() == size());
  normals_storage.resize(normals.size(), Eigen::Vector4d::Zero());
  for (int i = 0; i < normals.size(); i++) {
    normals_storage[i].head<D>() = normals[i].template head<D>().template cast<double>();
  }
  this->normals = normals_storage.data();
}

template <typename T, int D>
void FrameCPU::add_covs(const std::vector<Eigen::Matrix<T, D, D>, Eigen::aligned_allocator<Eigen::Matrix<T, D, D>>>& covs) {
  assert(covs.size() == size());
  covs_storage.resize(covs.size(), Eigen::Matrix4d::Zero());
  for (int i = 0; i < covs.size(); i++) {
    covs_storage[i].block<D, D>(0, 0) = covs[i].template block<D, D>(0, 0).template cast<double>();
  }
  this->covs = covs_storage.data();
}

template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1>>>&);
template FrameCPU::FrameCPU(const std::vector<Eigen::Matrix<double, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 1>>>&);

template void FrameCPU::add_times(const std::vector<float>& times);
template void FrameCPU::add_times(const std::vector<double>& times);
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
      std::transform(normals_f.begin(), normals_f.end(), frame->normals, [](const Eigen::Vector3f& p) { return Eigen::Vector4d(p[0], p[1], p[2], 1.0); });
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
  } else {
    std::cerr << "error: " << path << " does not constain points(_compact)?.bin" << std::endl;
    return nullptr;
  }

  return frame;
}

}  // namespace gtsam_ext
