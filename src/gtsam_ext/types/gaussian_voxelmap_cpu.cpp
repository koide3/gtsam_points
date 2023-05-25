// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_ext/types/gaussian_voxelmap_cpu.hpp>

#include <memory>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_ext/util/easy_profiler.hpp>

namespace gtsam_ext {

GaussianVoxel::GaussianVoxel() {
  last_lru_count = 0;
  finalized = false;
  num_points = 0;
  mean.setZero();
  cov.setZero();
}

GaussianVoxel::~GaussianVoxel() {}

void GaussianVoxel::append(const Eigen::Vector4d& mean_, const Eigen::Matrix4d& cov_) {
  if (finalized) {
    mean = num_points * mean;
    cov = num_points * cov;
    finalized = false;
  }

  num_points++;
  mean += mean_;
  cov += cov_;
}

void GaussianVoxel::finalize() {
  if (!finalized) {
    const double s = 1.0 / num_points;
    mean *= s;
    cov *= s;

    finalized = true;
  }
}

GaussianVoxelMapCPU::GaussianVoxelMapCPU(double resolution) : lru_count(0), lru_cycle(10), lru_thresh(10), resolution(resolution) {}

GaussianVoxelMapCPU::~GaussianVoxelMapCPU() {}

Eigen::Vector3i GaussianVoxelMapCPU::voxel_coord(const Eigen::Vector4d& x) const {
  return (x.array() / resolution - 0.5).floor().cast<int>().head<3>();
}

GaussianVoxel::Ptr GaussianVoxelMapCPU::lookup_voxel(const Eigen::Vector3i& coord) const {
  auto found = voxels.find(coord);
  if (found == voxels.end()) {
    return nullptr;
  }

  found->second->last_lru_count = lru_count;
  return found->second;
}

void GaussianVoxelMapCPU::insert(const Frame& frame) {
  if (!frame::has_points(frame) || !frame::has_covs(frame)) {
    std::cerr << "error: points/covs not allocated!!" << std::endl;
    abort();
  }

  lru_count++;

  std::unordered_set<GaussianVoxel*> updated_voxels;
  for (int i = 0; i < frame::size(frame); i++) {
    Eigen::Vector3i coord = voxel_coord(frame::point(frame, i));

    auto found = voxels.find(coord);
    if (found == voxels.end()) {
      GaussianVoxel::Ptr voxel(new GaussianVoxel());
      // found = voxels.insert(found, std::make_pair(coord, voxel));

      found = voxels.emplace_hint(found, coord, voxel);
    }

    auto& voxel = found->second;
    voxel->last_lru_count = lru_count;
    voxel->append(frame::point(frame, i), frame::cov(frame, i));

    updated_voxels.insert(voxel.get());
  }

  // Remove voxels that are not used recently
  const int lru_horizon = lru_count - lru_thresh;
  if (lru_horizon > 0 && (lru_horizon % lru_cycle) == 0) {
    for (auto voxel = voxels.begin(), last = voxels.end(); voxel != last;) {
      if (voxel->second->last_lru_count < lru_horizon) {
        voxel = voxels.erase(voxel);
      } else {
        ++voxel;
      }
    }
  }

  for (const auto& voxel : updated_voxels) {
    voxel->finalize();
  }
}

namespace {

struct GaussianVoxelData {
public:
  GaussianVoxelData() {}

  GaussianVoxelData(const Eigen::Vector3i& coord, const GaussianVoxel& voxel) {
    const auto& mean = voxel.mean;
    const auto& cov = voxel.cov;

    this->coord = coord;
    this->num_points = voxel.num_points;
    this->mean << mean[0], mean[1], mean[2];
    this->cov << cov(0, 0), cov(0, 1), cov(0, 2), cov(1, 1), cov(1, 2), cov(2, 2);
  }

  std::pair<Eigen::Vector3i, GaussianVoxel::Ptr> uncompact() const {
    auto voxel = std::make_shared<GaussianVoxel>();
    voxel->finalized = true;
    voxel->num_points = num_points;
    voxel->mean << mean.cast<double>(), 1.0;

    voxel->cov(0, 0) = cov[0];
    voxel->cov(0, 1) = voxel->cov(1, 0) = cov[1];
    voxel->cov(0, 2) = voxel->cov(2, 0) = cov[2];
    voxel->cov(1, 1) = cov[3];
    voxel->cov(1, 2) = voxel->cov(2, 1) = cov[4];
    voxel->cov(2, 2) = cov[5];

    return std::make_pair(coord, voxel);
  }

public:
  Eigen::Vector3i coord;
  int num_points;
  Eigen::Vector3f mean;
  Eigen::Matrix<float, 6, 1> cov;
};

}  // namespace

void GaussianVoxelMapCPU::save_compact(const std::string& path) const {
  std::vector<GaussianVoxelData> flat_voxels(voxels.size());
  std::transform(voxels.begin(), voxels.end(), flat_voxels.begin(), [](const auto& voxel) { return GaussianVoxelData(voxel.first, *voxel.second); });

  std::ofstream ofs(path);
  ofs << "compact " << 1 << std::endl;
  ofs << "resolution " << resolution << std::endl;
  ofs << "lru_count " << lru_count << std::endl;
  ofs << "lru_cycle " << lru_cycle << std::endl;
  ofs << "lru_thresh " << lru_thresh << std::endl;
  ofs << "voxel_bytes " << sizeof(GaussianVoxelData) << std::endl;
  ofs << "num_voxels " << flat_voxels.size() << std::endl;

  ofs.write(reinterpret_cast<const char*>(flat_voxels.data()), sizeof(GaussianVoxelData) * flat_voxels.size());
}

GaussianVoxelMapCPU::Ptr GaussianVoxelMapCPU::load(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "error: failed to open " << path << std::endl;
    return nullptr;
  }

  auto voxelmap = std::make_shared<GaussianVoxelMapCPU>(1.0);

  std::string token;
  bool compact;
  int voxel_bytes;
  int num_voxels;

  ifs >> token >> compact;
  ifs >> token >> voxelmap->resolution;
  ifs >> token >> voxelmap->lru_count;
  ifs >> token >> voxelmap->lru_cycle;
  ifs >> token >> voxelmap->lru_thresh;
  ifs >> token >> voxel_bytes;
  ifs >> token >> num_voxels;
  std::getline(ifs, token);

  std::vector<GaussianVoxelData> flat_voxels(num_voxels);
  ifs.read(reinterpret_cast<char*>(flat_voxels.data()), sizeof(GaussianVoxelData) * num_voxels);

  voxelmap->voxels.reserve(flat_voxels.size() * 2);
  for (const auto& voxel : flat_voxels) {
    voxelmap->voxels.emplace(voxel.uncompact());
  }

  return voxelmap;
}

}  // namespace gtsam_ext
