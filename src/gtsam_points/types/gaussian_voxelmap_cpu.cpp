// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

#include <memory>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>

namespace gtsam_points {

template class IncrementalVoxelMap<GaussianVoxel>;

void GaussianVoxel::add(const Setting& setting, const PointCloud& points, size_t i) {
  if (finalized) {
    this->finalized = false;
    this->mean *= num_points;
    this->cov *= num_points;
  }

  num_points++;
  this->mean += points.points[i];
  this->cov += points.covs[i];
}

void GaussianVoxel::finalize() {
  if (finalized) {
    return;
  }

  mean /= num_points;
  cov /= num_points;
  finalized = true;
}

GaussianVoxelMapCPU::GaussianVoxelMapCPU(double resolution) : IncrementalVoxelMap<GaussianVoxel>(resolution) {
  offsets = neighbor_offsets(1);
}

GaussianVoxelMapCPU::~GaussianVoxelMapCPU() {}

double GaussianVoxelMapCPU::voxel_resolution() const {
  return leaf_size();
}

Eigen::Vector3i GaussianVoxelMapCPU::voxel_coord(const Eigen::Vector4d& x) const {
  return fast_floor(x * inv_leaf_size).head<3>();
}

int GaussianVoxelMapCPU::lookup_voxel_index(const Eigen::Vector3i& coord) const {
  auto found = voxels.find(coord);
  if (found == voxels.end()) {
    return -1;
  }
  return found->second;
}

const GaussianVoxel& GaussianVoxelMapCPU::lookup_voxel(int voxel_id) const {
  return flat_voxels[voxel_id]->second;
}

void GaussianVoxelMapCPU::insert(const PointCloud& frame) {
  IncrementalVoxelMap<GaussianVoxel>::insert(frame);
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

  std::shared_ptr<std::pair<VoxelInfo, GaussianVoxel>> uncompact() const {
    auto uncompacted = std::make_shared<std::pair<VoxelInfo, GaussianVoxel>>();
    uncompacted->first.lru = 0;
    uncompacted->first.coord = coord;

    auto& voxel = uncompacted->second;
    voxel.finalized = true;
    voxel.num_points = num_points;
    voxel.mean << mean.cast<double>(), 1.0;

    voxel.cov(0, 0) = cov[0];
    voxel.cov(0, 1) = voxel.cov(1, 0) = cov[1];
    voxel.cov(0, 2) = voxel.cov(2, 0) = cov[2];
    voxel.cov(1, 1) = cov[3];
    voxel.cov(1, 2) = voxel.cov(2, 1) = cov[4];
    voxel.cov(2, 2) = cov[5];

    return uncompacted;
  }

public:
  Eigen::Vector3i coord;
  int num_points;
  Eigen::Vector3f mean;
  Eigen::Matrix<float, 6, 1> cov;
};

}  // namespace

void GaussianVoxelMapCPU::save_compact(const std::string& path) const {
  std::vector<GaussianVoxelData> serial_voxels(flat_voxels.size());
  std::transform(flat_voxels.begin(), flat_voxels.end(), serial_voxels.begin(), [](const auto& voxel) {
    return GaussianVoxelData(voxel->first.coord, voxel->second);
  });

  std::ofstream ofs(path);
  ofs << "compact " << 1 << std::endl;
  ofs << "resolution " << voxel_resolution() << std::endl;
  ofs << "lru_count " << lru_counter << std::endl;
  ofs << "lru_cycle " << lru_clear_cycle << std::endl;
  ofs << "lru_thresh " << lru_horizon << std::endl;
  ofs << "voxel_bytes " << sizeof(GaussianVoxelData) << std::endl;
  ofs << "num_voxels " << serial_voxels.size() << std::endl;

  ofs.write(reinterpret_cast<const char*>(serial_voxels.data()), sizeof(GaussianVoxelData) * serial_voxels.size());
}

GaussianVoxelMapCPU::Ptr GaussianVoxelMapCPU::load(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "error: failed to open " << path << std::endl;
    return nullptr;
  }

  auto voxelmap = std::make_shared<GaussianVoxelMapCPU>(1.0);

  std::string token;
  double resolution;
  bool compact;
  int voxel_bytes;
  int num_voxels;

  ifs >> token >> compact;
  ifs >> token >> resolution;
  ifs >> token >> voxelmap->lru_counter;
  ifs >> token >> voxelmap->lru_clear_cycle;
  ifs >> token >> voxelmap->lru_horizon;
  ifs >> token >> voxel_bytes;
  ifs >> token >> num_voxels;
  std::getline(ifs, token);

  std::vector<GaussianVoxelData> flat_voxels(num_voxels);
  ifs.read(reinterpret_cast<char*>(flat_voxels.data()), sizeof(GaussianVoxelData) * num_voxels);

  voxelmap->set_voxel_resolution(resolution);
  voxelmap->flat_voxels.reserve(flat_voxels.size() * 2);
  for (const auto& voxel : flat_voxels) {
    voxelmap->flat_voxels.emplace_back(voxel.uncompact());
    voxelmap->voxels[voxel.coord] = voxelmap->flat_voxels.size() - 1;
  }

  return voxelmap;
}

}  // namespace gtsam_points
