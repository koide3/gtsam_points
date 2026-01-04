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
#include <gtsam_points/types/gaussian_voxel_data.hpp>
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

  if (frame::has_intensities(points)) {
    this->intensity = std::max(this->intensity, frame::intensity(points, i));
  }
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
