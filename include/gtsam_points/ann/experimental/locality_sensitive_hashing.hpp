// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <random>
#include <algorithm>
#include <Eigen/Core>
#include <boost/functional/hash.hpp>
#include <gtsam_points/ann/knn_result.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

namespace gtsam_points {

template <typename Feature>
class LocalitySensitiveHashing : public NearestNeighborSearch {
public:
  using Ptr = std::shared_ptr<LocalitySensitiveHashing>;
  using ConstPtr = std::shared_ptr<const LocalitySensitiveHashing>;

  LocalitySensitiveHashing(
    const std::vector<Feature>& features,
    const Eigen::ArrayXd& offset,
    const Eigen::ArrayXd& scale,
    size_t num_buckets,
    size_t max_items_in_bucket,
    std::mt19937& mt)
  : features(features),
    offset(offset),
    scale(scale),
    num_buckets(num_buckets),
    max_items_in_bucket(max_items_in_bucket),
    buckets(num_buckets * max_items_in_bucket, std::numeric_limits<std::uint32_t>::max()) {
    //
    std::normal_distribution<> ndist(0.0, 0.1);
    for (size_t i = 0; i < features.size(); i++) {
      Eigen::ArrayXd noise(offset.size());
      std::generate(noise.data(), noise.data() + noise.size(), [&]() { return ndist(mt); });
      const Eigen::ArrayXi nf = (features[i].array() * scale + offset + noise).floor().template cast<int>();
      const auto hash = calc_hash(nf);

      const size_t bucket_index = hash % num_buckets;
      const auto bucket = buckets.data() + bucket_index * max_items_in_bucket;

      size_t insert_loc = 0;
      for (size_t j = 0; j < max_items_in_bucket; j++) {
        if (bucket[j] == std::numeric_limits<std::uint32_t>::max()) {
          insert_loc = i;
          break;
        }
      }

      if (insert_loc < max_items_in_bucket) {
        bucket[insert_loc] = i;
      } else {
        insert_loc = std::uniform_int_distribution<>(0, max_items_in_bucket)(mt);
        if (insert_loc < max_items_in_bucket) {
          bucket[insert_loc] = i;
        }
      }
    }
  }
  ~LocalitySensitiveHashing() {}

  size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist = std::numeric_limits<double>::max())
    const {
    const Eigen::ArrayXi nf = (Eigen::Map<const Eigen::ArrayXd>(pt, offset.size()) * scale + offset).floor().template cast<int>();
    const auto hash = calc_hash(nf);

    const size_t bucket_index = hash % num_buckets;
    const auto bucket = buckets.data() + bucket_index * max_items_in_bucket;

    KnnResult<-1> result(k_indices, k_sq_dists, k);

    for (size_t i = 0; i < max_items_in_bucket; i++) {
      if (bucket[i] == std::numeric_limits<std::uint32_t>::max()) {
        break;
      }

      const auto index = bucket[i];
      const auto& f = features[index];
      const double sq_dist = (f - Eigen::Map<const Eigen::VectorXd>(pt, offset.size())).squaredNorm();

      result.push(index, sq_dist);
    }

    return result.num_found();
  }

private:
  std::uint64_t calc_hash(const Eigen::ArrayXi& nf) const {
    std::uint64_t h = 0;
    for (int i = 0; i < nf.size(); i++) {
      boost::hash_combine(h, nf[i]);
    }
    return h;
  }

private:
  const std::vector<Feature>& features;
  const Eigen::ArrayXd offset;
  const Eigen::ArrayXd scale;
  const size_t max_items_in_bucket;
  const size_t num_buckets;
  std::vector<std::uint32_t> buckets;
};

class MultiProbeLocalitySensitiveHashing : public NearestNeighborSearch {
public:
  using Ptr = std::shared_ptr<MultiProbeLocalitySensitiveHashing>;
  using ConstPtr = std::shared_ptr<const MultiProbeLocalitySensitiveHashing>;

  MultiProbeLocalitySensitiveHashing() : num_probes(5), num_buckets(8192 * 16), max_items_in_bucket(32) {}
  ~MultiProbeLocalitySensitiveHashing() {}

  template <typename Feature>
  Eigen::ArrayXd calc_base_scale(const std::vector<Feature>& features, double div = 100.0, int max_scans = 5000) {
    const int step = features.size() < max_scans ? 1 : features.size() / max_scans;
    Eigen::ArrayXd min_f = features[0].array();
    Eigen::ArrayXd max_f = features[0].array();
    for (size_t i = 1; i < features.size(); i += step) {
      min_f = min_f.min(features[i].array());
      max_f = max_f.max(features[i].array());
    }
    return (max_f - min_f) / div;
  }

  template <typename Feature>
  void create_tables(const std::vector<Feature>& features, std::mt19937& mt) {
    // base_scale = calc_base_scale(features, 100.0, 5000);
    base_scale = Eigen::ArrayXd::Ones(features[0].size()) * 10.0;

    lshs.clear();
    std::uniform_real_distribution<> udist(0.0, 1.0);

    for (size_t i = 0; i < num_probes; i++) {
      Eigen::ArrayXd offset(base_scale.size());
      std::generate(offset.data(), offset.data() + offset.size(), [&]() { return udist(mt); });
      Eigen::ArrayXd scale = base_scale;
      lshs.push_back(std::make_shared<LocalitySensitiveHashing<Feature>>(features, offset, scale, num_buckets, max_items_in_bucket, mt));
    }
  }

  size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist = std::numeric_limits<double>::max())
    const override {
    KnnResult<-1> result(k_indices, k_sq_dists, k);

    for (const auto& lsh : lshs) {
      size_t num_found = lsh->knn_search(pt, k, k_indices, k_sq_dists, max_sq_dist);
      if (num_found == 0) {
        continue;
      }

      for (size_t i = 0; i < num_found; i++) {
        result.push(k_indices[i], k_sq_dists[i]);
      }
    }

    return result.num_found();
  }

private:
  size_t num_probes;
  size_t num_buckets;
  size_t max_items_in_bucket;

  Eigen::ArrayXd base_scale;

  std::vector<NearestNeighborSearch::Ptr> lshs;
};

}  // namespace gtsam_points