// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)
#include "gtsam_points_python.hpp"

#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/kdtreex.hpp>

using namespace gtsam_points;

namespace {

/// @brief Run batched knn search on a NearestNeighborSearch. Queries are given as [M, D] or [D].
///        Missing neighbors are filled with index=-1 and sq_dist=inf.
std::pair<py::array_t<std::int64_t>, py::array_t<double>>
batch_knn_search(const NearestNeighborSearch& search, const DoubleArray& queries, int k, int dim, double max_sq_dist) {
  if (k <= 0) {
    throw std::invalid_argument("k must be positive");
  }
  if (queries.ndim() != 1 && queries.ndim() != 2) {
    throw std::invalid_argument("queries must be [M, D] or [D] (ndim=" + std::to_string(queries.ndim()) + ")");
  }

  const bool single_query = queries.ndim() == 1;
  const py::ssize_t cols = single_query ? queries.shape(0) : queries.shape(1);
  const py::ssize_t num_queries = single_query ? 1 : queries.shape(0);

  // Points are 3D but query vectors must be homogeneous 4D internally
  const bool homogeneous = dim == 4;
  if (homogeneous ? (cols != 3 && cols != 4) : (cols != dim)) {
    throw std::invalid_argument("query dimension mismatch (cols=" + std::to_string(cols) + ")");
  }

  py::array_t<std::int64_t> indices({num_queries, static_cast<py::ssize_t>(k)});
  py::array_t<double> sq_dists({num_queries, static_cast<py::ssize_t>(k)});
  auto indices_view = indices.mutable_unchecked<2>();
  auto sq_dists_view = sq_dists.mutable_unchecked<2>();
  const double* data = queries.data();

  {
    py::gil_scoped_release release;
    std::vector<size_t> k_indices(k);
    std::vector<double> k_sq_dists(k);
    std::vector<double> query(homogeneous ? 4 : cols);

    for (py::ssize_t i = 0; i < num_queries; i++) {
      const double* q = data + i * cols;
      std::copy(q, q + cols, query.begin());
      if (homogeneous) {
        query[3] = 1.0;
      }

      const size_t num_found = search.knn_search(query.data(), k, k_indices.data(), k_sq_dists.data(), max_sq_dist);
      for (int j = 0; j < k; j++) {
        indices_view(i, j) = j < static_cast<int>(num_found) ? static_cast<std::int64_t>(k_indices[j]) : -1;
        sq_dists_view(i, j) = j < static_cast<int>(num_found) ? k_sq_dists[j] : std::numeric_limits<double>::infinity();
      }
    }
  }

  return {indices, sq_dists};
}

}  // namespace

// KdTree over feature vectors [N, D]. Holds a copy of the features.
struct KdTreeXWrapper : public NearestNeighborSearch {
  KdTreeXWrapper(std::vector<Eigen::VectorXd>&& features_) : features(std::move(features_)) {
    if (features.empty()) {
      throw std::invalid_argument("features must not be empty");
    }
    index = std::make_unique<KdTreeX<-1>>(features.data(), features.size());
  }

  size_t knn_search(const double* pt, size_t k, size_t* k_indices, double* k_sq_dists, double max_sq_dist) const override {
    return index->knn_search(pt, k, k_indices, k_sq_dists, max_sq_dist);
  }

  size_t radius_search(const double* pt, double radius, std::vector<size_t>& indices, std::vector<double>& sq_dists, int max_num_neighbors)
    const override {
    return index->radius_search(pt, radius, indices, sq_dists, max_num_neighbors);
  }

  std::vector<Eigen::VectorXd> features;
  std::unique_ptr<KdTreeX<-1>> index;
};

void define_ann(py::module_& m) {
  // gtsam_points::NearestNeighborSearch
  py::class_<NearestNeighborSearch, std::shared_ptr<NearestNeighborSearch>>(m, "NearestNeighborSearch", "Nearest neighbor search interface");

  // gtsam_points::KdTree
  py::class_<KdTree, NearestNeighborSearch, std::shared_ptr<KdTree>>(m, "KdTree", "KdTree-based nearest neighbor search")
    .def(
      py::init([](const PointCloud::ConstPtr& points, int build_num_threads) {
        if (!points->has_points()) {
          throw std::invalid_argument("points must have point coordinates");
        }
        py::gil_scoped_release release;
        return std::make_shared<KdTree>(points->points, points->size(), build_num_threads);
      }),
      py::arg("points"),
      py::arg("build_num_threads") = 1,
      py::keep_alive<1, 2>(),  // KdTree refers to the point data owned by the input point cloud
      "Create a KdTree for a point cloud")
    .def("__repr__", [](const KdTree& tree) { return "<gtsam_points.KdTree size=" + std::to_string(tree.num_points) + ">"; })
    .def("__len__", [](const KdTree& tree) { return tree.num_points; })
    .def(
      "knn_search",
      [](const KdTree& self, const DoubleArray& queries, int k, double max_sq_dist) { return batch_knn_search(self, queries, k, 4, max_sq_dist); },
      py::arg("queries"),
      py::arg("k"),
      py::arg("max_sq_dist") = std::numeric_limits<double>::max(),
      "Find k nearest neighbors for queries [M, 3] or [3]. Returns (indices [M, k], sq_dists [M, k]). Missing neighbors are filled with -1 / inf.")
    .def(
      "radius_search",
      [](const KdTree& self, const DoubleArray& query, double radius, int max_num_neighbors) {
        if (query.ndim() != 1 || (query.shape(0) != 3 && query.shape(0) != 4)) {
          throw std::invalid_argument("query must be a single point [3]");
        }
        const Eigen::Vector4d pt(query.at(0), query.at(1), query.at(2), 1.0);

        std::vector<size_t> indices;
        std::vector<double> sq_dists;
        {
          py::gil_scoped_release release;
          self.radius_search(pt.data(), radius, indices, sq_dists, max_num_neighbors);
        }
        return std::make_pair(convert_indices(indices), py::array_t<double>(sq_dists.size(), sq_dists.data()));
      },
      py::arg("query"),
      py::arg("radius"),
      py::arg("max_num_neighbors") = std::numeric_limits<int>::max(),
      "Find neighbors within a radius of a query point [3]. Returns (indices [n], sq_dists [n]).");

  // KdTree for feature vectors
  py::class_<KdTreeXWrapper, NearestNeighborSearch, std::shared_ptr<KdTreeXWrapper>>(m, "KdTreeX", "KdTree-based nearest neighbor search for feature vectors")
    .def(
      py::init([](const DoubleArray& features) {
        auto converted = convert_features(features);
        py::gil_scoped_release release;
        return std::make_shared<KdTreeXWrapper>(std::move(converted));
      }),
      py::arg("features"),
      "Create a KdTree for feature vectors [N, D] (a copy of the features is held internally)")
    .def("__repr__", [](const KdTreeXWrapper& tree) {
      return "<gtsam_points.KdTreeX size=" + std::to_string(tree.features.size()) + " dim=" + std::to_string(tree.features.front().size()) + ">";
    })
    .def("__len__", [](const KdTreeXWrapper& tree) { return tree.features.size(); })
    .def(
      "knn_search",
      [](const KdTreeXWrapper& self, const DoubleArray& queries, int k, double max_sq_dist) {
        return batch_knn_search(self, queries, k, self.features.front().size(), max_sq_dist);
      },
      py::arg("queries"),
      py::arg("k"),
      py::arg("max_sq_dist") = std::numeric_limits<double>::max(),
      "Find k nearest neighbors for query features [M, D] or [D]. Returns (indices [M, k], sq_dists [M, k]).");
}
