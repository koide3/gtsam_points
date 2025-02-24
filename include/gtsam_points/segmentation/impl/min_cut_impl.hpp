// SPDX-License-Identifier: MIT
// Copyright (c) 2025  Kenji Koide (k.koide@aist.go.jp)

// The min-cut code is inspired by the following PCL code:
// https://github.com/PointCloudLibrary/pcl/blob/master/segmentation/include/pcl/segmentation/impl/min_cut_segmentation.hpp

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id:$
 *
 */

#include <gtsam_points/types/frame_traits.hpp>
#include <gtsam_points/segmentation/min_cut.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/property_map/property_map.hpp>

namespace gtsam_points {

template <typename PointCloud>
MinCutResult min_cut_(const PointCloud& points, const NearestNeighborSearch& search, const size_t source_pt_index, const MinCutParams& params) {
  // Find k-nearest neighbors for all points
  std::vector<size_t> all_neighbors(frame::size(points) * params.k_neighbors);
  std::vector<double> all_neighbor_sq_dists(frame::size(points) * params.k_neighbors);
#pragma omp parallel for num_threads(params.num_threads) schedule(guided, 4)
  for (size_t i = 0; i < frame::size(points); i++) {
    size_t* k_neighbors = all_neighbors.data() + i * params.k_neighbors;
    double* k_sq_dists = all_neighbor_sq_dists.data() + i * params.k_neighbors;
    search.knn_search(frame::point(points, i).data(), params.k_neighbors, k_neighbors, k_sq_dists);
  }

  // clang-format off
  using Traits = boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS>;
  using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, ///
    boost::property<boost::vertex_index_t, long,//
    boost::property<boost::vertex_color_t, boost::default_color_type,//
    boost::property<boost::vertex_distance_t, long,//
    boost::property<boost::vertex_predecessor_t, Traits::edge_descriptor>>>>, //
    boost::property<boost::edge_capacity_t, double,//
    boost::property<boost::edge_residual_capacity_t, double,//
    boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>>; 
  using EdgeDescriptor = Traits::edge_descriptor;
  // clang-format on

  // Graph construction
  Graph g(frame::size(points));
  boost::property_map<Graph, boost::edge_capacity_t>::type capacity = get(boost::edge_capacity, g);
  boost::property_map<Graph, boost::edge_residual_capacity_t>::type residual_capacity = get(boost::edge_residual_capacity, g);
  boost::property_map<Graph, boost::edge_reverse_t>::type reverse_edges = get(boost::edge_reverse, g);
  boost::property_map<Graph, boost::vertex_color_t>::type color = get(boost::vertex_color, g);

  const auto add_edge = [&](size_t source, size_t target, double weight) {
    const auto edge = boost::add_edge(source, target, g);
    const auto reverse_edge = boost::add_edge(target, source, g);
    if (!edge.second || !reverse_edge.second) {
      return;
    }

    capacity[edge.first] = weight;
    capacity[reverse_edge.first] = 0.0;
    reverse_edges[edge.first] = reverse_edge.first;
    reverse_edges[reverse_edge.first] = edge.first;
  };

  const double inv_dist_sq_sigma = 1.0 / (params.distance_sigma * params.distance_sigma);
  const double inv_angle_sq_sigma = 1.0 / (params.angle_sigma * params.angle_sigma);

  std::vector<size_t> sink_pt_indices;
  for (size_t i = 0; i < frame::size(points); i++) {
    const size_t* neighbors = all_neighbors.data() + i * params.k_neighbors;
    const double* neighbor_sq_dists = all_neighbor_sq_dists.data() + i * params.k_neighbors;

    const Eigen::Vector4d& normal = frame::normal(points, i);

    const double dist_from_source = (frame::point(points, i) - frame::point(points, source_pt_index)).norm();

    // Mark as foreground if the point is within the foreground mask radius
    if (i != source_pt_index && dist_from_source < params.foreground_mask_radius) {
      add_edge(i, source_pt_index, params.foreground_weight);
    }
    // Mark as background if the point is outside the background mask radius
    if (dist_from_source > params.background_mask_radius) {
      sink_pt_indices.emplace_back(i);
    }

    // Add connectivity edges to the neighbors
    for (size_t k = 0; k < params.k_neighbors; k++) {
      const double weight_dist = std::exp(-neighbor_sq_dists[k] * inv_dist_sq_sigma);
      const double normal_diff = std::acos(std::abs(normal.dot(frame::normal(points, neighbors[k]))));
      const double weight_angle = std::exp(-normal_diff * normal_diff * inv_angle_sq_sigma);
      add_edge(i, neighbors[k], weight_dist * weight_angle);
    }
  }

  if (sink_pt_indices.empty()) {
    std::cerr << "No sink points" << std::endl;
    return MinCutResult();
  }

  size_t sink_pt_index = sink_pt_indices.front();
  for (size_t i = 1; i < sink_pt_indices.size(); i++) {
    add_edge(sink_pt_indices[i], sink_pt_index, params.background_weight);
  }

  // Run the min-cut algorithm
  MinCutResult result;
  result.source_index = source_pt_index;
  result.sink_index = sink_pt_index;
  result.max_flow = boost::boykov_kolmogorov_max_flow(g, source_pt_index, sink_pt_index);

  // Extract the cluster
  const auto source_color = color[source_pt_index];
  for (size_t i = 0; i < frame::size(points); i++) {
    if (color[i] == source_color) {
      result.cluster_indices.emplace_back(i);
    }
  }

  return result;
}

template <typename PointCloud>
MinCutResult min_cut_(const PointCloud& points, const NearestNeighborSearch& search, const Eigen::Vector4d& source_pt, const MinCutParams& params) {
  size_t source_pt_index;
  double sq_dist;
  if (!search.knn_search(source_pt.data(), 1, &source_pt_index, &sq_dist)) {
    std::cerr << "Failed to find the source point" << std::endl;
    return MinCutResult();
  }

  if (sq_dist > params.foreground_mask_radius * params.foreground_mask_radius) {
    std::cerr << "warning: The source point is too far from the point cloud" << std::endl;
  }

  return min_cut_(points, search, source_pt_index, params);
}

}  // namespace gtsam_points
