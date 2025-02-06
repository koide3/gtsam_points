// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_loam_factor.hpp>

#include <gtsam/geometry/SO3.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/factors/impl/scan_matching_reduction.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

template <typename TargetFrame, typename SourceFrame>
IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>::IntegratedPointToPlaneFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree)
: gtsam_points::IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source) {
  //
  if (!frame::has_points(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for loam" << std::endl;
    abort();
  }

  if (!frame::has_points(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for loam" << std::endl;
    abort();
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    this->target_tree.reset(new KdTree2<TargetFrame>(target));
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>::IntegratedPointToPlaneFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source)
: IntegratedPointToPlaneFactor_(target_key, source_key, target, source, nullptr) {}

template <typename TargetFrame, typename SourceFrame>
IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>::~IntegratedPointToPlaneFactor_() {}

template <typename TargetFrame, typename SourceFrame>
void IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedPointToPlaneFactor";
  if (is_binary) {
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;
  } else {
    std::cout << "(fixed, " << keyFormatter(this->keys()[0]) << ")" << std::endl;
  }

  std::cout << "|target|=" << frame::size(*target) << "pts, |source|=" << frame::size(*source) << "pts" << std::endl;
  std::cout << "num_threads=" << num_threads << ", max_corr_dist=" << std::sqrt(max_correspondence_distance_sq) << std::endl;
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>::update_correspondences(const Eigen::Isometry3d& delta) const {
  if (correspondences.size() == frame::size(*source) && (correspondence_update_tolerance_trans > 0.0 || correspondence_update_tolerance_rot > 0.0)) {
    Eigen::Isometry3d diff = delta.inverse() * last_correspondence_point;
    double diff_rot = Eigen::AngleAxisd(diff.linear()).angle();
    double diff_trans = diff.translation().norm();
    if (diff_rot < correspondence_update_tolerance_rot && diff_trans < correspondence_update_tolerance_trans) {
      return;
    }
  }

  last_correspondence_point = delta;
  correspondences.resize(frame::size(*source));

  const auto perpoint_task = [&](int i) {
    Eigen::Vector4d pt = delta * frame::point(*source, i);

    std::array<size_t, 3> k_indices;
    std::array<double, 3> k_sq_dists;
    size_t num_found = target_tree->knn_search(pt.data(), 3, k_indices.data(), k_sq_dists.data(), max_correspondence_distance_sq);

    if (num_found < 3 || k_sq_dists.back() > max_correspondence_distance_sq) {
      correspondences[i] = std::make_tuple(-1, -1, -1);
    } else {
      correspondences[i] = std::make_tuple(k_indices[0], k_indices[1], k_indices[2]);
    }
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < frame::size(*source); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame::size(*source), 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }
}

template <typename TargetFrame, typename SourceFrame>
double IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //
  if (correspondences.size() != frame::size(*source)) {
    update_correspondences(delta);
  }

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_target,
                               Eigen::Matrix<double, 6, 6>* H_source,
                               Eigen::Matrix<double, 6, 6>* H_target_source,
                               Eigen::Matrix<double, 6, 1>* b_target,
                               Eigen::Matrix<double, 6, 1>* b_source) {
    auto target_indices = correspondences[i];
    if (std::get<0>(target_indices) < 0) {
      return 0.0;
    }

    const auto& x_i = frame::point(*source, i);
    const auto& x_j = frame::point(*target, std::get<0>(target_indices));
    const auto& x_l = frame::point(*target, std::get<1>(target_indices));
    const auto& x_m = frame::point(*target, std::get<2>(target_indices));

    Eigen::Vector4d transed_x_i = delta * x_i;
    Eigen::Vector4d normal = Eigen::Vector4d::Zero();
    normal.head<3>() = (x_j - x_l).template head<3>().cross((x_j - x_m).template head<3>());
    normal = normal / normal.norm();

    Eigen::Vector4d residual = x_j - transed_x_i;
    Eigen::Vector4d plane_residual = residual.array() * normal.array();
    const double error = plane_residual.transpose() * plane_residual;

    if (H_target == nullptr) {
      return error;
    }

    Eigen::Matrix<double, 4, 6> J_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_target.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_x_i.head<3>());
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_source.block<3, 3>(0, 0) = delta.linear() * gtsam::SO3::Hat(x_i.template head<3>());
    J_source.block<3, 3>(0, 3) = -delta.linear();

    J_target = normal.asDiagonal() * J_target;
    J_source = normal.asDiagonal() * J_source;

    *H_target += J_target.transpose() * J_target;
    *H_source += J_source.transpose() * J_source;
    *H_target_source += J_target.transpose() * J_source;
    *b_target += J_target.transpose() * plane_residual;
    *b_source += J_source.transpose() * plane_residual;

    return error;
  };

  if (is_omp_default() || num_threads == 1) {
    return scan_matching_reduce_omp(perpoint_task, frame::size(*source), num_threads, H_target, H_source, H_target_source, b_target, b_source);
  } else {
    return scan_matching_reduce_tbb(perpoint_task, frame::size(*source), H_target, H_source, H_target_source, b_target, b_source);
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>::IntegratedPointToEdgeFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree)
: gtsam_points::IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source) {
  //
  if (!frame::has_points(*target) || !frame::has_points(*source)) {
    std::cerr << "error: target or source points has not been allocated!!" << std::endl;
    abort();
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    this->target_tree.reset(new KdTree2<TargetFrame>(target));
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>::IntegratedPointToEdgeFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source)
: gtsam_points::IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>(target_key, source_key, target, source, nullptr) {}

template <typename TargetFrame, typename SourceFrame>
IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>::~IntegratedPointToEdgeFactor_() {}

template <typename TargetFrame, typename SourceFrame>
void IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>::update_correspondences(const Eigen::Isometry3d& delta) const {
  if (correspondences.size() == frame::size(*source) && (correspondence_update_tolerance_trans > 0.0 || correspondence_update_tolerance_rot > 0.0)) {
    Eigen::Isometry3d diff = delta.inverse() * last_correspondence_point;
    double diff_rot = Eigen::AngleAxisd(diff.linear()).angle();
    double diff_trans = diff.translation().norm();
    if (diff_rot < correspondence_update_tolerance_rot && diff_trans < correspondence_update_tolerance_trans) {
      return;
    }
  }

  correspondences.resize(frame::size(*source));
  last_correspondence_point = delta;

  const auto perpoint_task = [&](int i) {
    Eigen::Vector4d pt = delta * frame::point(*source, i);

    std::array<size_t, 2> k_indices;
    std::array<double, 2> k_sq_dists;
    size_t num_found = target_tree->knn_search(pt.data(), 2, k_indices.data(), k_sq_dists.data(), max_correspondence_distance_sq);

    if (num_found < 2 || k_sq_dists.back() > max_correspondence_distance_sq) {
      correspondences[i] = std::make_tuple(-1, -1);
    } else {
      correspondences[i] = std::make_tuple(k_indices[0], k_indices[1]);
    }
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
    for (int i = 0; i < frame::size(*source); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame::size(*source), 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedPointToEdgeFactor";
  if (is_binary) {
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;
  } else {
    std::cout << "(fixed, " << keyFormatter(this->keys()[0]) << ")" << std::endl;
  }

  std::cout << "|target|=" << frame::size(*target) << "pts, |source|=" << frame::size(*source) << "pts" << std::endl;
  std::cout << "num_threads=" << num_threads << ", max_corr_dist=" << std::sqrt(max_correspondence_distance_sq) << std::endl;
}

template <typename TargetFrame, typename SourceFrame>
double IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //
  if (correspondences.size() != frame::size(*source)) {
    update_correspondences(delta);
  }

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_target,
                               Eigen::Matrix<double, 6, 6>* H_source,
                               Eigen::Matrix<double, 6, 6>* H_target_source,
                               Eigen::Matrix<double, 6, 1>* b_target,
                               Eigen::Matrix<double, 6, 1>* b_source) {
    auto target_indices = correspondences[i];
    if (std::get<0>(target_indices) < 0) {
      return 0.0;
    }

    const auto& x_i = frame::point(*source, i);
    const auto& x_j = frame::point(*target, std::get<0>(target_indices));
    const auto& x_l = frame::point(*target, std::get<1>(target_indices));

    Eigen::Vector4d transed_x_i = delta * x_i;

    double c_inv = 1.0 / (x_j - x_l).norm();
    Eigen::Vector4d x_ij = transed_x_i - x_j;
    Eigen::Vector4d x_il = transed_x_i - x_l;

    Eigen::Vector4d residual = Eigen::Vector4d::Zero();
    residual.head<3>() = x_ij.head<3>().cross(x_il.head<3>()) * c_inv;
    const double error = residual.dot(residual);

    if (H_target == nullptr) {
      return error;
    }

    Eigen::Matrix<double, 4, 6> J_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_target.block<3, 3>(0, 0) = -gtsam::SO3::Hat(transed_x_i.template head<3>());
    J_target.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_source.block<3, 3>(0, 0) = delta.linear() * gtsam::SO3::Hat(x_i.template head<3>());
    J_source.block<3, 3>(0, 3) = -delta.linear();

    Eigen::Matrix<double, 4, 4> J_e = Eigen::Matrix4d::Zero();
    J_e.block<1, 3>(0, 0) << 0.0, -x_j[2] + x_l[2], x_j[1] - x_l[1];
    J_e.block<1, 3>(1, 0) << x_j[2] - x_l[2], 0.0, -x_j[0] + x_l[0];
    J_e.block<1, 3>(2, 0) << -x_j[1] + x_l[1], x_j[0] - x_l[0], 0.0;

    J_target = c_inv * J_e * J_target;
    J_source = c_inv * J_e * J_source;

    *H_target += J_target.transpose() * J_target;
    *H_source += J_source.transpose() * J_source;
    *H_target_source += J_target.transpose() * J_source;
    *b_target += J_target.transpose() * residual;
    *b_source += J_source.transpose() * residual;

    return error;
  };

  if (is_omp_default() || num_threads == 1) {
    return scan_matching_reduce_omp(perpoint_task, frame::size(*source), num_threads, H_target, H_source, H_target_source, b_target, b_source);
  } else {
    return scan_matching_reduce_tbb(perpoint_task, frame::size(*source), H_target, H_source, H_target_source, b_target, b_source);
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedLOAMFactor_<TargetFrame, SourceFrame>::IntegratedLOAMFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target_edges,
  const std::shared_ptr<const TargetFrame>& target_planes,
  const std::shared_ptr<const SourceFrame>& source_edges,
  const std::shared_ptr<const SourceFrame>& source_planes,
  const std::shared_ptr<const NearestNeighborSearch>& target_edges_tree,
  const std::shared_ptr<const NearestNeighborSearch>& target_planes_tree)
: gtsam_points::IntegratedMatchingCostFactor(target_key, source_key),
  enable_correspondence_validation(false) {
  //
  edge_factor.reset(
    new IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>(target_key, source_key, target_edges, source_edges, target_edges_tree));
  plane_factor.reset(
    new IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>(target_key, source_key, target_planes, source_planes, target_planes_tree));
}

template <typename TargetFrame, typename SourceFrame>
IntegratedLOAMFactor_<TargetFrame, SourceFrame>::IntegratedLOAMFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target_edges,
  const std::shared_ptr<const TargetFrame>& target_planes,
  const std::shared_ptr<const SourceFrame>& source_edges,
  const std::shared_ptr<const SourceFrame>& source_planes)
: gtsam_points::IntegratedMatchingCostFactor(target_key, source_key),
  enable_correspondence_validation(false) {
  //
  edge_factor.reset(new IntegratedPointToEdgeFactor_<TargetFrame, SourceFrame>(target_key, source_key, target_edges, source_edges));
  plane_factor.reset(new IntegratedPointToPlaneFactor_<TargetFrame, SourceFrame>(target_key, source_key, target_planes, source_planes));
}

template <typename TargetFrame, typename SourceFrame>
IntegratedLOAMFactor_<TargetFrame, SourceFrame>::~IntegratedLOAMFactor_() {}

template <typename TargetFrame, typename SourceFrame>
void IntegratedLOAMFactor_<TargetFrame, SourceFrame>::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedLOAMFactor";
  if (is_binary) {
    std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;
  } else {
    std::cout << "(fixed, " << keyFormatter(this->keys()[0]) << ")" << std::endl;
  }
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedLOAMFactor_<TargetFrame, SourceFrame>::set_num_threads(int n) {
  edge_factor->set_num_threads(n);
  plane_factor->set_num_threads(n);
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedLOAMFactor_<TargetFrame, SourceFrame>::set_max_correspondence_distance(double dist_edge, double dist_plane) {
  edge_factor->set_max_correspondence_distance(dist_edge);
  plane_factor->set_max_correspondence_distance(dist_plane);
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedLOAMFactor_<TargetFrame, SourceFrame>::set_enable_correspondence_validation(bool enable) {
  enable_correspondence_validation = enable;
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedLOAMFactor_<TargetFrame, SourceFrame>::update_correspondences(const Eigen::Isometry3d& delta) const {
  edge_factor->update_correspondences(delta);
  plane_factor->update_correspondences(delta);

  validate_correspondences();
}

template <typename TargetFrame, typename SourceFrame>
double IntegratedLOAMFactor_<TargetFrame, SourceFrame>::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //

  double error = edge_factor->evaluate(delta, H_target, H_source, H_target_source, b_target, b_source);

  if (H_target && H_source && H_target_source && b_target && b_source) {
    Eigen::Matrix<double, 6, 6> H_t, H_s, H_ts;
    Eigen::Matrix<double, 6, 1> b_t, b_s;
    error += plane_factor->evaluate(delta, &H_t, &H_s, &H_ts, &b_t, &b_s);

    (*H_target) += H_t;
    (*H_source) += H_s;
    (*H_target_source) += H_ts;
    (*b_target) += b_t;
    (*b_source) += b_s;
  } else {
    error += plane_factor->evaluate(delta);
  }

  return error;
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedLOAMFactor_<TargetFrame, SourceFrame>::set_correspondence_update_tolerance(double angle, double trans) {
  plane_factor->set_correspondence_update_tolerance(angle, trans);
  edge_factor->set_correspondence_update_tolerance(angle, trans);
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedLOAMFactor_<TargetFrame, SourceFrame>::validate_correspondences() const {
  if (!enable_correspondence_validation) {
    return;
  }

  // Validate edge correspondences
  for (auto& corr : edge_factor->correspondences) {
    if (std::get<0>(corr) < 0) {
      continue;
    }

    const auto& pt1 = frame::point(*edge_factor->target, std::get<0>(corr));
    const auto& pt2 = frame::point(*edge_factor->target, std::get<1>(corr));

    const double v_theta1 = std::atan2(pt1.z(), pt1.template head<2>().norm());
    const double v_theta2 = std::atan2(pt2.z(), pt2.template head<2>().norm());

    // Reject pairs in a same scan line
    if (std::abs(v_theta1 - v_theta2) < 0.1 * M_PI / 180.0) {
      corr = std::make_tuple(-1, -1);
    }
  }

  // Validate plane correspondences
  for (auto& corr : plane_factor->correspondences) {
    if (std::get<0>(corr) < 0) {
      continue;
    }

    const auto& pt1 = frame::point(*plane_factor->target, std::get<0>(corr));
    const auto& pt2 = frame::point(*plane_factor->target, std::get<1>(corr));
    const auto& pt3 = frame::point(*plane_factor->target, std::get<2>(corr));

    const double v_theta1 = std::atan2(pt1.z(), pt1.template head<2>().norm());
    const double v_theta2 = std::atan2(pt2.z(), pt2.template head<2>().norm());
    const double v_theta3 = std::atan2(pt3.z(), pt3.template head<2>().norm());

    // Reject pairs in a same scan line
    if (std::abs(v_theta1 - v_theta2) < 0.1 * M_PI / 180.0 && std::abs(v_theta1 - v_theta3) < 0.1 * M_PI * 180.0) {
      corr = std::make_tuple(-1, -1, -1);
    }
  }
}

}  // namespace gtsam_points