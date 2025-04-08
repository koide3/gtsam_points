// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_ct_icp_factor.hpp>

#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree2.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/factors/impl/scan_matching_reduction.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace gtsam_points {

template <typename TargetFrame, typename SourceFrame>
IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::IntegratedCT_ICPFactor_(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree)
: gtsam::NonlinearFactor(gtsam::KeyVector{source_t0_key, source_t1_key}),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  target(target),
  source(source) {
  //
  if (!frame::has_points(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for ct_icp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_times(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for ct_icp" << std::endl;
    abort();
  }

  time_table.reserve(frame::size(*source) / 10);
  time_indices.reserve(frame::size(*source));

  const double time_eps = 1e-3;
  for (int i = 0; i < frame::size(*source); i++) {
    const double t = frame::time(*source, i);
    if (time_table.empty() || t - time_table.back() > time_eps) {
      time_table.push_back(t);
    }
    time_indices.push_back(time_table.size() - 1);
  }

  for (auto& t : time_table) {
    t = t / std::max(1e-9, time_table.back());
  }

  if (target_tree) {
    this->target_tree = target_tree;
  } else {
    this->target_tree.reset(new KdTree2<TargetFrame>(target));
  }
}

template <typename TargetFrame, typename SourceFrame>
IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::IntegratedCT_ICPFactor_(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source)
: IntegratedCT_ICPFactor_(source_t0_key, source_t1_key, target, source, nullptr) {}

template <typename TargetFrame, typename SourceFrame>
IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::~IntegratedCT_ICPFactor_() {}

template <typename TargetFrame, typename SourceFrame>
void IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
  std::cout << s << "IntegratedCT_ICPFactor";
  std::cout << "(" << keyFormatter(this->keys()[0]) << ", " << keyFormatter(this->keys()[1]) << ")" << std::endl;

  std::cout << "|target|=" << frame::size(*target) << "pts, |source|=" << frame::size(*source) << "pts" << std::endl;
  std::cout << "num_threads=" << num_threads << ", max_corr_dist=" << std::sqrt(max_correspondence_distance_sq) << std::endl;
}

template <typename TargetFrame, typename SourceFrame>
double IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::error(const gtsam::Values& values) const {
  update_poses(values);
  if (correspondences.size() != frame::size(*source)) {
    update_correspondences();
  }

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_target,
                               Eigen::Matrix<double, 6, 6>* H_source,
                               Eigen::Matrix<double, 6, 6>* H_target_source,
                               Eigen::Matrix<double, 6, 1>* b_target,
                               Eigen::Matrix<double, 6, 1>* b_source) {
    const long target_index = correspondences[i];
    if (target_index < 0) {
      return 0.0;
    }

    const int time_index = time_indices[i];
    const auto& pose = source_poses[time_index];

    const auto& source_pt = frame::point(*source, i);
    const auto& target_pt = frame::point(*target, target_index);
    const auto& target_normal = frame::normal(*target, target_index);

    gtsam::Point3 transed_source_pt = pose.transformFrom(source_pt.template head<3>().eval());
    gtsam::Point3 residual = transed_source_pt - target_pt.template head<3>();
    double error = gtsam::dot(residual, target_normal.template head<3>());

    return error * error;
  };

  if (is_omp_default() || num_threads == 1) {
    return scan_matching_reduce_omp(perpoint_task, frame::size(*this->source), this->num_threads, nullptr, nullptr, nullptr, nullptr, nullptr);
  } else {
    return scan_matching_reduce_tbb(perpoint_task, frame::size(*this->source), nullptr, nullptr, nullptr, nullptr, nullptr);
  }
}

template <typename TargetFrame, typename SourceFrame>
boost::shared_ptr<gtsam::GaussianFactor> IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::linearize(const gtsam::Values& values) const {
  if (!frame::has_normals(*target)) {
    std::cerr << "error: target cloud doesn't have normals!!" << std::endl;
    abort();
  }

  update_poses(values);
  update_correspondences();

  const auto perpoint_task = [&](
                               int i,
                               Eigen::Matrix<double, 6, 6>* H_00,
                               Eigen::Matrix<double, 6, 6>* H_11,
                               Eigen::Matrix<double, 6, 6>* H_01,
                               Eigen::Matrix<double, 6, 1>* b_0,
                               Eigen::Matrix<double, 6, 1>* b_1) {
    const long target_index = correspondences[i];
    if (target_index < 0) {
      return 0.0;
    }

    const int time_index = time_indices[i];

    const auto& pose = source_poses[time_index];
    const auto& H_pose_0 = pose_derivatives_t0[time_index];
    const auto& H_pose_1 = pose_derivatives_t1[time_index];

    const auto& source_pt = frame::point(*source, i);
    const auto& target_pt = frame::point(*target, target_index);
    const auto& target_normal = frame::normal(*target, target_index);

    gtsam::Matrix36 H_transed_pose;
    gtsam::Point3 transed_source_pt = pose.transformFrom(source_pt.template head<3>(), H_transed_pose);

    gtsam::Point3 residual = transed_source_pt - target_pt.template head<3>();

    gtsam::Matrix13 H_error_transed;
    double error = gtsam::dot(residual, target_normal.template head<3>(), H_error_transed);

    gtsam::Matrix16 H_error_pose = H_error_transed * H_transed_pose;
    gtsam::Matrix16 H_0 = H_error_pose * H_pose_0;
    gtsam::Matrix16 H_1 = H_error_pose * H_pose_1;

    *H_00 += H_0.transpose() * H_0;
    *H_11 += H_1.transpose() * H_1;
    *H_01 += H_0.transpose() * H_1;
    *b_0 += H_0.transpose() * error;
    *b_1 += H_1.transpose() * error;

    return error * error;
  };

  double error = 0.0;
  gtsam::Matrix6 H_00;
  gtsam::Matrix6 H_01;
  gtsam::Matrix6 H_11;
  gtsam::Vector6 b_0;
  gtsam::Vector6 b_1;

  if (is_omp_default() || num_threads == 1) {
    error = scan_matching_reduce_omp(perpoint_task, frame::size(*this->source), this->num_threads, &H_00, &H_11, &H_01, &b_0, &b_1);
  } else {
    error = scan_matching_reduce_tbb(perpoint_task, frame::size(*this->source), &H_00, &H_11, &H_01, &b_0, &b_1);
  }

  auto factor = gtsam::HessianFactor::shared_ptr(new gtsam::HessianFactor(this->keys_[0], this->keys_[1], H_00, H_01, -b_0, H_11, -b_1, error));
  return factor;
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::update_poses(const gtsam::Values& values) const {
  gtsam::Pose3 pose0 = values.at<gtsam::Pose3>(keys_[0]);
  gtsam::Pose3 pose1 = values.at<gtsam::Pose3>(keys_[1]);

  gtsam::Matrix6 H_delta_0, H_delta_1;
  gtsam::Pose3 delta = pose0.between(pose1, H_delta_0, H_delta_1);

  gtsam::Matrix6 H_vel_delta;
  gtsam::Vector6 vel = gtsam::Pose3::Logmap(delta, H_vel_delta);

  source_poses.resize(time_table.size());
  pose_derivatives_t0.resize(time_table.size());
  pose_derivatives_t1.resize(time_table.size());

  const auto task = [&](int i) {
    const double t = time_table[i];

    gtsam::Matrix6 H_inc_vel;
    gtsam::Pose3 inc = gtsam::Pose3::Expmap(t * vel, H_inc_vel);

    gtsam::Matrix6 H_pose_0_a, H_pose_inc;
    source_poses[i] = pose0.compose(inc, H_pose_0_a, H_pose_inc);

    gtsam::Matrix6 H_pose_delta = H_pose_inc * H_inc_vel * t * H_vel_delta;

    pose_derivatives_t0[i] = H_pose_0_a + H_pose_delta * H_delta_0;
    pose_derivatives_t1[i] = H_pose_delta * H_delta_1;
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(this->num_threads) schedule(guided, 32)
    for (int i = 0; i < time_table.size(); i++) {
      task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, time_table.size(), 32), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        task(i);
      }
    });
#else
    std::cerr << "error: TBB is not available" << std::endl;
    abort();
#endif
  }
}

template <typename TargetFrame, typename SourceFrame>
void IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::update_correspondences() const {
  correspondences.resize(frame::size(*source));

  const auto perpoint_task = [&](int i) {
    const int time_index = time_indices[i];

    const auto& pt = frame::point(*source, i);
    gtsam::Point3 transed_pt = source_poses[time_index] * pt.template head<3>();

    size_t k_index = -1;
    double k_sq_dist = -1;
    size_t num_found = target_tree->knn_search(transed_pt.data(), 1, &k_index, &k_sq_dist, max_correspondence_distance_sq);

    if (num_found == 0 || k_sq_dist > max_correspondence_distance_sq) {
      correspondences[i] = -1;
    } else {
      correspondences[i] = k_index;
    }
  };

  if (is_omp_default() || num_threads == 1) {
#pragma omp parallel for num_threads(this->num_threads) schedule(guided, 8)
    for (int i = 0; i < frame::size(*this->source); i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, frame::size(*this->source), 8), [&](const tbb::blocked_range<int>& range) {
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
std::vector<Eigen::Vector4d> IntegratedCT_ICPFactor_<TargetFrame, SourceFrame>::deskewed_source_points(const gtsam::Values& values, bool local) {
  update_poses(values);

  if (local) {
    for (auto& pose : source_poses) {
      pose = values.at<gtsam::Pose3>(keys_[0]).inverse() * pose;
    }
  }

  std::vector<Eigen::Vector4d> deskewed(frame::size(*source));
  for (int i = 0; i < frame::size(*source); i++) {
    const int time_index = time_indices[i];
    const auto& pose = source_poses[time_index];
    deskewed[i] = pose.matrix() * (Eigen::Vector4d() << frame::point(*source, i).template head<3>(), 1.0).finished();
  }

  return deskewed;
}

}  // namespace gtsam_points