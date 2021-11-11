#include <gtsam_ext/factors/integrated_colored_gicp_factor.hpp>

#include <vector>
#include <gtsam_ext/ann/kdtree.hpp>
#include <gtsam_ext/ann/nearest_neighbor_search.hpp>

namespace gtsam_ext {

IntegratedColoredGICPFactor::IntegratedColoredGICPFactor(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const Frame::ConstPtr& target,
  const Frame::ConstPtr& source,
  const std::shared_ptr<NearestNeighborSearch>& target_tree,
  const IntensityGradients::ConstPtr& target_gradients)
: IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  photometric_term_weight(0.25),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source),
  target_tree(target_tree),
  target_gradients(target_gradients) {
  //
  if (!target->has_points() || !target->has_normals() || !target->has_covs() || !target->has_intensities()) {
    std::cerr << "error: target frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }

  if (!source->has_points() || !source->has_covs() || !source->has_intensities()) {
    std::cerr << "error: source frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }
}

IntegratedColoredGICPFactor::~IntegratedColoredGICPFactor() {}

void IntegratedColoredGICPFactor::update_correspondences(const Eigen::Isometry3d& delta) const {
  bool do_update = true;
  if (correspondences.size() == source->size() && (correspondence_update_tolerance_trans > 0.0 || correspondence_update_tolerance_rot > 0.0)) {
    Eigen::Isometry3d diff = delta.inverse() * last_correspondence_point;
    double diff_rot = Eigen::AngleAxisd(diff.linear()).angle();
    double diff_trans = diff.translation().norm();
    if (diff_rot < correspondence_update_tolerance_rot && diff_trans < correspondence_update_tolerance_trans) {
      do_update = false;
    }
  }

  correspondences.resize(source->size());
  mahalanobis.resize(source->size());

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < source->size(); i++) {
    if (do_update) {
      Eigen::Vector4d pt = delta * source->points[i];

      size_t k_index = -1;
      double k_sq_dist = -1;
      size_t num_found = target_tree->knn_search(pt.data(), 1, &k_index, &k_sq_dist);
      correspondences[i] = k_sq_dist < max_correspondence_distance_sq ? k_index : -1;
    }

    if (correspondences[i] < 0) {
      mahalanobis[i].setIdentity();
    } else {
      const auto& target_cov = target->covs[correspondences[i]];
      Eigen::Matrix4d RCR = (target_cov + delta.matrix() * source->covs[i] * delta.matrix().transpose());
      RCR(3, 3) = 1.0;
      mahalanobis[i] = RCR.inverse();
      mahalanobis[i](3, 3) = 0.0;
    }
  }

  last_correspondence_point = delta;
}

double IntegratedColoredGICPFactor::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //
  const double geometric_term_weight = 1.0 - photometric_term_weight;

  double sum_errors_geom = 0.0;
  double sum_errors_photo = 0.0;

  if (H_target) {
    H_target->setZero();
    H_source->setZero();
    H_target_source->setZero();
    b_target->setZero();
    b_source->setZero();
  }

  //
  for (int i = 0; i < source->size(); i++) {
    const int target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    // source
    const auto& mean_A = source->points[i];
    const auto& cov_A = source->covs[i];
    const double intensity_A = source->intensities[i];

    // target
    const auto& mean_B = target->points[target_index];
    const auto& cov_B = target->covs[target_index];
    const auto& normal_B = target->normals[target_index];
    const auto& gradient_B = target_gradients->intensity_gradients[target_index];
    const double intensity_B = target->intensities[target_index];

    const Eigen::Vector4d transed_A = delta * mean_A;

    // geometric error
    const Eigen::Vector4d error_geom = transed_A - mean_B;
    sum_errors_geom += 0.5 * error_geom.transpose() * geometric_term_weight * mahalanobis[i] * error_geom;

    // photometric error
    const Eigen::Vector4d projected = transed_A - (transed_A - mean_B).dot(normal_B) * normal_B;
    const Eigen::Vector4d offset = projected - mean_B;
    const double error_photo = intensity_B + gradient_B.dot(offset) - intensity_A;

    sum_errors_photo += 0.5 * error_photo * photometric_term_weight * error_photo;

    if (!H_target) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> J_transed_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_transed_target.block<3, 3>(0, 0) = gtsam::SO3::Hat(transed_A.head<3>());
    J_transed_target.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_transed_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_transed_source.block<3, 3>(0, 0) = -delta.linear() * gtsam::SO3::Hat(mean_A.head<3>());
    J_transed_source.block<3, 3>(0, 3) = delta.linear();

    // geometric error
    const auto& J_egeom_target = J_transed_target;
    const auto& J_egeom_source = J_transed_source;

    (*H_target) += J_egeom_target.transpose() * geometric_term_weight * mahalanobis[i] * J_egeom_target;
    (*H_source) += J_egeom_source.transpose() * geometric_term_weight * mahalanobis[i] * J_egeom_source;
    (*H_target_source) += J_egeom_target.transpose() * geometric_term_weight * mahalanobis[i] * J_egeom_source;
    (*b_target) += J_egeom_target.transpose() * geometric_term_weight * mahalanobis[i] * error_geom;
    (*b_source) += J_egeom_source.transpose() * geometric_term_weight * mahalanobis[i] * error_geom;

    // photometric error
    Eigen::Matrix<double, 4, 4> J_projected_transed = Eigen::Matrix4d::Identity() - normal_B * normal_B.transpose();
    J_projected_transed(3, 3) = 0.0;
    const auto& J_offset_transed = J_projected_transed;

    Eigen::Matrix<double, 1, 4> J_ephoto_offset = gradient_B;
    Eigen::Matrix<double, 1, 4> J_ephoto_transed = J_ephoto_offset * J_offset_transed;

    Eigen::Matrix<double, 1, 6> J_ephoto_target = J_ephoto_transed * J_transed_target;
    Eigen::Matrix<double, 1, 6> J_ephoto_source = J_ephoto_transed * J_transed_source;

    (*H_target) += J_ephoto_target.transpose() * photometric_term_weight * J_ephoto_target;
    (*H_source) += J_ephoto_source.transpose() * photometric_term_weight * J_ephoto_source;
    (*H_target_source) += J_ephoto_target.transpose() * photometric_term_weight * J_ephoto_source;
    (*b_target) += J_ephoto_target.transpose() * photometric_term_weight * error_photo;
    (*b_source) += J_ephoto_source.transpose() * photometric_term_weight * error_photo;
  }

  return sum_errors_geom + sum_errors_photo;
}

}  // namespace gtsam_ext