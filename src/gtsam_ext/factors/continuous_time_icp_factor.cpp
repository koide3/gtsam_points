#include <gtsam_ext/factors/continuous_time_icp_factor.hpp>

#include <memory>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam_ext/util/expressions.hpp>

#include <gtsam_ext/types/kdtree.hpp>

namespace gtsam_ext {

CTICPFactorExpr::CTICPFactorExpr(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const std::shared_ptr<const Frame>& target,
  const std::shared_ptr<const KdTree>& target_tree,
  const double source_t0,
  const double source_t1,
  const double source_ti,
  const gtsam::Point3& source_pt,
  const gtsam::SharedNoiseModel& noise_model)
: gtsam::NoiseModelFactor(noise_model, gtsam::cref_list_of<2>(source_t0_key)(source_t1_key)),
  target(target),
  target_tree(target_tree),
  source_time((source_ti - source_t0) / (source_t1 - source_t0)),
  source_pt(source_pt),
  transed_source_pt(transform_source_point()),
  target_index(-1),
  error_expr(0.0) {
  //
  if (target->points == nullptr) {
    std::cerr << "error: target points have not been allocated!!" << std::endl;
    abort();
  }

  if (target->normals == nullptr) {
    std::cerr << "error: target doesn't have normals!!" << std::endl;
    abort();
  }
}

CTICPFactorExpr::~CTICPFactorExpr() {}

gtsam::Vector CTICPFactorExpr::unwhitenedError(const gtsam::Values& values, boost::optional<std::vector<gtsam::Matrix>&> H) const {
  // Update corresponding point at the first call and for every linearization
  if (target_index < 0 || H) {
    update_correspondence(values);
    error_expr = calc_error();
  }

  gtsam::Vector err(1);
  err[0] = error_expr.value(values, H);
  return err;
}

void CTICPFactorExpr::update_correspondence(const gtsam::Values& values) const {
  gtsam::Point3 pt = transed_source_pt.value(values);

  size_t k_index = -1;
  double k_sq_dists = -1;
  target_tree->index.knnSearch(pt.data(), 1, &k_index, &k_sq_dists);
  target_index = k_index;
}

gtsam::Point3_ CTICPFactorExpr::transform_source_point() const {
  // We use expmap/logmap interpolation instead of SLERP
  // for simplicity of the implementation
  gtsam::Pose3_ pose0(keys()[0]);
  gtsam::Pose3_ pose1(keys()[1]);
  gtsam::Vector6_ vel = gtsam_ext::logmap(gtsam::between(pose0, pose1));
  gtsam::Pose3_ inc = gtsam_ext::expmap(source_time * vel);
  gtsam::Pose3_ pose = gtsam::compose(pose0, inc);

  return gtsam::transformFrom(pose, gtsam::Point3_(source_pt));
}

gtsam::Double_ CTICPFactorExpr::calc_error() const {
  // We omit alpha (planarity term) in Eq. (3)
  const auto& target_pt = target->points[target_index];
  const auto& target_normal = target->normals[target_index];

  gtsam::Point3_ error = gtsam::Point3_(target_pt.head<3>()) - transed_source_pt;
  return gtsam::dot(error, gtsam::Point3_(target_normal.head<3>()));
}

/**
 * @brief IntegratedCTICPFactorExpr
 */
IntegratedCTICPFactorExpr::IntegratedCTICPFactorExpr(const gtsam::NonlinearFactorGraph::shared_ptr& graph) : gtsam::NonlinearFactor(graph->keys()), graph(graph) {}

IntegratedCTICPFactorExpr::~IntegratedCTICPFactorExpr() {}

double IntegratedCTICPFactorExpr::error(const gtsam::Values& values) const {
  return graph->error(values);
}

boost::shared_ptr<gtsam::GaussianFactor> IntegratedCTICPFactorExpr::linearize(const gtsam::Values& values) const {
  // I'm not 100% sure if it is a correct way
  gtsam::Ordering ordering;
  ordering.push_back(keys()[0]);
  ordering.push_back(keys()[1]);

  auto linearized = graph->linearize(values);
  auto hessian = linearized->hessian(ordering);
  const auto& H = hessian.first;
  const auto& b = hessian.second;

  return gtsam::GaussianFactor::shared_ptr(
    new gtsam::HessianFactor(keys()[0], keys()[1], H.block<6, 6>(0, 0), H.block<6, 6>(0, 6), b.head<6>(), H.block<6, 6>(6, 6), b.tail<6>(), 0.0));
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> IntegratedCTICPFactorExpr::deskewed_source_points(const gtsam::Values& values) const {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;
  for (const auto& factor : *graph) {
    auto f = boost::dynamic_pointer_cast<gtsam_ext::CTICPFactorExpr>(factor);
    if (f) {
      points.push_back(f->transform_source_point().value(values));
    }
  }
  return points;
}

gtsam::NonlinearFactorGraph::shared_ptr
create_cticp_factors(gtsam::Key source_t0_key, gtsam::Key source_t1_key, const Frame::ConstPtr& target, const Frame::ConstPtr& source, const gtsam::SharedNoiseModel& noise_model) {
  gtsam::NonlinearFactorGraph::shared_ptr factors(new gtsam::NonlinearFactorGraph);
  std::shared_ptr<KdTree> target_tree(new KdTree(target->size(), target->points));

  for (int i = 0; i < source->size(); i++) {
    const double source_t0 = source->times[0];
    const double source_t1 = source->times[source->size() - 1];
    const double source_ti = source->times[i];

    gtsam::NonlinearFactor::shared_ptr factor(
      new CTICPFactorExpr(source_t0_key, source_t1_key, target, target_tree, source_t0, source_t1, source_ti, source->points[i].head<3>(), noise_model));
    factors->push_back(factor);
  }

  return factors;
}

IntegratedCTICPFactorExpr::shared_ptr create_integrated_cticp_factor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const Frame::ConstPtr& target,
  const Frame::ConstPtr& source,
  const gtsam::SharedNoiseModel& noise_model) {
  //
  auto factors = create_cticp_factors(source_t0_key, source_t1_key, target, source, noise_model);
  return IntegratedCTICPFactorExpr::shared_ptr(new IntegratedCTICPFactorExpr(factors));
}

}  // namespace gtsam_ext