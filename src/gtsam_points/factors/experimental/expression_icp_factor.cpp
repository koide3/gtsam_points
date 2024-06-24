// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/experimental/expression_icp_factor.hpp>

#include <memory>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam_points/util/expressions.hpp>

#include <gtsam_points/ann/kdtree.hpp>

namespace gtsam_points {

ICPFactorExpr::ICPFactorExpr(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const PointCloud>& target,
  const std::shared_ptr<const KdTree>& target_tree,
  const gtsam::Point3& source,
  const gtsam::SharedNoiseModel& noise_model)
: gtsam::NoiseModelFactor(noise_model, gtsam::KeyVector{target_key, source_key}),
  target(target),
  target_tree(target_tree),
  delta(gtsam::between(gtsam::Pose3_(target_key), gtsam::Pose3_(source_key))),
  source(source),
  target_index(-1),
  error_expr(calc_error()) {}

ICPFactorExpr::~ICPFactorExpr() {}

gtsam::Vector ICPFactorExpr::unwhitenedError(const gtsam::Values& values, boost::optional<std::vector<gtsam::Matrix>&> H) const {
  // Update corresponding point at the first call and every linearization call
  if (target_index < 0 || H) {
    update_correspondence(values);
    error_expr = calc_error();
  }

  return error_expr.value(values, H);
}

void ICPFactorExpr::update_correspondence(const gtsam::Values& values) const {
  gtsam::Point3 transed_source = delta.value(values) * source;

  size_t k_index = -1;
  double k_sq_dists = -1;
  target_tree->knn_search(transed_source.data(), 1, &k_index, &k_sq_dists);
  target_index = k_index;
}

gtsam::Point3_ ICPFactorExpr::calc_error() const {
  if (target_index < 0) {
    return gtsam::Point3_(gtsam::Point3::Zero());
  }

  return gtsam::Point3_(target->points[target_index].head<3>()) - gtsam::transformFrom(delta, gtsam::Point3_(source));
}

// A class to hold a set of ICP factors
class IntegratedICPFactorExpr : public gtsam::NonlinearFactor {
public:
  IntegratedICPFactorExpr(const gtsam::NonlinearFactorGraph::shared_ptr& graph) : gtsam::NonlinearFactor(graph->keys()), graph(graph) {}

  virtual size_t dim() const override { return 3; }

  virtual double error(const gtsam::Values& values) const override { return graph->error(values); }
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override {
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

  gtsam::NonlinearFactorGraph::shared_ptr graph;
};

gtsam::NonlinearFactorGraph::shared_ptr
create_icp_factors(gtsam::Key target_key, gtsam::Key source_key, const PointCloud::ConstPtr& target, const PointCloud::ConstPtr& source, const gtsam::SharedNoiseModel& noise_model) {
  gtsam::NonlinearFactorGraph::shared_ptr factors(new gtsam::NonlinearFactorGraph);

  std::shared_ptr<KdTree> target_tree(new KdTree(target->points, target->size()));

  for (int i = 0; i < source->size(); i++) {
    gtsam::NonlinearFactor::shared_ptr factor(new ICPFactorExpr(target_key, source_key, target, target_tree, source->points[i].head<3>(), noise_model));
    factors->add(factor);
  }

  return factors;
}

gtsam::NonlinearFactor::shared_ptr
create_integrated_icp_factor(gtsam::Key target_key, gtsam::Key source_key, const PointCloud::ConstPtr& target, const PointCloud::ConstPtr& source, const gtsam::SharedNoiseModel& noise_model)
//
{
  auto factors = create_icp_factors(target_key, source_key, target, source, noise_model);
  return gtsam::NonlinearFactor::shared_ptr(new IntegratedICPFactorExpr(factors));
}

}  // namespace gtsam_points