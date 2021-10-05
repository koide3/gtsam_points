#pragma once

#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_ext/types/frame.hpp>

namespace gtsam_ext {

class KdTree;

/**
 * @brief Continuous Time ICP Factor
 * @note  This implementation is really slow and not well tested
 * @ref Bellenbach et al., "CT-ICP: Real-time Elastic LiDAR Odometry with Loop Closure", 2021
 */
class CTICPFactorExpr : public gtsam::NoiseModelFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CTICPFactorExpr(
    gtsam::Key source_t0_key,
    gtsam::Key source_t1_key,
    const std::shared_ptr<const Frame>& target,
    const std::shared_ptr<const KdTree>& target_tree,
    const double source_t0,     // time of the very first point in source
    const double source_t1,     // time of the very last point in source
    const double source_ti,     // time of the point in interest (source_pt)
    const gtsam::Point3& source_pt,
    const gtsam::SharedNoiseModel& noise_model);

  ~CTICPFactorExpr();

  virtual gtsam::Vector unwhitenedError(const gtsam::Values& values, boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const;

  void update_correspondence(const gtsam::Values& values) const;

  gtsam::Point3_ transform_source_point() const;
  gtsam::Double_ calc_error() const;

private:
  const std::shared_ptr<const Frame> target;
  const std::shared_ptr<const KdTree> target_tree;

  const double source_time;       // source point time \in [0.0, 1.0]
  const gtsam::Point3 source_pt;  // source point
  mutable gtsam::Point3_ transed_source_pt;

  mutable int target_index;
  mutable gtsam::Double_ error_expr;
};

class IntegratedCTICPFactorExpr : public gtsam::NonlinearFactor {
public:
  using shared_ptr = boost::shared_ptr<IntegratedCTICPFactorExpr>;

  IntegratedCTICPFactorExpr(const gtsam::NonlinearFactorGraph::shared_ptr& graph);
  ~IntegratedCTICPFactorExpr();

  virtual size_t dim() const override { return 6; }

  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> deskewed_source_points(const gtsam::Values& values) const;

  gtsam::NonlinearFactorGraph::shared_ptr graph;
};

/**
 * @brief Create a set of CT-ICP factors
 */
gtsam::NonlinearFactorGraph::shared_ptr
create_cticp_factors(gtsam::Key source_t0_key, gtsam::Key source_t1_key, const Frame::ConstPtr& target, const Frame::ConstPtr& source, const gtsam::SharedNoiseModel& noise_model);

/**
 * @brief Create a nonlinear factor that wraps a set of CT-ICP factors
 */
IntegratedCTICPFactorExpr::shared_ptr create_integrated_cticp_factor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const Frame::ConstPtr& target,
  const Frame::ConstPtr& source,
  const gtsam::SharedNoiseModel& noise_model);

}  // namespace gtsam_ext