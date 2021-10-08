#pragma once

#include <gtsam_ext/factors/integrated_ct_icp_factor.hpp>

namespace gtsam_ext {

class IntegratedCT_GICPFactor : public IntegratedCT_ICPFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using shared_ptr = boost::shared_ptr<IntegratedCT_GICPFactor>;

  IntegratedCT_GICPFactor(
    gtsam::Key source_t0_key,
    gtsam::Key source_t1_key,
    const gtsam_ext::Frame::ConstPtr& target,
    const gtsam_ext::Frame::ConstPtr& source,
    const std::shared_ptr<NearestNeighborSearch>& target_tree);

  IntegratedCT_GICPFactor(gtsam::Key source_t0_key, gtsam::Key source_t1_key, const gtsam_ext::Frame::ConstPtr& target, const gtsam_ext::Frame::ConstPtr& source);

  virtual ~IntegratedCT_GICPFactor() override;

  virtual double error(const gtsam::Values& values) const override;
  virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const override;

protected:
  virtual void update_correspondences() const override;

  mutable std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis;
};

}  // namespace gtsam_ext
