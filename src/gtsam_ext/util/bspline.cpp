#include <gtsam_ext/util/bspline.hpp>

#include <gtsam_ext/util/expressions.hpp>

namespace gtsam_ext {

gtsam::Pose3_ bspline(const gtsam::Key key0, const gtsam::Key key1, const gtsam::Key key2, const gtsam::Key key3, const gtsam::Double_& t) {
  gtsam::Double_ t2 = t * t;
  gtsam::Double_ t3 = t2 * t;

  gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Pose3_ pose0(key0);
  const gtsam::Pose3_ pose1(key1);
  const gtsam::Pose3_ pose2(key2);
  const gtsam::Pose3_ pose3(key3);

  const gtsam::Rot3_ rot0 = gtsam::rotation(pose0);
  const gtsam::Rot3_ rot1 = gtsam::rotation(pose1);
  const gtsam::Rot3_ rot2 = gtsam::rotation(pose2);
  const gtsam::Rot3_ rot3 = gtsam::rotation(pose3);

  const gtsam::Rot3_ r_delta1 = gtsam_ext::expmap(gtsam_ext::scale(beta1, gtsam::logmap(rot0, rot1)));
  const gtsam::Rot3_ r_delta2 = gtsam_ext::expmap(gtsam_ext::scale(beta2, gtsam::logmap(rot1, rot2)));
  const gtsam::Rot3_ r_delta3 = gtsam_ext::expmap(gtsam_ext::scale(beta3, gtsam::logmap(rot2, rot3)));

  const gtsam::Vector3_ trans0 = gtsam_ext::translation(pose0);
  const gtsam::Vector3_ trans1 = gtsam_ext::translation(pose1);
  const gtsam::Vector3_ trans2 = gtsam_ext::translation(pose2);
  const gtsam::Vector3_ trans3 = gtsam_ext::translation(pose3);

  const gtsam::Vector3_ t_delta1 = gtsam_ext::scale(beta1, gtsam::between(trans0, trans1));
  const gtsam::Vector3_ t_delta2 = gtsam_ext::scale(beta2, gtsam::between(trans1, trans2));
  const gtsam::Vector3_ t_delta3 = gtsam_ext::scale(beta3, gtsam::between(trans2, trans3));

  const gtsam::Rot3_ rot = gtsam::compose(gtsam::compose(gtsam::compose(rot0, r_delta1), r_delta2), r_delta3);
  const gtsam::Vector3_ trans = gtsam::compose(gtsam::compose(gtsam::compose(trans0, t_delta1), t_delta2), t_delta3);
  const gtsam::Pose3_ pose = gtsam_ext::se3(rot, trans);

  return pose;
}

gtsam::Pose3_ bspline_se3(const gtsam::Key key0, const gtsam::Key key1, const gtsam::Key key2, const gtsam::Key key3, const gtsam::Double_& t) {
  gtsam::Double_ t2 = t * t;
  gtsam::Double_ t3 = t2 * t;

  gtsam::Double_ beta1 = gtsam::Double_(5.0 / 6.0) + 3.0 / 6.0 * t - 3.0 / 6.0 * t2 + 1.0 / 6.0 * t3;
  gtsam::Double_ beta2 = gtsam::Double_(1.0 / 6.0) + 3.0 / 6.0 * t + 3.0 / 6.0 * t2 - 2.0 / 6.0 * t3;
  gtsam::Double_ beta3 = 1.0 / 6.0 * t3;

  const gtsam::Pose3_ pose0(key0);
  const gtsam::Pose3_ pose1(key1);
  const gtsam::Pose3_ pose2(key2);
  const gtsam::Pose3_ pose3(key3);

  const gtsam::Pose3_ delta1 = gtsam_ext::expmap(gtsam_ext::scale(beta1, gtsam::logmap(pose0, pose1)));
  const gtsam::Pose3_ delta2 = gtsam_ext::expmap(gtsam_ext::scale(beta2, gtsam::logmap(pose1, pose2)));
  const gtsam::Pose3_ delta3 = gtsam_ext::expmap(gtsam_ext::scale(beta3, gtsam::logmap(pose2, pose3)));

  const gtsam::Pose3_ pose = gtsam::compose(gtsam::compose(gtsam::compose(pose0, delta1), delta2), delta3);

  return pose;
}

}