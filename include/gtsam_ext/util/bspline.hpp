#pragma once

#include <gtsam/slam/expressions.h>

namespace gtsam_ext {

/**
 * @brief B-Spline pose interpolation
 *        Rotation and translation will be independently interpolated
 * @note  Requirement: t0 < t1 < t2 < t3 and t is the normalized time between t1 and t2 in [0, 1]
 * @ref   Sec. 2.2 in https://www.robots.ox.ac.uk/~mobile/Theses/StewartThesis.pdf
 * @param key0  Pose at t0
 * @param key1  Pose at t1
 * @param key2  Pose at t2
 * @param key3  Pose at t3
 * @param t     Normalized time between t1 and t2 in [0, 1]
 * @return      Interpolated pose
 */
gtsam::Pose3_ bspline(const gtsam::Key key0, const gtsam::Key key1, const gtsam::Key key2, const gtsam::Key key3, const gtsam::Double_& t);

/**
 * @brief B-Spline pose interpolation
 *        Rotation and translation will be jointly interpolated
 *        This would be suitable for interpolating twist motion (e.g., vehicle motion)
 * @note  Requirement: t0 < t1 < t2 < t3 and t is the normalized time between t1 and t2 in [0, 1]
 * @ref   Sec. 2.2 in https://www.robots.ox.ac.uk/~mobile/Theses/StewartThesis.pdf
 * @param key0  Pose at t0
 * @param key1  Pose at t1
 * @param key2  Pose at t2
 * @param key3  Pose at t3
 * @param t     Normalized time between t1 and t2 in [0, 1]
 * @return      Interpolated pose
 */
gtsam::Pose3_ bspline_se3(const gtsam::Key key0, const gtsam::Key key1, const gtsam::Key key2, const gtsam::Key key3, const gtsam::Double_& t);

}  // namespace gtsam_ext
