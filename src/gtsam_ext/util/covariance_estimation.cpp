#include <gtsam_ext/util/covariance_estimation.hpp>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <nanoflann.hpp>

namespace gtsam_ext {

namespace {

struct KdTree {
public:
  using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 3>;

  KdTree(int num_points, const Eigen::Vector4d* points) : num_points(num_points), points(points), index(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    index.buildIndex();
  }

  inline size_t kdtree_get_point_count() const { return num_points; }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }

  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }

public:
  const int num_points;
  const Eigen::Vector4d* points;

  Index index;
};

}  // namespace

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> estimate_covariances(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  int k_neighbors) {
  //
  KdTree tree(points.size(), points.data());

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs(points.size());
  for (int i = 0; i < points.size(); i++) {
    std::vector<size_t> k_indices(k_neighbors);
    std::vector<double> k_sq_dists(k_neighbors);
    tree.index.knnSearch(points[i].data(), k_neighbors, &k_indices[0], &k_sq_dists[0]);

    Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
    Eigen::Matrix4d sum_covs = Eigen::Matrix4d::Zero();

    for (int j = 0; j < k_neighbors; j++) {
      const auto& pt = points[k_indices[j]];
      sum_points += pt;
      sum_covs += pt * pt.transpose();
    }

    Eigen::Vector4d mean = sum_points / k_neighbors;
    Eigen::Matrix4d cov = (sum_covs - mean * sum_points.transpose()) / k_neighbors;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
    eig.computeDirect(cov.block<3, 3>(0, 0));

    Eigen::Vector3d values(1e-3, 1.0, 1.0);

    covs[i].setZero();
    covs[i].block<3, 3>(0, 0) = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().inverse();
  }

  return covs;
}

}  // namespace gtsam_ext
