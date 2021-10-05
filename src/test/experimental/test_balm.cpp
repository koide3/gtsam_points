#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigen>


struct BALMFeature {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BALMFeature(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points) {
    Eigen::Vector3d sum_pts = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sum_cross = Eigen::Matrix3d::Zero();
    for(const auto& pt: points) {
      sum_pts += pt;
      sum_cross += pt * pt.transpose();
    }

    num_points = points.size();
    mean = sum_pts / points.size();
    cov = (sum_cross - mean * sum_pts.transpose()) / points.size();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    eigenvalues = eig.eigenvalues();
    eigenvectors = eig.eigenvectors();
  }

  template<int k>
  Eigen::Matrix<double, 1, 3> Ji(const Eigen::Vector3d& p_i) const {
    Eigen::Vector3d u = eigenvectors.col(k);
    return 2.0 / num_points * (p_i - mean).transpose() * u * u.transpose();
  }

  template<int k>
  Eigen::Matrix3d Hij(const Eigen::Vector3d& p_i, const Eigen::Vector3d& p_j, bool i_equals_j) const {
    const int N = num_points;
    Eigen::Matrix3d F_k;
    F_k.row(0) = Fmn<0, k>(p_j);
    F_k.row(1) = Fmn<1, k>(p_j);
    F_k.row(2) = Fmn<2, k>(p_j);

    const auto& u_k = eigenvectors.col(k);
    const auto& U = eigenvectors;

    if (i_equals_j) {
      const auto t1 = (N - 1) / static_cast<double>(N) * u_k * u_k.transpose();
      const auto t2 = u_k * (p_i - mean).transpose() * U * F_k;
      const auto t3 = U * F_k * (u_k.transpose() * (p_i - mean));
      Eigen::Matrix3d H = 2.0 / N * (t1 + t2 + t3);
      return H;
    } else {
      const auto t1 = -1.0 / N * u_k * u_k.transpose();
      const auto t2 = u_k * (p_i - mean).transpose() * U * F_k;
      const auto t3 = U * F_k * (u_k.transpose() * (p_i - mean));
      Eigen::Matrix3d H = 2.0 / N * (t1 + t2 + t3);

      return H;
    }
  }

  template<int m, int n>
  Eigen::Matrix<double, 1, 3> Fmn(const Eigen::Vector3d& pt) const {
    if constexpr (m == n) {
      return Eigen::Matrix<double, 1, 3>::Zero();
    } else {
      const double l_m = eigenvalues[m];
      const double l_n = eigenvalues[n];
      const auto& u_m = eigenvectors.col(m);
      const auto& u_n = eigenvectors.col(n);

      const auto lhs = (pt - mean).transpose() / (num_points * (l_n - l_m));
      const auto rhs = u_m * u_n.transpose() + u_n * u_m.transpose();
      return lhs * rhs;
    }
  }

  int num_points;
  Eigen::Vector3d mean;
  Eigen::Matrix3d cov;

  Eigen::Vector3d eigenvalues;
  Eigen::Matrix3d eigenvectors;
};

template <typename Func>
Eigen::MatrixXd numerical_hessian(const Func& f, const Eigen::VectorXd& x, double eps = 1e-6) {
  const int N = x.size();
  Eigen::MatrixXd h(N, N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(N);
      dx[i] = eps;

      auto first = [&](const Eigen::VectorXd& dy) {
        const double y0 = f(x - dx + dy);
        const double y1 = f(x + dx + dy);
        return (y1 - y0) / (2.0 * eps);
      };

      Eigen::VectorXd dy = Eigen::VectorXd::Zero(N);
      dy[j] = eps;

      const double dx0 = first(-dy);
      const double dx1 = first(dy);

      h(i, j) = (dx1 - dx0) / (2.0 * eps);
    }
  }

  return h;
}

int main(int argc, char** argv) {
  int num_points = 6;

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3 * num_points);
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points(num_points);
  for (int i = 0; i < num_points; i++) {
    points[i] = Eigen::Vector3d::Random();
    x0.block<3, 1>(3 * i, 0) = points[i];
  }

  auto calc_lambda = [](const Eigen::VectorXd& x) {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points(x.size() / 3);
    for (int i = 0; i < x.size() / 3; i++) {
      points[i] = x.block<3, 1>(3 * i, 0);
    }

    BALMFeature feature(points);
    return feature.eigenvalues[0];
  };

  Eigen::MatrixXd H = numerical_hessian(calc_lambda, x0);
  std::cout << "--- Hn ---" << std::endl << H << std::endl;

  BALMFeature feature(points);
  Eigen::MatrixXd Ha(num_points * 3, num_points * 3);
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      Ha.block<3, 3>(3 * i, 3 * j) = feature.Hij<0>(points[i], points[j], i == j);
    }
  }
  std::cout << "--- Ha ---" << std::endl << Ha << std::endl;

  return 0;
}