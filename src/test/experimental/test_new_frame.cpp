#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct Frame {
  std::vector<double> stamps;
  std::vector<Eigen::Vector4d> points;
};

template <typename T>
struct traits {
  static bool has_times(const T&) { return false; }
};

template <>
struct traits<Frame> {
  static bool has_times(const Frame& frame) { return true; }
};

int main(int argc, char** argv) {
  return 0;
}