#include <regex>
#include <chrono>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/format.hpp>

#include <gtsam/nonlinear/Values.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/util/read_points.hpp>

#ifdef GTSAM_POINTS_USE_CUDA
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/cuda/cuda_device_prop.hpp>
#endif

class StopWatch {
public:
  StopWatch() {}

  void start() { t1 = std::chrono::high_resolution_clock::now(); }
  void stop(const std::string& label) {
    t2 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
    std::cout << boost::format("- %-20s: %.3f[msec]") % label % elapsed << std::endl;
  }

private:
  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
};

template <typename T, int N>
double benchmark_eigen_fixed() {
  using MatrixN = Eigen::Matrix<T, N, N>;
  using VectorN = Eigen::Matrix<T, N, 1>;

  std::cout << boost::format("eigen_fixed(%s%d)") % typeid(T).name() % N << std::endl;

  const int num_points = 8192 * 10;
  const int num_iterations = 1024 * 16;

  StopWatch stopwatch;
  double sink = 0.0;

  std::vector<VectorN> points(num_points, VectorN::Zero());
  std::vector<VectorN> transformed(num_points);
  MatrixN transformation = MatrixN::Identity();

  stopwatch.start();
  for (int k = 0; k < num_iterations; k++) {
    for (int i = 0; i < num_points; i++) {
      transformed[i] = transformation * points[i];
    }
  }
  stopwatch.stop("mat4 x vec4");

  stopwatch.start();
  for (int k = 0; k < num_iterations; k++) {
#pragma omp parallel for
    for (int i = 0; i < num_points; i++) {
      transformed[i] = transformation * points[i];
    }
  }
  stopwatch.stop("mat4 x vec4 (par)");

  return sink;
}

template <typename T>
double benchmark_eigen_dynamic() {
  const int matrix_size = 1024;
  const int num_iterations = 1024 * 16;

  StopWatch stopwatch;
  double sink = 0.0;

  Eigen::Matrix<T, -1, -1> A = Eigen::Matrix<T, -1, -1>::Zero(matrix_size, matrix_size);
  Eigen::Matrix<T, -1, -1> b = Eigen::Matrix<T, -1, -1>::Zero(matrix_size, 1);

  stopwatch.start();
  for (int k = 0; k < num_iterations; k++) {
    sink += (A * b).sum();
  }
  stopwatch.stop("matX x vecX");

  return sink;
}

void benchmark_alignment(const std::string& factor_type, int num_threads, int num_factors) {
  std::cout << boost::format("%s (%d threads, x%d factors)") % factor_type % num_threads % num_factors << std::endl;

  const std::string dump_path = "data/kitti_07_dump";
  std::ifstream ifs(dump_path + "/graph.txt");
  if (!ifs) {
    std::cerr << "error: failed to open " << dump_path << "/graph.txt" << std::endl;
  }

  gtsam::Values gt_values;
  gtsam::Values values;
  for (int i = 0; i < 5; i++) {
    std::string token;
    Eigen::Vector3d trans;
    Eigen::Quaterniond quat;
    ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

    gtsam::Vector6 noise_gen = (gtsam::Vector6() << 0.0, 0.1, 0.2, 0.3, 0.4, 0.5).finished();
    gtsam::Vector6 noise_vec = (i + noise_gen.array()).sin();
    noise_vec.head<3>() *= 0.1;
    noise_vec.tail<3>() *= 1.0;
    gtsam::Pose3 noise = gtsam::Pose3::Expmap(noise_vec);

    gtsam::Pose3 pose(gtsam::Rot3(quat.toRotationMatrix()), trans);
    gt_values.insert(i, pose);
    values.insert(i, pose * noise);
  }

  StopWatch stopwatch;

  std::vector<gtsam_points::PointCloud::Ptr> frames;
  std::vector<std::shared_ptr<gtsam_points::KdTree>> trees;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxels;

  stopwatch.start();
  for (int i = 0; i < 5; i++) {
    const auto points = gtsam_points::read_points((boost::format("%s/%06d/points.bin") % dump_path % i).str());

    if (factor_type.find("GPU") == std::string::npos) {
      auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
      frame->add_covs(gtsam_points::estimate_covariances(frame->points, frame->size(), 10, num_threads));
      frames.emplace_back(frame);
    } else {
#ifdef GTSAM_POINTS_USE_CUDA
      auto frame = std::make_shared<gtsam_points::PointCloudGPU>(points);
      frame->add_covs(gtsam_points::estimate_covariances(frame->points, frame->size(), 10, num_threads));
      frames.emplace_back(frame);
#else
      std::cerr << "error: gtsam_points was built without CUDA!!" << std::endl;
#endif
    }
  }
  stopwatch.stop("frame creation");

  if (factor_type == "GICP") {
    stopwatch.start();
    for (int i = 0; i < 5; i++) {
      trees.emplace_back(std::make_shared<gtsam_points::KdTree>(frames[i]->points, frames[i]->size()));
    }
    stopwatch.stop("kdtree creation");
  } else if (factor_type == "VGICP") {
    stopwatch.start();
    for (int i = 0; i < 5; i++) {
      voxels.emplace_back(std::make_shared<gtsam_points::GaussianVoxelMapCPU>(1.0));
      voxels.back()->insert(*frames[i]);
    }
    stopwatch.stop("voxelmap creation");
  } else if (factor_type == "VGICP_GPU") {
#ifdef GTSAM_POINTS_USE_CUDA
    stopwatch.start();
    for (int i = 0; i < 5; i++) {
      voxels.emplace_back(std::make_shared<gtsam_points::GaussianVoxelMapGPU>(1.0));
      voxels.back()->insert(*frames[i]);
    }
    stopwatch.stop("voxelmap creation");
#else
    std::cerr << "error: gtsam_points was built without CUDA!!" << std::endl;
#endif
  } else {
    std::cerr << "error: unknown factor type " << factor_type << std::endl;
  }

#ifdef GTSAM_POINTS_USE_CUDA
  gtsam_points::StreamTempBufferRoundRobin stream_buffer_roundrobin;
#endif

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gt_values.at<gtsam::Pose3>(0), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  for (int k = 0; k < num_factors; k++) {
    for (int i = 0; i < 5; i++) {
      for (int j = i + 1; j < 5; j++) {
        if (factor_type == "GICP") {
          auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(i, j, frames[i], frames[j], trees[i]);
          factor->set_num_threads(num_threads);
          graph.add(factor);
        } else if (factor_type == "VGICP") {
          auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(i, j, voxels[i], frames[j]);
          factor->set_num_threads(num_threads);
          graph.add(factor);
        } else if (factor_type == "VGICP_GPU") {
#ifdef GTSAM_POINTS_USE_CUDA
          const auto stream_buffer = stream_buffer_roundrobin.get_stream_buffer();
          const auto stream = stream_buffer.first;
          const auto buffer = stream_buffer.second;

          auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(i, j, voxels[i], frames[j], stream, buffer);
          graph.add(factor);
#else
          std::cerr << "error: gtsam_points was built without CUDA!!" << std::endl;
#endif
        }
      }
    }
  }

  gtsam_points::LevenbergMarquardtExtParams lm_params;
  gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);

  stopwatch.start();
  values = optimizer.optimize();
  stopwatch.stop("optimization");

  for (int i = 0; i < 5; i++) {
    const gtsam::Pose3 error = gt_values.at<gtsam::Pose3>(i).inverse() * values.at<gtsam::Pose3>(i);
    const double trans_error = error.translation().norm();
    const double rot_error = Eigen::AngleAxisd(error.rotation().matrix()).angle();

    if (trans_error > 0.1 || rot_error > 0.01) {
      std::cerr << "error: too large estimation error!!" << std::endl;
      std::cerr << "     : frame=" << i << " err_t=" << trans_error << " err_r=" << rot_error << std::endl;
    }
  }
}

void print_info() {
  std::ifstream ifs("/proc/cpuinfo");
  if (ifs) {
    std::string line;
    std::regex pattern("model name\\s+:\\s*(.*)");
    while (!ifs.eof() && std::getline(ifs, line)) {
      std::smatch matched;
      if (std::regex_search(line, matched, pattern)) {
        std::cout << "CPU model  : " << matched.str(1) << std::endl;
        break;
      }
    }
  }

#ifdef GTSAM_POINTS_USE_CUDA
  const auto cuda_devices = gtsam_points::cuda_device_names();
  for (int i = 0; i < cuda_devices.size(); i++) {
    std::cout << boost::format("GPU model%d : %s") % i % cuda_devices[i] << std::endl;
  }

#endif

  std::cout << "SIMD in use: " << Eigen::SimdInstructionSetsInUse() << std::endl;
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  print_info();

  benchmark_eigen_fixed<float, 4>();
  benchmark_eigen_dynamic<float>();
  std::cout << std::endl;

  benchmark_eigen_fixed<double, 4>();
  benchmark_eigen_dynamic<double>();
  std::cout << std::endl;

#ifdef _OPENMP
  const int max_num_threads = omp_get_max_threads();
#else
  const int max_num_threads = 1;
#endif

  benchmark_alignment("GICP", 1, 1);
  benchmark_alignment("GICP", max_num_threads, 1);
  benchmark_alignment("GICP", max_num_threads, 10);
  std::cout << std::endl;

  benchmark_alignment("VGICP", 1, 1);
  benchmark_alignment("VGICP", max_num_threads, 1);
  benchmark_alignment("VGICP", max_num_threads, 10);
  std::cout << std::endl;

#ifdef GTSAM_POINTS_USE_CUDA
  benchmark_alignment("VGICP_GPU", 1, 1);
  benchmark_alignment("VGICP_GPU", max_num_threads, 1);
  benchmark_alignment("VGICP_GPU", max_num_threads, 10);
  benchmark_alignment("VGICP_GPU", max_num_threads, 100);
#endif
}