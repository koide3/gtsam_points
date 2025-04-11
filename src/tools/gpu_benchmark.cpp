#define FMT_HEADER_ONLY

#include <regex>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include <boost/program_options.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/base/serialization.h>

#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <guik/viewer/light_viewer.hpp>

// Export GTSAM classes for serialization
BOOST_CLASS_EXPORT_GUID(gtsam::GaussianFactorGraph, "gtsam::GaussianFactorGraph")
BOOST_CLASS_EXPORT_GUID(gtsam::JacobianFactor, "gtsam::JacobianFactor");
BOOST_CLASS_EXPORT_GUID(gtsam::HessianFactor, "gtsam::HessianFactor");

/// @brief Random noise feeder
/// @details This class generates random noise for the benchmark. It can also read noise from a file for reproducibility.
struct NoiseFeeder {
public:
  /// @brief Default constructor
  /// @details Generates random noise using a normal distribution.
  NoiseFeeder() : count(0) {
    std::mt19937 mt;
    std::normal_distribution<> ndist(0.0, 1.0);
    noise.resize(8192);
    std::generate(noise.begin(), noise.end(), [&] { return ndist(mt); });
  }

  /// @brief Read noise from a text file.
  void read(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
      std::cerr << "error : failed to open " << path << std::endl;
      return;
    }

    noise.clear();
    std::string line;
    while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
      noise.emplace_back(std::stod(line));
    }
  }

  /// @brief Take the next noise value.
  double operator()() { return noise[(count++) % noise.size()]; }

public:
  int count;                  ///< Generated noise count
  std::vector<double> noise;  ///< Noise values
};

/// @brief Estimation frame that contains the pose and point cloud data.
struct Frame {
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  int id;                  ///< Frame ID
  double stamp;            ///< Frame timestamp
  Eigen::Isometry3d pose;  ///< Frame pose in world coordinates (groundtruth)

  gtsam_points::PointCloudGPU::Ptr points;                        ///< Point cloud
  std::vector<gtsam_points::GaussianVoxelMapGPU::Ptr> voxelmaps;  ///< Voxel maps
};

/// @brief Benchmark dataset
struct Dataset {
public:
  using Ptr = std::shared_ptr<Dataset>;
  using ConstPtr = std::shared_ptr<const Dataset>;

  /// @brief Constructor
  /// @details Read the dataset from a directory
  Dataset(const std::filesystem::path& data_path) : data_path(data_path) {
    std::cout << "reading " + data_path.string() + "...\n" << std::flush;

    std::ifstream ifs(data_path);
    if (!ifs) {
      std::cerr << "error: failed to open " << data_path << std::endl;
      return;
    }

    bool reading_keyframes = false;
    std::string line;
    while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
      if (line == "frames") {
        continue;
      }
      if (line == "keyframes") {
        reading_keyframes = true;
        continue;
      }

      const std::string f = "[+-]?[0-9]+\\.[0-9]+";
      std::regex pattern(fmt::format("([0-9]+) ({}) se3\\(({}),({}),({}),({}),({}),({}),({})+\\)", f, f, f, f, f, f, f, f));
      std::smatch match;
      if (std::regex_search(line, match, pattern)) {
        auto frame = std::make_shared<Frame>();
        frame->id = std::stoi(match[1]);
        frame->stamp = std::stod(match[2]);

        const auto found = std::find_if(frames.begin(), frames.end(), [frame](const auto& f) { return f->id == frame->id; });
        if (reading_keyframes && found != frames.end()) {
          keyframes.emplace_back(*found);
          continue;
        }

        const Eigen::Vector3d trans(std::stod(match[3]), std::stod(match[4]), std::stod(match[5]));
        const Eigen::Quaterniond quat(std::stod(match[9]), std::stod(match[6]), std::stod(match[7]), std::stod(match[8]));
        frame->pose.setIdentity();
        frame->pose.linear() = quat.normalized().toRotationMatrix();
        frame->pose.translation() = trans;

        const std::string framename = reading_keyframes ? "keyframe" : "frame";
        const std::string path = data_path.parent_path() / fmt::format("{}_{:06d}", framename, frame->id);

        auto points = gtsam_points::PointCloudCPU::load(path);
        frame->points = gtsam_points::PointCloudGPU::clone(*points);

        const std::array<double, 2> resolutions = {0.5, 1.0};
        for (const auto r : resolutions) {
          auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(r);
          voxelmap->insert(*frame->points);
          frame->voxelmaps.emplace_back(voxelmap);
        }

        if (!reading_keyframes) {
          frames.emplace_back(frame);
        } else {
          keyframes.emplace_back(frame);
        }
      } else {
        std::cerr << "warning : invalid line " << line << std::endl;
      }
    }

    std::cout << fmt::format("|frames|={} |keyframes|={}", frames.size(), keyframes.size()) << std::endl;
  }

public:
  std::filesystem::path data_path;    ///< Dataset path
  std::vector<Frame::Ptr> frames;     ///< Frames
  std::vector<Frame::Ptr> keyframes;  ///< Keyframes
};

/// @brief Benchmark class
class Benchmark {
public:
  Benchmark(const Dataset::ConstPtr& dataset, NoiseFeeder& noise, std::ostream& log_os, const boost::program_options::variables_map& vm)
  : dataset(dataset) {
    std::cout << "dataset=" << dataset->data_path.string() << std::endl;
    log_os << "dataset=" << dataset->data_path.string() << std::endl;

    // Create a stream temp buffer round robin
    roundrobin.reset(new gtsam_points::StreamTempBufferRoundRobin(4));

    create_graph(0.1, 0.025, noise);

    // Visualization
    if (guik::running()) {
      auto viewer = guik::viewer();
      viewer->disable_vsync();
      viewer->clear_drawables();

      for (const auto& frame : dataset->frames) {
        viewer->update_points(
          "frame_" + std::to_string(frame->id),
          frame->points->points,
          frame->points->size(),
          guik::Rainbow(values.at<gtsam::Pose3>(frame->id).matrix()));
      }

      viewer->lookat(dataset->frames.front()->pose.translation().cast<float>());
      viewer->spin_once();
    }

    // Verify linearization result
    verify(log_os, vm);

    // Optimize graph for benchmark
    optimize(log_os, vm);
  }

private:
  /// @brief Create a graph to align all frames and keyframes
  void create_graph(double noise_t, double noise_r, NoiseFeeder& noise) {
    for (const auto& frame : dataset->frames) {
      // Add noise to the groundtruth pose
      const Eigen::Vector3d tvec = Eigen::Vector3d(noise(), noise(), noise()) * noise_t;
      const Eigen::Vector3d rvec = Eigen::Vector3d(noise(), noise(), noise()) * noise_r;
      const gtsam::Pose3 noise(gtsam::Rot3::Expmap(rvec), tvec);
      values.insert(frame->id, gtsam::Pose3(frame->pose.matrix()) * noise);
    }

    const auto create_binary_factor = [&](const Frame::ConstPtr& target, const Frame::ConstPtr& source) {
      auto [stream, buffer] = roundrobin->get_stream_buffer();
      for (const auto& voxelmap : target->voxelmaps) {
        auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(target->id, source->id, voxelmap, source->points, stream, buffer);
        graph.add(factor);
      }
    };

    const auto create_unary_factor = [&](const Frame::ConstPtr& keyframe, const Frame::ConstPtr& source) {
      auto [stream, buffer] = roundrobin->get_stream_buffer();
      for (const auto& voxelmap : keyframe->voxelmaps) {
        auto factor = gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(
          gtsam::Pose3(keyframe->pose.matrix()),
          source->id,
          voxelmap,
          source->points,
          stream,
          buffer);
        graph.add(factor);
      }
    };

    for (int i = 0; i < dataset->frames.size(); i++) {
      const auto& frame_i = dataset->frames[i];

      // Create binary factors for the last 3 frames
      for (int j = i - 3; j >= 0 && j < i; j++) {
        const auto& frame_j = dataset->frames[j];
        create_binary_factor(frame_j, frame_i);
      }

      // Create binary factors for the keyframes
      for (const auto& keyframe : dataset->keyframes) {
        if (keyframe->id >= frame_i->id) {
          break;
        }

        const auto found = std::find_if(dataset->frames.begin(), dataset->frames.end(), [keyframe](const auto& f) { return f->id == keyframe->id; });

        if (found != dataset->frames.end()) {
          create_binary_factor(*found, frame_i);
        } else {
          create_unary_factor(keyframe, frame_i);
        }
      }
    }
  }

  /// @brief Verify the linearization result
  void verify(std::ostream& log_os, const boost::program_options::variables_map& vm) {
    int verify_count = 0;

    const auto verify_log_path = [&] {
      std::string filename = dataset->data_path.string();
      std::transform(filename.begin(), filename.end(), filename.begin(), [](unsigned char c) { return c == '/' ? '_' : c; });
      return vm["verification_log_path"].as<std::string>() + "/" + filename + "." + std::to_string(verify_count++);
    };

    const auto verify = [&](const gtsam::Values& values) {
      if (!vm.count("verify") && !vm.count("save_verification_log")) {
        return;
      }

      auto factors_gpu = gtsam_points::create_nonlinear_factor_set_gpu();
      factors_gpu->add(graph);
      factors_gpu->linearize(values);
      auto linearized = graph.linearize(values);

      const auto log_path = verify_log_path();
      if (vm.count("save_verification_log")) {
        std::cout << "saving verification log to " << log_path << std::endl;
        gtsam::serializeToBinaryFile(*linearized, log_path);
      }

      if (vm.count("verify")) {
        std::cout << "loading verification log from " << log_path << std::endl;
        gtsam::GaussianFactorGraph logged;
        gtsam::deserializeFromBinaryFile(log_path, logged);

        if (logged.size() != linearized->size()) {
          std::cerr << "verification failed!" << std::endl;
          std::cerr << "size mismatch: logged=" << logged.size() << " linearized=" << linearized->size() << std::endl;
        } else {
          for (int i = 0; i < logged.size(); i++) {
            const gtsam::Matrix inf_log = logged[i]->augmentedInformation();
            const gtsam::Matrix inf_lin = linearized->at(i)->augmentedInformation();

            const gtsam::Matrix err = (inf_log - inf_lin).cwiseAbs();
            const gtsam::Matrix rerr = err.array() / (inf_log.array().abs() + 1e-6);

            if (rerr.maxCoeff() > 1e-3) {
              std::cerr << "verification failed!" << std::endl;
              std::cerr << "*** logged ***" << std::endl << inf_log << std::endl;
              std::cerr << "*** linearized ***" << std::endl << inf_lin << std::endl;
              std::cerr << "*** rerr ***" << std::endl << rerr << std::endl;
            }
          }
        }
      }
    };
    verify(values);
  }

  /// @brief Optimize the graph using Levenberg-Marquardt algorithm
  void optimize(std::ostream& log_os, const boost::program_options::variables_map& vm) {
    gtsam_points::LevenbergMarquardtExtParams lm_params;
    lm_params.setMaxIterations(20);

    lm_params.status_msg_callback = [&](const std::string& msg) {
      log_os << msg << std::endl;
      std::cout << msg << std::endl;
    };
    lm_params.callback = [this](const gtsam_points::LevenbergMarquardtOptimizationStatus& lm_status, const gtsam::Values& values) {
      if (!guik::running()) {
        return;
      }
      auto viewer = guik::viewer();
      for (const auto& frame : dataset->frames) {
        const auto drawable = viewer->find_drawable("frame_" + std::to_string(frame->id));
        drawable.first->set_model_matrix(values.at<gtsam::Pose3>(frame->id).matrix());
      }
      viewer->spin_once();
    };

    const auto t1 = std::chrono::high_resolution_clock::now();
    gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();
    const auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "total=" << std::chrono::duration<double>(t2 - t1).count() * 1e3 << "msec" << std::endl;
    log_os << "total=" << std::chrono::duration<double>(t2 - t1).count() * 1e3 << "msec" << std::endl;
  }

private:
  Dataset::ConstPtr dataset;                                             ///< Dataset
  std::unique_ptr<gtsam_points::StreamTempBufferRoundRobin> roundrobin;  ///< Stream temp buffer round robin

  gtsam::Values values;               ///< Initial sensor poses
  gtsam::NonlinearFactorGraph graph;  ///< Nonlinear factor graph
};

/// @brief Entry point
int main(int argc, char** argv) {
  // program options
  using namespace boost::program_options;

  options_description desc("Options");
  desc.add_options()                                                                                                                 //
    ("help,h", "Help")                                                                                                               //
    ("dataset_path", value<std::string>(), "Dataset path")                                                                           //
    ("log_dst_path", value<std::string>()->default_value("/tmp/benchmark.txt"), "Destimation path to write benchmark results")       //
    ("verification_log_path", value<std::string>()->default_value("/tmp/verify_log"), "Destimation path to write verification log")  //
    ("save_verification_log", "Save verification log")                                                                               //
    ("verify", "Verify linearization results")                                                                                       //
    ("visualize,v", "Enable visualization.")                                                                                         //
    ;

  positional_options_description p;
  p.add("dataset_path", 1);
  p.add("log_dst_path", 1);

  variables_map vm;
  store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  // Register NonlinearFactorSetGPU for batch linearization
  std::cout << "registering NonlinearFactorSetGPU..." << std::endl;
  gtsam_points::LinearizationHook::register_hook([] { return gtsam_points::create_nonlinear_factor_set_gpu(); });

  const std::string dataset_path = vm["dataset_path"].as<std::string>();

  // Read noise
  NoiseFeeder noise;
  noise.read(dataset_path + "/noise.txt");

  // Find datasets in dataset_path
  std::vector<std::filesystem::path> paths;
  for (const auto& path : std::filesystem::recursive_directory_iterator(dataset_path)) {
    if (path.path().filename() != "data.txt") {
      continue;
    }
    paths.emplace_back(path.path());
  }
  std::sort(paths.begin(), paths.end());

  // Read datasets
  std::vector<Dataset::Ptr> datasets(paths.size());
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < paths.size(); i++) {
    datasets[i] = std::make_shared<Dataset>(paths[i]);
  }

  if (vm.count("visualize")) {
    auto viewer = guik::viewer();
    viewer->disable_vsync();
  }

  if (vm.count("save_verification_log")) {
    std::filesystem::create_directories(vm["verification_log_path"].as<std::string>());
  }

  // Run benchmark
  std::ofstream ofs(vm["log_dst_path"].as<std::string>());
  for (const auto& dataset : datasets) {
    Benchmark benchmark(dataset, noise, ofs, vm);
  }

  return 0;
}