#define FMT_HEADER_ONLY

#include <fstream>
#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include <boost/program_options.hpp>

#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>

#include <guik/viewer/light_viewer.hpp>

#include <benchmark/dataset.hpp>
#include <benchmark/noise_feeder.hpp>
#include <benchmark/benchmark_overlap.hpp>

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
    ("recreate_voxelmaps", "Recreating voxelmaps (This will overwrite voxelmap data in the input directory)")                        //
    ("sequential", "Sequantially load and evaluate datasets one-by-one (to save memory)")                                            //
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

  if (vm.count("visualize")) {
    auto viewer = guik::viewer();
    viewer->disable_vsync();
  }

  if (vm.count("save_verification_log")) {
    std::filesystem::create_directories(vm["verification_log_path"].as<std::string>());
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

  int verification_failed_count = 0;
  std::ofstream ofs(vm["log_dst_path"].as<std::string>());

  if (!vm.count("sequential")) {
    // Read datasets in parallel
    std::vector<Dataset::Ptr> datasets(paths.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < paths.size(); i++) {
      datasets[i] = std::make_shared<Dataset>(paths[i], vm.count("recreate_voxelmaps"));
    }

    // Run benchmark sequentially
    for (int i = 0; i < paths.size(); i++) {
      const auto& dataset = datasets[i];
      BenchmarkOverlap benchmark(dataset, noise, ofs, vm);
      verification_failed_count += benchmark.is_verification_failed();
    }
  } else {
    for (int i = 0; i < paths.size(); i++) {
      const auto dataset = std::make_shared<Dataset>(paths[i], vm.count("recreate_voxelmaps"));
      BenchmarkOverlap benchmark(dataset, noise, ofs, vm);
      verification_failed_count += benchmark.is_verification_failed();
    }
  }

  std::cout << "verification failed " << verification_failed_count << " / " << paths.size() << std::endl;

  return 0;
}