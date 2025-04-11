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
#include <benchmark/benchmark.hpp>

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