#include <benchmark/benchmark.hpp>

#include <gtsam/base/serialization.h>

#include <gtsam/base/make_shared.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#include <guik/viewer/light_viewer.hpp>

Benchmark::Benchmark(const Dataset::ConstPtr& dataset, NoiseFeeder& noise, std::ostream& log_os, const boost::program_options::variables_map& vm)
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

void Benchmark::create_graph(double noise_t, double noise_r, NoiseFeeder& noise) {
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

void Benchmark::verify(std::ostream& log_os, const boost::program_options::variables_map& vm) {
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

          if (rerr.maxCoeff() > 1e-2) {
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

void Benchmark::optimize(std::ostream& log_os, const boost::program_options::variables_map& vm) {
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
