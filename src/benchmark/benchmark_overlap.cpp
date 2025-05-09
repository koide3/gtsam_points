#include <benchmark/benchmark_overlap.hpp>

#include <chrono>
#include <gtsam/geometry/Pose3.h>
#include <guik/viewer/light_viewer.hpp>

BenchmarkOverlap::BenchmarkOverlap(
  const Dataset::ConstPtr& dataset,
  NoiseFeeder& noise,
  std::ostream& log_os,
  const boost::program_options::variables_map& vm)
: dataset(dataset),
  verification_failed(false) {
  std::cout << "dataset=" << dataset->data_path.string() << std::endl;
  log_os << "dataset=" << dataset->data_path.string() << std::endl;

  create_pairs(noise);
  bench(log_os, vm);
}

void BenchmarkOverlap::create_pairs(NoiseFeeder& noise) {
  for (int i = 0; i < dataset->frames.size(); i++) {
    for (int j = i - 2; 0 < j && j < i; j++) {
      const auto& frame_i = dataset->frames[i];
      const auto& frame_j = dataset->frames[j];

      // Add noise to the groundtruth pose
      const Eigen::Vector3d tvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.1;
      const Eigen::Vector3d rvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.025;
      const gtsam::Pose3 noise(gtsam::Rot3::Expmap(rvec), tvec);

      const Eigen::Isometry3d delta = frame_j->pose.inverse() * frame_i->pose * Eigen::Isometry3d(noise.matrix());

      pairwise_voxelmaps.emplace_back(frame_j->voxelmaps.back());
      pairwise_points.emplace_back(frame_i->points);
      pairwise_deltas.emplace_back(delta);

      const double overlap = gtsam_points::overlap(frame_j->voxelmaps_cpu.back(), frame_i->points, delta);
      pairwise_overlaps.emplace_back(overlap);
    }

    for (const auto& keyframe : dataset->keyframes) {
      const auto& frame_i = dataset->frames[i];
      const auto& frame_j = keyframe;
      // Add noise to the groundtruth pose
      const Eigen::Vector3d tvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.1;
      const Eigen::Vector3d rvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.025;
      const gtsam::Pose3 noise(gtsam::Rot3::Expmap(rvec), tvec);

      const Eigen::Isometry3d delta = frame_j->pose.inverse() * frame_i->pose * Eigen::Isometry3d(noise.matrix());

      pairwise_voxelmaps.emplace_back(frame_j->voxelmaps.back());
      pairwise_points.emplace_back(frame_i->points);
      pairwise_deltas.emplace_back(delta);

      const double overlap = gtsam_points::overlap(frame_j->voxelmaps_cpu.back(), frame_i->points, delta);
      pairwise_overlaps.emplace_back(overlap);
    }
  }

  for (int i = 0; i < dataset->frames.size(); i++) {
    std::vector<gtsam_points::GaussianVoxelMap::ConstPtr> voxelmaps_gpu;
    std::vector<gtsam_points::GaussianVoxelMap::ConstPtr> voxelmaps_cpu;
    std::vector<Eigen::Isometry3d> deltas;

    for (int j = i - 2; 0 < j && j < i; j++) {
      const auto& frame_i = dataset->frames[i];
      const auto& frame_j = dataset->frames[j];

      // Add noise to the groundtruth pose
      const Eigen::Vector3d tvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.1;
      const Eigen::Vector3d rvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.025;
      const gtsam::Pose3 noise(gtsam::Rot3::Expmap(rvec), tvec);

      const Eigen::Isometry3d delta = frame_j->pose.inverse() * frame_i->pose * Eigen::Isometry3d(noise.matrix());
      deltas.emplace_back(delta);
      voxelmaps_gpu.emplace_back(frame_j->voxelmaps.back());
      voxelmaps_cpu.emplace_back(frame_j->voxelmaps_cpu.back());
    }

    for (const auto& keyframe : dataset->keyframes) {
      const auto& frame_i = dataset->frames[i];
      const auto& frame_j = keyframe;
      // Add noise to the groundtruth pose
      const Eigen::Vector3d tvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.1;
      const Eigen::Vector3d rvec = Eigen::Vector3d(noise(), noise(), noise()) * 0.025;
      const gtsam::Pose3 noise(gtsam::Rot3::Expmap(rvec), tvec);

      const Eigen::Isometry3d delta = frame_j->pose.inverse() * frame_i->pose * Eigen::Isometry3d(noise.matrix());

      deltas.emplace_back(delta);
      voxelmaps_gpu.emplace_back(frame_j->voxelmaps.back());
      voxelmaps_cpu.emplace_back(frame_j->voxelmaps_cpu.back());
    }

    set_voxelmaps.emplace_back(voxelmaps_gpu);
    set_points.emplace_back(dataset->frames[i]->points);
    set_deltas.emplace_back(deltas);

    const double overlap = gtsam_points::overlap(voxelmaps_cpu, dataset->frames[i]->points, deltas);
    set_overlaps.emplace_back(overlap);
  }
}

void BenchmarkOverlap::bench(std::ostream& log_os, const boost::program_options::variables_map& vm) {
  std::vector<double> overlaps(pairwise_points.size());

  const auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < pairwise_points.size(); i++) {
    overlaps[i] = gtsam_points::overlap_gpu(pairwise_voxelmaps[i], pairwise_points[i], pairwise_deltas[i], stream);
  }
  const auto t2 = std::chrono::high_resolution_clock::now();

  if (vm.count("verify")) {
    for (int i = 0; i < pairwise_points.size(); i++) {
      const double err = std::abs(overlaps[i] - pairwise_overlaps[i]);
      if (err > 1e-3) {
        std::cerr << "verification failed!" << std::endl;
        std::cerr << "overlap=" << overlaps[i] << " pairwise_overlap=" << pairwise_overlaps[i] << std::endl;
        verification_failed = true;
      }
    }
  }

  std::cout << "pairwise=" << std::chrono::duration<double>(t2 - t1).count() * 1e3 << "msec" << std::endl;
  log_os << "pairwise=" << std::chrono::duration<double>(t2 - t1).count() * 1e3 << "msec" << std::endl;

  overlaps.clear();
  const auto t3 = std::chrono::high_resolution_clock::now();
  overlaps = gtsam_points::overlap_gpu(pairwise_voxelmaps, pairwise_points, pairwise_deltas, stream);
  const auto t4 = std::chrono::high_resolution_clock::now();

  if (vm.count("verify")) {
    for (int i = 0; i < pairwise_points.size(); i++) {
      const double err = std::abs(overlaps[i] - pairwise_overlaps[i]);
      if (err > 1e-3) {
        std::cerr << "verification failed!" << std::endl;
        std::cerr << "overlap=" << overlaps[i] << " pairwise_overlap=" << pairwise_overlaps[i] << std::endl;
        verification_failed = true;
      }
    }
  }

  std::cout << "pairwise_batch=" << std::chrono::duration<double>(t4 - t3).count() * 1e3 << "msec" << std::endl;
  log_os << "pairwise_batch=" << std::chrono::duration<double>(t4 - t3).count() * 1e3 << "msec" << std::endl;

  overlaps.clear();
  const auto t5 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < set_points.size(); i++) {
    overlaps[i] = gtsam_points::overlap_gpu(set_voxelmaps[i], set_points[i], set_deltas[i], stream);
  }
  const auto t6 = std::chrono::high_resolution_clock::now();

  if (vm.count("verify")) {
    for (int i = 0; i < set_points.size(); i++) {
      const double err = std::abs(overlaps[i] - set_overlaps[i]);
      if (err > 1e-3) {
        std::cerr << "verification failed!" << std::endl;
        std::cerr << "overlap=" << overlaps[i] << " set_overlap=" << set_overlaps[i] << std::endl;
        verification_failed = true;
      }
    }
  }

  std::cout << "set=" << std::chrono::duration<double>(t6 - t5).count() * 1e3 << "msec" << std::endl;
  log_os << "set=" << std::chrono::duration<double>(t6 - t5).count() * 1e3 << "msec" << std::endl;
}
